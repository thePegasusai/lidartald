import { injectable } from 'tsyringe';
import Redis from 'ioredis'; // v5.3.2
import { JWTService } from './jwt';

// Role enumeration constants
export const ROLES = {
    GUEST: 'guest',
    BASIC_USER: 'basic_user',
    PREMIUM_USER: 'premium_user',
    DEVELOPER: 'developer',
    ADMIN: 'admin'
} as const;

// Role hierarchy levels (higher number = more access)
const ROLE_HIERARCHY = {
    [ROLES.ADMIN]: 4,
    [ROLES.DEVELOPER]: 3,
    [ROLES.PREMIUM_USER]: 2,
    [ROLES.BASIC_USER]: 1,
    [ROLES.GUEST]: 0
} as const;

// System-wide permissions
export const PERMISSIONS = {
    SCAN: 'scan:execute',
    FLEET_CREATE: 'fleet:create',
    FLEET_JOIN: 'fleet:join',
    PROFILE_READ: 'profile:read',
    PROFILE_WRITE: 'profile:write',
    ADMIN_ACCESS: 'admin:access'
} as const;

// Permission validation utility
export function validatePermission(permission: string): boolean {
    const permissionPattern = /^[a-z]+:[a-z_]+$/;
    if (!permission || typeof permission !== 'string') {
        return false;
    }
    if (!permissionPattern.test(permission)) {
        return false;
    }
    if (permission.length > 50) {
        return false;
    }
    return Object.values(PERMISSIONS).includes(permission as any);
}

// Role-permission mapping type
type RolePermissionMap = {
    [key in typeof ROLES[keyof typeof ROLES]]: Set<string>;
};

@injectable()
export class RBACService {
    private readonly redisClient: Redis;
    private readonly permissionCache: Map<string, Set<string>>;
    private readonly rolePermissions: RolePermissionMap;

    constructor(
        private readonly jwtService: JWTService,
        redisConfig: Redis.RedisOptions
    ) {
        // Initialize Redis client for distributed caching
        this.redisClient = new Redis({
            ...redisConfig,
            retryStrategy: (times: number) => Math.min(times * 50, 2000)
        });

        // Initialize in-memory cache
        this.permissionCache = new Map();

        // Initialize role-permission mappings
        this.rolePermissions = {
            [ROLES.GUEST]: new Set([PERMISSIONS.SCAN]),
            [ROLES.BASIC_USER]: new Set([
                PERMISSIONS.SCAN,
                PERMISSIONS.FLEET_JOIN,
                PERMISSIONS.PROFILE_READ
            ]),
            [ROLES.PREMIUM_USER]: new Set([
                PERMISSIONS.SCAN,
                PERMISSIONS.FLEET_JOIN,
                PERMISSIONS.FLEET_CREATE,
                PERMISSIONS.PROFILE_READ,
                PERMISSIONS.PROFILE_WRITE
            ]),
            [ROLES.DEVELOPER]: new Set([
                PERMISSIONS.SCAN,
                PERMISSIONS.FLEET_JOIN,
                PERMISSIONS.FLEET_CREATE,
                PERMISSIONS.PROFILE_READ,
                PERMISSIONS.PROFILE_WRITE
            ]),
            [ROLES.ADMIN]: new Set([
                PERMISSIONS.SCAN,
                PERMISSIONS.FLEET_JOIN,
                PERMISSIONS.FLEET_CREATE,
                PERMISSIONS.PROFILE_READ,
                PERMISSIONS.PROFILE_WRITE,
                PERMISSIONS.ADMIN_ACCESS
            ])
        };

        // Set up cache invalidation listener
        this.setupCacheInvalidation();
    }

    private setupCacheInvalidation(): void {
        this.redisClient.subscribe('rbac:cache:invalidate');
        this.redisClient.on('message', (channel: string, message: string) => {
            if (channel === 'rbac:cache:invalidate') {
                const { role, permission } = JSON.parse(message);
                this.invalidateCache(role, permission);
            }
        });
    }

    private invalidateCache(role: string, permission?: string): void {
        if (permission) {
            const cached = this.permissionCache.get(role);
            if (cached) {
                cached.delete(permission);
            }
        } else {
            this.permissionCache.delete(role);
        }
    }

    private async cachePermission(role: string, permission: string, hasAccess: boolean): Promise<void> {
        const cacheKey = `rbac:${role}:${permission}`;
        await this.redisClient.setex(cacheKey, 300, hasAccess ? '1' : '0');
        
        const cached = this.permissionCache.get(role) || new Set();
        if (hasAccess) {
            cached.add(permission);
        }
        this.permissionCache.set(role, cached);
    }

    public async hasPermission(role: string, permission: string): Promise<boolean> {
        // Validate inputs
        if (!this.validateRole(role) || !validatePermission(permission)) {
            return false;
        }

        // Check cache first
        const cached = this.permissionCache.get(role);
        if (cached?.has(permission)) {
            return true;
        }

        // Check role hierarchy
        const roleLevel = ROLE_HIERARCHY[role as keyof typeof ROLE_HIERARCHY];
        const hasAccess = Array.from(this.rolePermissions[role as keyof RolePermissionMap])
            .some(p => p === permission);

        // Cache the result
        await this.cachePermission(role, permission, hasAccess);

        return hasAccess;
    }

    public async requirePermission(role: string, requiredPermissions: string[]): Promise<boolean> {
        if (!this.validateRole(role) || !requiredPermissions.every(validatePermission)) {
            return false;
        }

        const permissionChecks = await Promise.all(
            requiredPermissions.map(permission => this.hasPermission(role, permission))
        );

        return permissionChecks.every(Boolean);
    }

    public async getRolePermissions(role: string): Promise<string[]> {
        if (!this.validateRole(role)) {
            return [];
        }

        const roleLevel = ROLE_HIERARCHY[role as keyof typeof ROLE_HIERARCHY];
        const permissions = new Set<string>();

        // Get direct permissions
        const directPermissions = this.rolePermissions[role as keyof RolePermissionMap];
        directPermissions.forEach(p => permissions.add(p));

        // Get inherited permissions from lower roles
        Object.entries(ROLE_HIERARCHY)
            .filter(([_, level]) => level < roleLevel)
            .forEach(([inheritedRole]) => {
                this.rolePermissions[inheritedRole as keyof RolePermissionMap]
                    .forEach(p => permissions.add(p));
            });

        return Array.from(permissions);
    }

    public validateRole(role: string): boolean {
        return Object.values(ROLES).includes(role as any) &&
               typeof ROLE_HIERARCHY[role as keyof typeof ROLE_HIERARCHY] === 'number';
    }

    public async clearCache(): Promise<void> {
        this.permissionCache.clear();
        await this.redisClient.flushdb();
    }
}