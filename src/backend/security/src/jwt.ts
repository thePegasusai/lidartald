import { injectable } from 'tsyringe';
import jwt from 'jsonwebtoken'; // v9.0.0
import ms from 'ms'; // v2.1.3
import NodeCache from 'node-cache'; // v5.1.2
import winston from 'winston'; // v3.8.2
import { EncryptionService } from './encryption';

// Constants
const JWT_SECRET = process.env.JWT_SECRET || 'your-256-bit-secret';
const ACCESS_TOKEN_EXPIRY = '1h';
const REFRESH_TOKEN_EXPIRY = '7d';
const TOKEN_RATE_LIMIT = 100;
const TOKEN_CACHE_TTL = 300; // 5 minutes

// Types
interface TokenPayload {
    userId: string;
    roles: string[];
    permissions: string[];
    deviceId: string;
    fingerprint: string;
    iat?: number;
    exp?: number;
    nbf?: number;
}

interface JWTOptions {
    expiresIn?: string;
    audience?: string;
    issuer?: string;
    subject?: string;
    algorithm?: jwt.Algorithm;
}

interface DeviceInfo {
    deviceId: string;
    hardwareId: string;
    firmwareVersion: string;
    securityLevel: number;
}

// Decorators
function RateLimit(limit: number) {
    return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
        const original = descriptor.value;
        let tokens = limit;
        let lastReset = Date.now();

        descriptor.value = async function (...args: any[]) {
            const now = Date.now();
            if (now - lastReset >= 60000) {
                tokens = limit;
                lastReset = now;
            }
            if (tokens <= 0) {
                throw new Error('Rate limit exceeded for token generation');
            }
            tokens--;
            return original.apply(this, args);
        };
        return descriptor;
    };
}

function SecurityAudit(target: any) {
    // Add security audit logging to all methods
    for (const propertyName of Object.getOwnPropertyNames(target.prototype)) {
        const descriptor = Object.getOwnPropertyDescriptor(target.prototype, propertyName);
        const isMethod = descriptor?.value instanceof Function;
        if (!isMethod) continue;

        const originalMethod = descriptor?.value;
        descriptor!.value = async function (...args: any[]) {
            try {
                const result = await originalMethod.apply(this, args);
                this.logger.info(`Security operation completed: ${propertyName}`, {
                    operation: propertyName,
                    status: 'success'
                });
                return result;
            } catch (error) {
                this.logger.error(`Security operation failed: ${propertyName}`, {
                    operation: propertyName,
                    error: error.message,
                    status: 'failure'
                });
                throw error;
            }
        };
        Object.defineProperty(target.prototype, propertyName, descriptor!);
    }
}

@RateLimit(TOKEN_RATE_LIMIT)
export function generateTokenPayload(user: any, deviceInfo: DeviceInfo): TokenPayload {
    const fingerprint = generateDeviceFingerprint(deviceInfo);
    
    return {
        userId: user.id,
        roles: user.roles,
        permissions: user.permissions,
        deviceId: deviceInfo.deviceId,
        fingerprint,
        nbf: Math.floor(Date.now() / 1000),
    };
}

function generateDeviceFingerprint(deviceInfo: DeviceInfo): string {
    const data = `${deviceInfo.deviceId}:${deviceInfo.hardwareId}:${deviceInfo.firmwareVersion}`;
    return Buffer.from(data).toString('base64');
}

@injectable()
@SecurityAudit
export class JWTService {
    private readonly logger: winston.Logger;
    private readonly tokenCache: NodeCache;
    private readonly revokedTokens: Set<string>;

    constructor(private readonly encryptionService: EncryptionService) {
        this.logger = winston.createLogger({
            level: 'info',
            format: winston.format.json(),
            transports: [
                new winston.transports.File({ filename: 'security.log' })
            ]
        });

        this.tokenCache = new NodeCache({
            stdTTL: TOKEN_CACHE_TTL,
            checkperiod: 60,
            useClones: false
        });

        this.revokedTokens = new Set<string>();
    }

    public async signToken(
        payload: TokenPayload,
        options: JWTOptions = {}
    ): Promise<string> {
        try {
            // Encrypt sensitive claims
            const sensitiveData = {
                roles: payload.roles,
                permissions: payload.permissions
            };
            
            const encryptedClaims = await this.encryptionService.encryptData(
                JSON.stringify(sensitiveData),
                payload.userId
            );

            const finalPayload = {
                ...payload,
                enc: encryptedClaims,
                iat: Math.floor(Date.now() / 1000)
            };

            const token = jwt.sign(finalPayload, JWT_SECRET, {
                expiresIn: options.expiresIn || ACCESS_TOKEN_EXPIRY,
                algorithm: 'HS512',
                ...options
            });

            // Cache token metadata
            this.tokenCache.set(token, {
                userId: payload.userId,
                deviceId: payload.deviceId,
                fingerprint: payload.fingerprint
            });

            return token;
        } catch (error) {
            this.logger.error('Token signing failed', { error: error.message });
            throw new Error('Failed to sign token');
        }
    }

    public async verifyAndDecode(token: string): Promise<TokenPayload> {
        try {
            // Check revocation
            if (this.revokedTokens.has(token)) {
                throw new Error('Token has been revoked');
            }

            // Verify token
            const decoded = jwt.verify(token, JWT_SECRET, {
                algorithms: ['HS512']
            }) as TokenPayload;

            // Verify cache
            const cached = this.tokenCache.get<{
                userId: string;
                deviceId: string;
                fingerprint: string;
            }>(token);

            if (!cached || 
                cached.userId !== decoded.userId || 
                cached.deviceId !== decoded.deviceId || 
                cached.fingerprint !== decoded.fingerprint) {
                throw new Error('Token validation failed');
            }

            // Decrypt sensitive claims
            if (decoded.enc) {
                const decrypted = await this.encryptionService.decryptData(
                    decoded.enc,
                    decoded.userId
                );
                const sensitiveData = JSON.parse(decrypted.toString());
                decoded.roles = sensitiveData.roles;
                decoded.permissions = sensitiveData.permissions;
                delete decoded.enc;
            }

            return decoded;
        } catch (error) {
            this.logger.error('Token verification failed', { error: error.message });
            throw new Error('Invalid token');
        }
    }

    public async refreshToken(token: string): Promise<string> {
        try {
            const decoded = await this.verifyAndDecode(token);
            const newPayload = {
                ...decoded,
                iat: Math.floor(Date.now() / 1000)
            };

            return this.signToken(newPayload, {
                expiresIn: REFRESH_TOKEN_EXPIRY
            });
        } catch (error) {
            this.logger.error('Token refresh failed', { error: error.message });
            throw new Error('Failed to refresh token');
        }
    }

    public async revokeToken(token: string): Promise<void> {
        try {
            const decoded = await this.verifyAndDecode(token);
            this.revokedTokens.add(token);
            this.tokenCache.del(token);
            
            this.logger.info('Token revoked', {
                userId: decoded.userId,
                deviceId: decoded.deviceId
            });
        } catch (error) {
            this.logger.error('Token revocation failed', { error: error.message });
            throw new Error('Failed to revoke token');
        }
    }
}