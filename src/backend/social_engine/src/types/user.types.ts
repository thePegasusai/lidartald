import { z } from 'zod'; // v3.x
import { ROLES, PERMISSIONS } from '../../security/src/rbac';

/**
 * Core user entity interface representing the base user data structure
 */
export interface User {
    id: string;
    email: string;
    username: string;
    role: typeof ROLES[keyof typeof ROLES];
    permissions: Array<typeof PERMISSIONS[keyof typeof PERMISSIONS]>;
    createdAt: Date;
    updatedAt: Date;
}

/**
 * Extended user profile information including gaming and social features
 */
export interface UserProfile {
    userId: string;
    displayName: string;
    level: number;
    experience: number;
    lastActive: Date;
    preferences: UserPreferences;
    fleetHistory: string[];
    achievements?: string[];
    totalGamesPlayed?: number;
    winRate?: number;
}

/**
 * User device and gameplay preferences configuration
 */
export interface UserPreferences {
    scanResolution: number; // Range: 0.01-1.00
    scanRange: number; // Range: 1-5 meters
    autoJoinFleet: boolean;
    defaultGameMode: string;
    notificationsEnabled?: boolean;
    privacySettings?: {
        profileVisibility: 'public' | 'friends' | 'private';
        locationSharing: boolean;
        fleetDiscoverable: boolean;
    };
}

/**
 * Data transfer object for user creation with required fields
 */
export interface CreateUserDTO {
    email: string;
    username: string;
    password: string;
    initialPreferences?: Partial<UserPreferences>;
}

/**
 * Data transfer object for user profile updates
 */
export interface UpdateUserDTO {
    email?: string;
    username?: string;
    preferences?: Partial<UserPreferences>;
    displayName?: string;
}

/**
 * Zod schema for validating user preferences
 */
export const userPreferencesSchema = z.object({
    scanResolution: z.number()
        .min(0.01, 'Scan resolution must be at least 0.01')
        .max(1, 'Scan resolution cannot exceed 1.00'),
    scanRange: z.number()
        .min(1, 'Scan range must be at least 1 meter')
        .max(5, 'Scan range cannot exceed 5 meters'),
    autoJoinFleet: z.boolean(),
    defaultGameMode: z.string(),
    notificationsEnabled: z.boolean().optional(),
    privacySettings: z.object({
        profileVisibility: z.enum(['public', 'friends', 'private']),
        locationSharing: z.boolean(),
        fleetDiscoverable: z.boolean()
    }).optional()
});

/**
 * Zod schema for validating user creation
 */
export const createUserSchema = z.object({
    email: z.string()
        .email('Invalid email format')
        .min(5, 'Email must be at least 5 characters')
        .max(255, 'Email cannot exceed 255 characters'),
    username: z.string()
        .min(3, 'Username must be at least 3 characters')
        .max(30, 'Username cannot exceed 30 characters')
        .regex(/^[a-zA-Z0-9_-]+$/, 'Username can only contain letters, numbers, underscores, and hyphens'),
    password: z.string()
        .min(8, 'Password must be at least 8 characters')
        .regex(
            /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d]{8,}$/,
            'Password must contain at least one uppercase letter, one lowercase letter, and one number'
        ),
    initialPreferences: userPreferencesSchema.partial().optional()
});

/**
 * Zod schema for validating user profile updates
 */
export const updateUserSchema = z.object({
    email: z.string()
        .email('Invalid email format')
        .min(5, 'Email must be at least 5 characters')
        .max(255, 'Email cannot exceed 255 characters')
        .optional(),
    username: z.string()
        .min(3, 'Username must be at least 3 characters')
        .max(30, 'Username cannot exceed 30 characters')
        .regex(/^[a-zA-Z0-9_-]+$/, 'Username can only contain letters, numbers, underscores, and hyphens')
        .optional(),
    preferences: userPreferencesSchema.partial().optional(),
    displayName: z.string()
        .min(1, 'Display name is required')
        .max(50, 'Display name cannot exceed 50 characters')
        .optional()
});

/**
 * Type guard to check if an object is a valid User
 */
export function isUser(obj: any): obj is User {
    return (
        typeof obj === 'object' &&
        typeof obj.id === 'string' &&
        typeof obj.email === 'string' &&
        typeof obj.username === 'string' &&
        Object.values(ROLES).includes(obj.role) &&
        Array.isArray(obj.permissions) &&
        obj.permissions.every(p => Object.values(PERMISSIONS).includes(p)) &&
        obj.createdAt instanceof Date &&
        obj.updatedAt instanceof Date
    );
}