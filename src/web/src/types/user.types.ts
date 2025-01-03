import { z } from 'zod'; // v3.x
import { FleetParticipant } from './fleet.types';

/**
 * Enumeration of possible user roles with corresponding access levels
 * Based on the authorization matrix from technical specifications
 */
export enum UserRole {
    GUEST = 'GUEST',           // Scan-only access
    BASIC_USER = 'BASIC_USER', // Local + Join Fleet access
    PREMIUM_USER = 'PREMIUM',  // Full local + Fleet creation
    DEVELOPER = 'DEVELOPER',   // Full access + debug tools
    ADMIN = 'ADMIN'           // Complete system access
}

/**
 * Interface defining user privacy settings
 * Implements granular control over data sharing and retention
 */
export interface UserPrivacySettings {
    /** Controls location sharing visibility */
    shareLocation: boolean;
    
    /** Controls activity and status visibility */
    shareActivity: boolean;
    
    /** Controls fleet history visibility */
    shareFleetHistory: boolean;
    
    /** Data retention period in days (1-365) */
    dataRetentionDays: number;
}

/**
 * Interface for user preferences and customization
 */
interface UserPreferences {
    /** UI theme preference */
    theme: 'light' | 'dark' | 'system';
    
    /** Preferred scanning resolution in cm */
    scanResolution: number;
    
    /** Preferred scan range in meters */
    scanRange: number;
    
    /** Notification preferences */
    notifications: {
        fleetInvites: boolean;
        proximityAlerts: boolean;
        achievementAlerts: boolean;
    };
    
    /** Accessibility settings */
    accessibility: {
        highContrast: boolean;
        reducedMotion: boolean;
    };
}

/**
 * Interface for enhanced user location tracking
 * Includes privacy controls and accuracy metrics
 */
export interface UserLocation {
    /** Associated user identifier */
    userId: string;
    
    /** Distance from reference point in meters */
    distance: number;
    
    /** Timestamp of last location update */
    lastUpdate: Date;
    
    /** Location accuracy in meters */
    accuracy: number;
    
    /** Geographical coordinates */
    coordinates: {
        lat: number;
        lng: number;
    };
    
    /** Visibility status based on privacy settings */
    isVisible: boolean;
}

/**
 * Comprehensive user profile interface
 * Contains all user-related data with privacy considerations
 */
export interface UserProfile {
    /** Unique user identifier */
    userId: string;
    
    /** User's display name */
    displayName: string;
    
    /** User's current level */
    level: number;
    
    /** Experience points */
    experience: number;
    
    /** Last activity timestamp */
    lastActive: Date;
    
    /** User preferences */
    preferences: UserPreferences;
    
    /** Array of fleet IDs user has participated in */
    fleetHistory: string[];
    
    /** Array of earned achievement IDs */
    achievementIds: string[];
    
    /** Privacy settings */
    privacySettings: UserPrivacySettings;
}

/**
 * Interface for user session data
 */
interface UserSession {
    /** Session identifier */
    sessionId: string;
    
    /** User role for the session */
    role: UserRole;
    
    /** Session start timestamp */
    startTime: Date;
    
    /** Last activity timestamp */
    lastActivity: Date;
    
    /** Associated device ID */
    deviceId: string;
    
    /** Session authentication token */
    authToken: string;
}

/**
 * Zod validation schema for user data
 * Implements strict validation rules for user-related data
 */
export const userSchema = z.object({
    email: z.string()
        .email('Invalid email format')
        .min(5, 'Email too short')
        .max(255, 'Email too long'),
    
    username: z.string()
        .min(3, 'Username must be at least 3 characters')
        .max(30, 'Username cannot exceed 30 characters')
        .regex(/^[a-zA-Z0-9_-]+$/, 'Username can only contain letters, numbers, underscores, and hyphens'),
    
    password: z.string()
        .min(8, 'Password must be at least 8 characters')
        .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]+$/, 
            'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character'),
    
    privacySettings: z.object({
        shareLocation: z.boolean(),
        shareActivity: z.boolean(),
        shareFleetHistory: z.boolean(),
        dataRetentionDays: z.number()
            .min(1, 'Minimum retention period is 1 day')
            .max(365, 'Maximum retention period is 365 days')
    })
});

/**
 * Type for user creation/update payload validation
 */
export type UserPayload = z.infer<typeof userSchema>;