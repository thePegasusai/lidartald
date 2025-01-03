import { z } from 'zod'; // v3.21.4
import { User } from '../types/user.types';
import { Fleet } from '../types/fleet.types';
import { ScanParameters } from '../types/lidar.types';

// Global validation constants
const EMAIL_REGEX = /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i;
const USERNAME_MIN_LENGTH = 3;
const USERNAME_MAX_LENGTH = 30;
const FLEET_NAME_MIN_LENGTH = 3;
const FLEET_NAME_MAX_LENGTH = 50;
const MIN_FLEET_DEVICES = 2;
const MAX_FLEET_DEVICES = 32;
const MIN_SCAN_RESOLUTION = 0.01; // cm
const MAX_SCAN_RANGE = 5.0; // meters
const MAX_SCAN_RATE = 30; // Hz

/**
 * Schema for validating user privacy settings
 * Implements security classifications from technical specifications
 */
const privacySettingsSchema = z.object({
    shareLocation: z.boolean(),
    shareActivity: z.boolean(),
    shareFleetHistory: z.boolean(),
    dataRetentionDays: z.number()
        .min(1, 'Minimum retention period is 1 day')
        .max(365, 'Maximum retention period is 365 days')
});

/**
 * Schema for validating user profile data
 * Implements comprehensive validation rules with security considerations
 */
const userProfileSchema = z.object({
    email: z.string()
        .regex(EMAIL_REGEX, 'Invalid email format')
        .min(5, 'Email too short')
        .max(255, 'Email too long')
        .trim(),

    username: z.string()
        .min(USERNAME_MIN_LENGTH, `Username must be at least ${USERNAME_MIN_LENGTH} characters`)
        .max(USERNAME_MAX_LENGTH, `Username cannot exceed ${USERNAME_MAX_LENGTH} characters`)
        .regex(/^[a-zA-Z0-9_-]+$/, 'Username can only contain letters, numbers, underscores, and hyphens')
        .trim(),

    privacySettings: privacySettingsSchema
});

/**
 * Schema for validating fleet configuration
 * Implements mesh networking and device management rules
 */
const fleetConfigurationSchema = z.object({
    name: z.string()
        .min(FLEET_NAME_MIN_LENGTH, `Fleet name must be at least ${FLEET_NAME_MIN_LENGTH} characters`)
        .max(FLEET_NAME_MAX_LENGTH, `Fleet name cannot exceed ${FLEET_NAME_MAX_LENGTH} characters`)
        .regex(/^[a-zA-Z0-9\s-]+$/, 'Fleet name can only contain letters, numbers, spaces, and hyphens')
        .trim(),

    maxDevices: z.number()
        .min(MIN_FLEET_DEVICES, `Fleet must allow at least ${MIN_FLEET_DEVICES} devices`)
        .max(MAX_FLEET_DEVICES, `Fleet cannot exceed ${MAX_FLEET_DEVICES} devices`),

    status: z.enum(['INITIALIZING', 'ACTIVE', 'SYNCING', 'DISCONNECTED']),

    meshNetworkStatus: z.object({
        connected: z.boolean(),
        latency: z.number()
            .min(0, 'Latency cannot be negative')
            .max(1000, 'Latency exceeds maximum threshold')
    })
});

/**
 * Schema for validating LiDAR scan parameters
 * Implements hardware specification constraints
 */
const scanParametersSchema = z.object({
    resolution: z.number()
        .min(MIN_SCAN_RESOLUTION, `Resolution must be at least ${MIN_SCAN_RESOLUTION}cm`)
        .max(1.0, 'Resolution cannot exceed 1.0cm'),

    range: z.number()
        .min(0.1, 'Range must be at least 0.1m')
        .max(MAX_SCAN_RANGE, `Range cannot exceed ${MAX_SCAN_RANGE}m`),

    scanRate: z.number()
        .min(1, 'Scan rate must be at least 1Hz')
        .max(MAX_SCAN_RATE, `Scan rate cannot exceed ${MAX_SCAN_RATE}Hz`)
});

/**
 * Validates user profile data with enhanced privacy settings and security classifications
 * @param profile User profile data to validate
 * @returns True if validation passes, throws ZodError with detailed context if validation fails
 */
export const validateUserProfile = (profile: User): boolean => {
    try {
        userProfileSchema.parse(profile);
        return true;
    } catch (error) {
        console.error('[Validation Error] User Profile:', error);
        throw error;
    }
};

/**
 * Validates fleet configuration with mesh networking and device management rules
 * @param config Fleet configuration to validate
 * @returns True if validation passes, throws ZodError with detailed context if validation fails
 */
export const validateFleetConfiguration = (config: Fleet): boolean => {
    try {
        fleetConfigurationSchema.parse(config);
        return true;
    } catch (error) {
        console.error('[Validation Error] Fleet Configuration:', error);
        throw error;
    }
};

/**
 * Validates LiDAR scanning parameters against hardware specifications
 * @param params Scan parameters to validate
 * @returns True if validation passes, throws ZodError with detailed context if validation fails
 */
export const validateScanParameters = (params: ScanParameters): boolean => {
    try {
        scanParametersSchema.parse(params);
        return true;
    } catch (error) {
        console.error('[Validation Error] Scan Parameters:', error);
        throw error;
    }
};

/**
 * Sanitizes user input to prevent XSS and injection attacks
 * @param input String input to sanitize
 * @returns Sanitized string
 */
const sanitizeInput = (input: string): string => {
    return input
        .replace(/[<>]/g, '') // Remove potential HTML tags
        .replace(/['"]/g, '') // Remove quotes
        .trim();
};

/**
 * Validates and sanitizes general string input
 * @param input String to validate
 * @param minLength Minimum allowed length
 * @param maxLength Maximum allowed length
 * @returns Sanitized string if valid, throws error if invalid
 */
export const validateInput = (
    input: string,
    minLength: number,
    maxLength: number
): string => {
    const sanitized = sanitizeInput(input);
    
    if (sanitized.length < minLength) {
        throw new Error(`Input must be at least ${minLength} characters`);
    }
    
    if (sanitized.length > maxLength) {
        throw new Error(`Input cannot exceed ${maxLength} characters`);
    }
    
    return sanitized;
};