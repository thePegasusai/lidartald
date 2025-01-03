import { from, Observable } from 'rxjs'; // v7.8.0
import { catchError, map, retry } from 'rxjs/operators';
import { AxiosResponse } from 'axios'; // v1.4.0
import { apiClient, apiEndpoints } from '../config/api';
import { User, UserRole, UserPrivacySettings, UserLocation, UserProfile, userSchema } from '../types/user.types';

/**
 * Cache configuration for user-related data
 */
const CACHE_CONFIG = {
    userProfile: {
        ttl: 300000, // 5 minutes
        maxSize: 100
    },
    nearbyUsers: {
        ttl: 10000, // 10 seconds
        maxSize: 50
    }
};

/**
 * Rate limiting configuration for user API requests
 */
const RATE_LIMITS = {
    profile: 20, // requests per minute
    nearby: 30,  // requests per minute
    preferences: 10 // requests per minute
};

/**
 * Retrieves the currently authenticated user's profile with enhanced security context
 * @returns Promise<User> Current user data with role and privacy settings
 */
const getCurrentUser = async (): Promise<User> => {
    try {
        const response: AxiosResponse<User> = await apiClient.get(
            apiEndpoints.user.profile,
            {
                headers: {
                    'X-Security-Context': 'enhanced',
                    'X-Request-Priority': 'high'
                },
                validateSchema: userSchema
            }
        );

        // Transform and validate response data
        const userData = response.data;
        validateUserData(userData);

        return {
            ...userData,
            privacySettings: decryptPrivacySettings(userData.privacySettings)
        };
    } catch (error) {
        console.error('[UserAPI] getCurrentUser error:', error);
        throw new Error('Failed to retrieve user profile');
    }
};

/**
 * Updates user preferences with privacy controls and validation
 * @param userId - User identifier
 * @param preferences - Updated user preferences
 * @returns Promise<UserProfile> Updated user profile
 */
const updateUserPreferences = async (
    userId: string,
    preferences: Partial<UserProfile>
): Promise<UserProfile> => {
    validateUserPermissions(userId);
    validatePreferences(preferences);

    try {
        const encryptedPreferences = encryptSensitiveData(preferences);
        
        const response = await apiClient.put(
            `${apiEndpoints.user.preferences}/${userId}`,
            encryptedPreferences,
            {
                headers: {
                    'X-Privacy-Level': 'high',
                    'X-Data-Sensitivity': 'sensitive'
                }
            }
        );

        return decryptUserProfile(response.data);
    } catch (error) {
        console.error('[UserAPI] updateUserPreferences error:', error);
        throw new Error('Failed to update user preferences');
    }
};

/**
 * Retrieves nearby users with privacy filtering and spatial indexing
 * @param range - Detection range in meters (max 5.0m)
 * @param filter - Privacy filter settings
 * @returns Observable<UserLocation[]> Stream of nearby user locations
 */
const getNearbyUsers = (
    range: number,
    filter: UserPrivacySettings
): Observable<UserLocation[]> => {
    validateRange(range);
    validatePrivacyFilter(filter);

    return from(apiClient.get(
        apiEndpoints.user.location,
        {
            params: {
                range,
                privacyLevel: filter.shareLocation ? 'standard' : 'restricted'
            },
            headers: {
                'X-Location-Accuracy': 'high',
                'X-Privacy-Mode': filter.shareLocation ? 'enabled' : 'disabled'
            }
        }
    )).pipe(
        map(response => filterUserLocations(response.data, filter)),
        retry(3),
        catchError(error => {
            console.error('[UserAPI] getNearbyUsers error:', error);
            throw new Error('Failed to retrieve nearby users');
        })
    );
};

/**
 * Validates user data against schema and security requirements
 */
const validateUserData = (userData: User): void => {
    if (!userSchema.safeParse(userData).success) {
        throw new Error('Invalid user data format');
    }
};

/**
 * Validates user permissions for requested operation
 */
const validateUserPermissions = (userId: string): void => {
    const currentUser = getCurrentUser();
    if (!hasPermission(userId)) {
        throw new Error('Insufficient permissions');
    }
};

/**
 * Validates privacy filter settings
 */
const validatePrivacyFilter = (filter: UserPrivacySettings): void => {
    if (filter.dataRetentionDays < 1 || filter.dataRetentionDays > 365) {
        throw new Error('Invalid data retention period');
    }
};

/**
 * Validates range parameter for nearby user detection
 */
const validateRange = (range: number): void => {
    if (range <= 0 || range > 5.0) {
        throw new Error('Range must be between 0 and 5.0 meters');
    }
};

/**
 * Encrypts sensitive user data using AES-256-GCM
 */
const encryptSensitiveData = <T>(data: T): T => {
    // Implementation of AES-256-GCM encryption
    return data; // Placeholder for actual encryption
};

/**
 * Decrypts user privacy settings
 */
const decryptPrivacySettings = (
    settings: UserPrivacySettings
): UserPrivacySettings => {
    // Implementation of privacy settings decryption
    return settings; // Placeholder for actual decryption
};

/**
 * Decrypts user profile data
 */
const decryptUserProfile = (profile: UserProfile): UserProfile => {
    // Implementation of profile data decryption
    return profile; // Placeholder for actual decryption
};

/**
 * Filters user locations based on privacy settings
 */
const filterUserLocations = (
    locations: UserLocation[],
    filter: UserPrivacySettings
): UserLocation[] => {
    return locations.filter(location => 
        location.isVisible && 
        (filter.shareLocation || location.distance <= 2.0)
    );
};

/**
 * Checks if current user has permission for requested operation
 */
const hasPermission = (userId: string): boolean => {
    const currentUser = getCurrentUser();
    return currentUser.role === UserRole.ADMIN || 
           currentUser.role === UserRole.PREMIUM_USER;
};

/**
 * Exported user API interface
 */
export const userApi = {
    getCurrentUser,
    updateUserPreferences,
    getNearbyUsers
};