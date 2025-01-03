import { ENVIRONMENT_CONSTANTS } from './constants';
import { SurfaceClassification } from '../types/environment.types';
import * as THREE from 'three'; // v0.150.0

/**
 * Environment scan settings based on LiDAR hardware specifications
 * Enforces 30Hz scan rate, 0.01cm resolution, and 5m range requirements
 */
export const scanSettings = {
    RATE_HZ: 30,
    RESOLUTION_CM: 0.01,
    RANGE_M: 5.0,
    MIN_SCAN_RATE_HZ: 30,
    MAX_RESOLUTION_CM: 0.01,
    MAX_RANGE_M: 5.0,
    SCAN_BUFFER_SIZE: 1024,
    SCAN_TIMEOUT_MS: 100
} as const;

/**
 * Feature detection configuration for real-time environment processing
 * Optimized for 30Hz update rate with high-confidence detection
 */
export const featureDetection = {
    MIN_CONFIDENCE: 0.85,
    UPDATE_INTERVAL_MS: 33, // ~30Hz to match scan rate
    MAX_FEATURES: ENVIRONMENT_CONSTANTS.MAX_FEATURES_PER_SCAN,
    FEATURE_TYPES: [
        SurfaceClassification.Floor,
        SurfaceClassification.Wall,
        SurfaceClassification.Obstacle
    ],
    DETECTION_MODES: ['FAST', 'ACCURATE'] as const,
    CONFIDENCE_LEVELS: {
        HIGH: 0.95,
        MEDIUM: 0.85,
        LOW: 0.75
    },
    PROCESSING_PRIORITY: 'HIGH' as const
} as const;

/**
 * Environment mapping limits and performance thresholds
 * Configured for optimal real-time processing and memory usage
 */
export const environmentLimits = {
    MAX_AREA_M2: ENVIRONMENT_CONSTANTS.MAX_MAP_SIZE_M2,
    MIN_POINTS_PER_SURFACE: ENVIRONMENT_CONSTANTS.MIN_SURFACE_POINTS,
    MAX_PROCESSING_TIME_MS: 50, // Ensures <50ms latency requirement
    MAX_MEMORY_USAGE_MB: 512,
    MAX_GPU_USAGE_PERCENT: 80,
    DEGRADATION_THRESHOLDS: {
        MEMORY_WARNING: 450, // MB
        GPU_WARNING: 70, // Percent
        PROCESSING_WARNING: 45 // MS
    }
} as const;

/**
 * Environment configuration interface
 */
interface EnvironmentConfig {
    scanSettings: typeof scanSettings;
    featureDetection: typeof featureDetection;
    environmentLimits: typeof environmentLimits;
}

/**
 * Validates environment configuration against system requirements
 * @param config Configuration object to validate
 * @returns boolean indicating if configuration is valid
 */
export function validateEnvironmentConfig(config: Record<string, any>): boolean {
    try {
        // Validate scan rate meets minimum requirement
        if (config.scanSettings?.RATE_HZ < scanSettings.MIN_SCAN_RATE_HZ) {
            return false;
        }

        // Validate resolution meets precision requirement
        if (config.scanSettings?.RESOLUTION_CM > scanSettings.MAX_RESOLUTION_CM) {
            return false;
        }

        // Validate range is within hardware limits
        if (config.scanSettings?.RANGE_M > scanSettings.MAX_RANGE_M) {
            return false;
        }

        // Validate feature detection parameters
        if (config.featureDetection?.MIN_CONFIDENCE < featureDetection.CONFIDENCE_LEVELS.LOW) {
            return false;
        }

        // Validate environment limits
        if (config.environmentLimits?.MAX_AREA_M2 > environmentLimits.MAX_AREA_M2) {
            return false;
        }

        return true;
    } catch (error) {
        console.error('Environment config validation failed:', error);
        return false;
    }
}

/**
 * Retrieves current environment configuration based on system capabilities
 * @returns EnvironmentConfig object with current settings
 */
export function getEnvironmentConfig(): EnvironmentConfig {
    // Create base configuration
    const config: EnvironmentConfig = {
        scanSettings,
        featureDetection,
        environmentLimits
    };

    try {
        // Check GPU capabilities for processing optimization
        const gpu = getGPUCapabilities();
        if (gpu.isLowPower) {
            config.featureDetection.PROCESSING_PRIORITY = 'FAST';
            config.environmentLimits.MAX_GPU_USAGE_PERCENT = 60;
        }

        // Check available memory for buffer sizing
        const memory = getSystemMemory();
        if (memory.available < 1024) { // Less than 1GB
            config.environmentLimits.MAX_MEMORY_USAGE_MB = 256;
            config.scanSettings.SCAN_BUFFER_SIZE = 512;
        }

        return config;
    } catch (error) {
        console.warn('Using default environment config:', error);
        return config;
    }
}

/**
 * Helper function to check GPU capabilities
 * @returns GPU capability information
 */
function getGPUCapabilities(): { isLowPower: boolean } {
    // Implementation would depend on platform-specific GPU detection
    return { isLowPower: false };
}

/**
 * Helper function to check available system memory
 * @returns Memory information in MB
 */
function getSystemMemory(): { available: number } {
    // Implementation would depend on platform-specific memory detection
    return { available: 2048 };
}

// Export default environment configuration
export const environmentConfig = {
    scanSettings,
    featureDetection,
    environmentLimits
} as const;