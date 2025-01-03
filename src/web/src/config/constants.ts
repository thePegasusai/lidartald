import { SurfaceClassification } from '../types/environment.types';
import { FleetStatus } from '../types/fleet.types';

/**
 * LiDAR scanning and processing constants
 * Based on hardware specifications: 30Hz scan rate, 0.01cm resolution, 5m range
 */
export const LIDAR_CONSTANTS = {
    SCAN_RATE_HZ: 30,
    MIN_RESOLUTION_CM: 0.01,
    MAX_RANGE_M: 5.0,
    MIN_FEATURE_CONFIDENCE: 0.85,
    POINT_CLOUD_UPDATE_INTERVAL_MS: 33, // ~30Hz refresh rate
    VALIDATION_RANGES: {
        SCAN_RATE_HZ: [20, 60],
        MIN_RESOLUTION_CM: [0.01, 0.1],
        MAX_RANGE_M: [1.0, 10.0]
    }
} as const;

/**
 * Fleet ecosystem constants
 * Supports up to 32 connected devices with real-time synchronization
 */
export const FLEET_CONSTANTS = {
    MAX_DEVICES: 32,
    SYNC_INTERVAL_MS: 50, // <50ms network latency requirement
    CONNECTION_TIMEOUT_MS: 5000,
    HEARTBEAT_INTERVAL_MS: 1000,
    MAX_RETRY_ATTEMPTS: 3,
    VALID_STATUSES: [FleetStatus.ACTIVE, FleetStatus.SYNCING],
    VALIDATION_RANGES: {
        MAX_DEVICES: [1, 32],
        SYNC_INTERVAL_MS: [20, 100],
        CONNECTION_TIMEOUT_MS: [1000, 10000]
    }
} as const;

/**
 * Environment mapping and feature detection constants
 * Optimized for real-time surface classification and feature tracking
 */
export const ENVIRONMENT_CONSTANTS = {
    MAX_MAP_SIZE_M2: 25,
    MIN_SURFACE_POINTS: 10,
    MAX_FEATURES_PER_SCAN: 100,
    FEATURE_UPDATE_INTERVAL_MS: 100,
    VALID_SURFACES: [
        SurfaceClassification.Floor,
        SurfaceClassification.Wall
    ],
    VALIDATION_RANGES: {
        MAX_MAP_SIZE_M2: [10, 50],
        MIN_SURFACE_POINTS: [5, 20],
        MAX_FEATURES_PER_SCAN: [50, 200]
    }
} as const;

/**
 * UI rendering and animation constants
 * Targets 60 FPS with optimized animation timings
 */
export const UI_CONSTANTS = {
    TARGET_FPS: 60,
    ANIMATION_DURATION_MS: 300,
    TOOLTIP_DELAY_MS: 200,
    MAX_NOTIFICATIONS: 5,
    ALERT_TIMEOUT_MS: 3000,
    VALIDATION_RANGES: {
        TARGET_FPS: [30, 120],
        ANIMATION_DURATION_MS: [100, 1000],
        MAX_NOTIFICATIONS: [1, 10]
    }
} as const;

/**
 * API communication constants
 * Configured for optimal network performance and reliability
 */
export const API_CONSTANTS = {
    BASE_URL: '/api/v1',
    TIMEOUT_MS: 5000,
    MAX_RETRIES: 3,
    RETRY_DELAY_MS: 1000,
    VALIDATION_RANGES: {
        TIMEOUT_MS: [1000, 10000],
        MAX_RETRIES: [1, 5],
        RETRY_DELAY_MS: [500, 5000]
    }
} as const;

/**
 * WebSocket connection constants
 * Optimized for real-time fleet communication
 */
export const WEBSOCKET_CONSTANTS = {
    RECONNECT_INTERVAL_MS: 1000,
    MAX_RECONNECT_ATTEMPTS: 5,
    PING_INTERVAL_MS: 30000,
    VALIDATION_RANGES: {
        RECONNECT_INTERVAL_MS: [500, 5000],
        MAX_RECONNECT_ATTEMPTS: [3, 10],
        PING_INTERVAL_MS: [15000, 60000]
    }
} as const;

/**
 * Type definitions for validation ranges
 * Ensures type safety when accessing validation boundaries
 */
type ValidationRanges<T> = {
    [K in keyof T]: [number, number];
};

/**
 * Type-safe validation range accessors
 */
export type LidarValidationRanges = ValidationRanges<typeof LIDAR_CONSTANTS.VALIDATION_RANGES>;
export type FleetValidationRanges = ValidationRanges<typeof FLEET_CONSTANTS.VALIDATION_RANGES>;
export type EnvironmentValidationRanges = ValidationRanges<typeof ENVIRONMENT_CONSTANTS.VALIDATION_RANGES>;
export type UIValidationRanges = ValidationRanges<typeof UI_CONSTANTS.VALIDATION_RANGES>;