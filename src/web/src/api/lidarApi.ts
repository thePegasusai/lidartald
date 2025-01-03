import { z } from 'zod'; // v3.21.4
import retry from 'axios-retry'; // v3.5.0
import logger from 'winston'; // v3.9.0
import { apiClient, apiEndpoints } from '../config/api';
import { Point3D } from '../types/lidar.types';
import { LIDAR_CONSTANTS } from '../config/constants';

/**
 * Default scan parameters based on hardware specifications
 */
export const DEFAULT_SCAN_PARAMETERS = {
    resolution: LIDAR_CONSTANTS.MIN_RESOLUTION_CM, // 0.01cm resolution
    range: LIDAR_CONSTANTS.MAX_RANGE_M, // 5.0m range
    scanRate: LIDAR_CONSTANTS.SCAN_RATE_HZ // 30Hz scan rate
} as const;

/**
 * Retry configuration for network resilience
 */
const RETRY_CONFIG = {
    retries: 3,
    retryDelay: 1000,
    retryCondition: 'isNetworkOrIdempotentRequestError'
} as const;

/**
 * Zod schema for scan parameters validation
 */
const ScanParametersSchema = z.object({
    resolution: z.number()
        .min(LIDAR_CONSTANTS.VALIDATION_RANGES.MIN_RESOLUTION_CM[0])
        .max(LIDAR_CONSTANTS.VALIDATION_RANGES.MIN_RESOLUTION_CM[1]),
    range: z.number()
        .min(LIDAR_CONSTANTS.VALIDATION_RANGES.MAX_RANGE_M[0])
        .max(LIDAR_CONSTANTS.VALIDATION_RANGES.MAX_RANGE_M[1]),
    scanRate: z.number()
        .min(LIDAR_CONSTANTS.VALIDATION_RANGES.SCAN_RATE_HZ[0])
        .max(LIDAR_CONSTANTS.VALIDATION_RANGES.SCAN_RATE_HZ[1])
});

/**
 * Zod schema for point cloud validation
 */
const PointCloudSchema = z.object({
    points: z.array(z.object({
        x: z.number(),
        y: z.number(),
        z: z.number(),
        intensity: z.number().min(0).max(1)
    })),
    quality: z.number().min(0).max(1),
    density: z.number().min(0)
});

/**
 * Decorator for parameter validation
 */
function validateScanParameters(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    descriptor.value = function(...args: any[]) {
        const parameters = args[0];
        try {
            ScanParametersSchema.parse(parameters);
        } catch (error) {
            logger.error('Invalid scan parameters', { error, parameters });
            throw new Error('Invalid scan parameters');
        }
        return originalMethod.apply(this, args);
    };
    return descriptor;
}

/**
 * Decorator for retry logic
 */
function withRetry(config: typeof RETRY_CONFIG) {
    return function(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
        const originalMethod = descriptor.value;
        descriptor.value = async function(...args: any[]) {
            try {
                retry(apiClient, config);
                return await originalMethod.apply(this, args);
            } catch (error) {
                logger.error('API request failed after retries', { error, method: propertyKey });
                throw error;
            }
        };
        return descriptor;
    };
}

/**
 * Cache decorator for point cloud data
 */
function withCache(options: { ttl: number }) {
    const cache = new Map<string, { data: any; timestamp: number }>();
    
    return function(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
        const originalMethod = descriptor.value;
        descriptor.value = async function(...args: any[]) {
            const key = args[0];
            const now = Date.now();
            const cached = cache.get(key);
            
            if (cached && (now - cached.timestamp) < options.ttl) {
                return cached.data;
            }
            
            const result = await originalMethod.apply(this, args);
            cache.set(key, { data: result, timestamp: now });
            return result;
        };
        return descriptor;
    };
}

/**
 * Initiates a new LiDAR scan with specified parameters
 * @param parameters Scan configuration parameters
 * @returns Promise with scan ID and initial quality metrics
 */
@validateScanParameters
@withRetry(RETRY_CONFIG)
export async function startScan(parameters: typeof DEFAULT_SCAN_PARAMETERS): Promise<{ scanId: string; quality: number }> {
    const response = await apiClient.post(apiEndpoints.lidar.scan, {
        ...parameters,
        timestamp: Date.now()
    });
    
    logger.info('Scan initiated', { scanId: response.data.scanId, parameters });
    return response.data;
}

/**
 * Retrieves point cloud data for a specific scan
 * @param scanId Unique identifier for the scan
 * @returns Promise with point cloud data and quality metrics
 */
@validateScanParameters
@withRetry(RETRY_CONFIG)
@withCache({ ttl: 5000 }) // 5-second cache for point cloud data
export async function getPointCloud(scanId: string): Promise<{ points: Point3D[]; quality: number; density: number }> {
    const response = await apiClient.get(`${apiEndpoints.lidar.process}/${scanId}`);
    
    try {
        const validatedData = PointCloudSchema.parse(response.data);
        logger.debug('Point cloud retrieved', { 
            scanId, 
            pointCount: validatedData.points.length,
            quality: validatedData.quality 
        });
        return validatedData;
    } catch (error) {
        logger.error('Invalid point cloud data format', { error, scanId });
        throw new Error('Invalid point cloud data format');
    }
}

/**
 * Exports the default scan parameters for external use
 */
export { DEFAULT_SCAN_PARAMETERS };