import { z } from 'zod'; // v3.21.4
import { apiClient } from '../config/api';
import { EnvironmentMap, EnvironmentUpdate } from '../types/environment.types';
import { Point3D } from '../types/lidar.types';
import { ENVIRONMENT_CONSTANTS } from '../config/constants';

/**
 * API endpoints for environment-related operations
 */
const ENVIRONMENT_API_ENDPOINTS = {
    GET_MAP: '/api/v1/environment/map',
    CREATE_MAP: '/api/v1/environment/map',
    UPDATE_MAP: '/api/v1/environment/map/:id',
    GET_FEATURES: '/api/v1/environment/features',
    UPDATE_FEATURES: '/api/v1/environment/features/:id'
} as const;

/**
 * Constants for environment API operations
 */
const UPDATE_BATCH_SIZE = 100;
const SCAN_INTERVAL_MS = 33; // 30Hz scan rate
const RETRY_CONFIG = {
    MAX_RETRIES: 3,
    BACKOFF_MS: 100,
    TIMEOUT_MS: 5000
};

/**
 * Zod validation schemas for environment data
 */
const Point3DSchema = z.object({
    x: z.number(),
    y: z.number(),
    z: z.number(),
    intensity: z.number().min(0).max(1)
});

const FeatureSchema = z.object({
    id: z.string().uuid(),
    type: z.string(),
    points: z.array(Point3DSchema),
    classification: z.enum(['FLOOR', 'WALL', 'OBSTACLE', 'UNKNOWN']),
    confidence: z.number().min(0).max(1),
    metadata: z.record(z.any()),
    lastUpdated: z.number()
});

const EnvironmentMapSchema = z.object({
    id: z.string().uuid(),
    timestamp: z.number(),
    points: z.array(Point3DSchema),
    features: z.array(FeatureSchema),
    resolution: z.number().min(0.01),
    version: z.number(),
    lastProcessed: z.number()
});

const UpdateRequestSchema = z.object({
    mapId: z.string().uuid(),
    newPoints: z.array(Point3DSchema),
    sequenceNumber: z.number(),
    partialUpdate: z.boolean()
});

/**
 * Decorator for input validation
 */
function validateInput(schema: z.ZodSchema) {
    return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
        const originalMethod = descriptor.value;
        descriptor.value = async function (...args: any[]) {
            try {
                schema.parse(args[0]);
                return await originalMethod.apply(this, args);
            } catch (error) {
                console.error('[Validation Error]', error);
                throw new Error('Invalid input data');
            }
        };
        return descriptor;
    };
}

/**
 * Decorator for retry logic
 */
function withRetry(config: typeof RETRY_CONFIG) {
    return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
        const originalMethod = descriptor.value;
        descriptor.value = async function (...args: any[]) {
            let retries = 0;
            while (retries < config.MAX_RETRIES) {
                try {
                    return await originalMethod.apply(this, args);
                } catch (error) {
                    retries++;
                    if (retries === config.MAX_RETRIES) throw error;
                    await new Promise(resolve => setTimeout(resolve, config.BACKOFF_MS * Math.pow(2, retries)));
                }
            }
        };
        return descriptor;
    };
}

/**
 * Environment API client for handling environment-related operations
 */
export const environmentApi = {
    /**
     * Retrieves the current environment map with all features
     */
    @validateInput(z.string().uuid())
    @withRetry(RETRY_CONFIG)
    async getEnvironmentMap(mapId: string): Promise<EnvironmentMap> {
        const response = await apiClient.get(`${ENVIRONMENT_API_ENDPOINTS.GET_MAP}/${mapId}`, {
            validateSchema: EnvironmentMapSchema,
            timeout: RETRY_CONFIG.TIMEOUT_MS
        });
        return response.data;
    },

    /**
     * Creates a new environment map from initial scan data
     */
    @validateInput(z.object({ points: z.array(Point3DSchema), resolution: z.number().min(0.01) }))
    @withRetry(RETRY_CONFIG)
    async createEnvironmentMap(points: Point3D[], resolution: number): Promise<EnvironmentMap> {
        if (points.length > ENVIRONMENT_CONSTANTS.MAX_FEATURES_PER_SCAN) {
            throw new Error(`Exceeds maximum points limit of ${ENVIRONMENT_CONSTANTS.MAX_FEATURES_PER_SCAN}`);
        }

        const response = await apiClient.post(ENVIRONMENT_API_ENDPOINTS.CREATE_MAP, {
            points,
            resolution,
            timestamp: Date.now(),
            version: 1
        }, {
            validateSchema: EnvironmentMapSchema,
            timeout: RETRY_CONFIG.TIMEOUT_MS
        });
        return response.data;
    },

    /**
     * Updates an existing environment map with new scan data
     */
    @validateInput(UpdateRequestSchema)
    @withRetry(RETRY_CONFIG)
    async updateEnvironmentMap(update: EnvironmentUpdate): Promise<void> {
        const chunks = [];
        for (let i = 0; i < update.newPoints.length; i += UPDATE_BATCH_SIZE) {
            chunks.push(update.newPoints.slice(i, i + UPDATE_BATCH_SIZE));
        }

        for (const chunk of chunks) {
            await apiClient.put(
                ENVIRONMENT_API_ENDPOINTS.UPDATE_MAP.replace(':id', update.mapId),
                {
                    ...update,
                    newPoints: chunk,
                    timestamp: Date.now()
                },
                { timeout: RETRY_CONFIG.TIMEOUT_MS }
            );
        }
    },

    /**
     * Retrieves environment features for a specific map
     */
    @validateInput(z.string().uuid())
    @withRetry(RETRY_CONFIG)
    async getEnvironmentFeatures(mapId: string): Promise<EnvironmentMap['features']> {
        const response = await apiClient.get(`${ENVIRONMENT_API_ENDPOINTS.GET_FEATURES}/${mapId}`, {
            validateSchema: z.array(FeatureSchema),
            timeout: RETRY_CONFIG.TIMEOUT_MS
        });
        return response.data;
    }
};