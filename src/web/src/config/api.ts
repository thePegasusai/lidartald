import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from 'axios'; // v1.4.0
import { z } from 'zod'; // v3.21.4
import { API_CONSTANTS } from './constants';
import { UserRole } from '../types/user.types';

/**
 * API endpoint configuration for different services
 */
export const apiEndpoints = {
    auth: {
        login: '/auth/login',
        logout: '/auth/logout',
        refresh: '/auth/refresh',
        verify: '/auth/verify'
    },
    user: {
        profile: '/user/profile',
        preferences: '/user/preferences',
        privacy: '/user/privacy',
        location: '/user/location'
    },
    fleet: {
        create: '/fleet/create',
        join: '/fleet/join',
        sync: '/fleet/sync',
        status: '/fleet/status',
        metrics: '/fleet/metrics'
    },
    lidar: {
        scan: '/lidar/scan',
        process: '/lidar/process',
        features: '/lidar/features',
        environment: '/lidar/environment'
    }
} as const;

/**
 * Default API configuration
 */
export const apiConfig: AxiosRequestConfig = {
    baseURL: API_CONSTANTS.BASE_URL,
    timeout: API_CONSTANTS.TIMEOUT_MS,
    headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Client-Version': '1.0.0'
    }
};

/**
 * Rate limiting configuration - 60 requests per minute
 */
const rateLimiter = {
    tokens: 60,
    interval: 60000, // 1 minute
    lastRefill: Date.now(),
    queue: [] as (() => void)[]
};

/**
 * Handles API errors with detailed logging and formatting
 */
const handleApiError = (error: AxiosError): never => {
    const errorResponse = {
        code: error.response?.status || 500,
        message: error.response?.data?.message || 'An unexpected error occurred',
        details: error.response?.data?.details || {},
        timestamp: new Date().toISOString()
    };

    // Log error metrics for monitoring
    console.error('[API Error]', {
        ...errorResponse,
        url: error.config?.url,
        method: error.config?.method
    });

    throw errorResponse;
};

/**
 * Validates API response data against schema
 */
const validateApiResponse = <T>(data: unknown, schema: z.ZodSchema<T>): T => {
    try {
        return schema.parse(data);
    } catch (error) {
        console.error('[Schema Validation Error]', error);
        throw new Error('Invalid API response format');
    }
};

/**
 * Creates configured axios instance with interceptors
 */
const createApiClient = (): AxiosInstance => {
    const client = axios.create(apiConfig);

    // Authentication interceptor
    client.interceptors.request.use((config) => {
        const token = localStorage.getItem('authToken');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    });

    // Rate limiting interceptor
    client.interceptors.request.use((config) => {
        return new Promise((resolve) => {
            const now = Date.now();
            if (now - rateLimiter.lastRefill >= rateLimiter.interval) {
                rateLimiter.tokens = 60;
                rateLimiter.lastRefill = now;
            }

            if (rateLimiter.tokens > 0) {
                rateLimiter.tokens--;
                resolve(config);
            } else {
                rateLimiter.queue.push(() => resolve(config));
                setTimeout(() => {
                    rateLimiter.tokens = 60;
                    rateLimiter.queue.forEach(resolve => resolve());
                    rateLimiter.queue = [];
                }, rateLimiter.interval - (now - rateLimiter.lastRefill));
            }
        });
    });

    // Monitoring interceptor
    client.interceptors.request.use((config) => {
        config.metadata = { startTime: Date.now() };
        return config;
    });

    client.interceptors.response.use(
        (response) => {
            const duration = Date.now() - (response.config.metadata?.startTime || 0);
            // Track API metrics
            console.debug('[API Metrics]', {
                url: response.config.url,
                method: response.config.method,
                status: response.status,
                duration,
                size: JSON.stringify(response.data).length
            });
            return response;
        },
        (error: AxiosError) => {
            const duration = Date.now() - (error.config?.metadata?.startTime || 0);
            // Track error metrics
            console.error('[API Error Metrics]', {
                url: error.config?.url,
                method: error.config?.method,
                status: error.response?.status,
                duration,
                error: error.message
            });
            return Promise.reject(error);
        }
    );

    // Retry logic with exponential backoff
    client.interceptors.response.use(
        response => response,
        async (error) => {
            const config = error.config;
            config.retryCount = config.retryCount || 0;

            if (config.retryCount < API_CONSTANTS.MAX_RETRIES && error.response?.status >= 500) {
                config.retryCount += 1;
                const delay = Math.min(1000 * Math.pow(2, config.retryCount), 5000);
                await new Promise(resolve => setTimeout(resolve, delay));
                return client(config);
            }

            return Promise.reject(error);
        }
    );

    // Response validation interceptor
    client.interceptors.response.use(
        response => {
            if (response.config.validateSchema) {
                response.data = validateApiResponse(
                    response.data,
                    response.config.validateSchema
                );
            }
            return response;
        }
    );

    return client;
};

/**
 * Export configured API client instance
 */
export const apiClient = createApiClient();

/**
 * Role-based endpoint access configuration
 */
export const roleAccess = {
    [UserRole.GUEST]: ['auth/login', 'lidar/scan'],
    [UserRole.BASIC_USER]: ['auth/*', 'user/*', 'lidar/*', 'fleet/join'],
    [UserRole.PREMIUM_USER]: ['*'],
    [UserRole.DEVELOPER]: ['*'],
    [UserRole.ADMIN]: ['*']
} as const;

/**
 * Export type definitions for API configuration
 */
export type ApiEndpoints = typeof apiEndpoints;
export type ApiConfig = typeof apiConfig;
export type RoleAccess = typeof roleAccess;