import Redis from 'ioredis'; // v5.x
import { z } from 'zod'; // v3.x
import { FleetStatus, Fleet } from '../types/fleet.types';

// Environment variables with defaults
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
const REDIS_KEY_PREFIX = 'tald:';
const REDIS_TTL = 3600; // 1 hour default TTL
const REDIS_MAX_MEMORY = 52428800; // 50MB memory limit
const REDIS_PERSISTENCE_INTERVAL = 900000; // 15 minutes persistence interval

/**
 * Enhanced Redis configuration interface with cluster and monitoring support
 */
interface RedisConfig {
    url: string;
    keyPrefix: string;
    ttl: number;
    maxRetriesPerRequest?: number;
    enableReadyCheck?: boolean;
    connectionTimeout?: number;
    clusterMode: boolean;
    maxMemory: number;
    persistenceInterval: number;
    sslEnabled: boolean;
    monitoring?: {
        enableMetrics: boolean;
        alertThreshold: number;
    };
    connectionPool?: {
        minIdle: number;
        maxConnections: number;
    };
}

/**
 * Zod schema for Redis configuration validation
 */
const redisConfigSchema = z.object({
    url: z.string().url('Invalid Redis URL'),
    keyPrefix: z.string().min(1).max(20),
    ttl: z.number().min(60).max(86400),
    maxRetriesPerRequest: z.number().min(1).max(10).optional(),
    enableReadyCheck: z.boolean().optional(),
    connectionTimeout: z.number().min(1000).max(30000).optional(),
    clusterMode: z.boolean(),
    maxMemory: z.number().min(1048576).max(1073741824), // 1MB to 1GB
    persistenceInterval: z.number().min(60000).max(3600000),
    sslEnabled: z.boolean(),
    monitoring: z.object({
        enableMetrics: z.boolean(),
        alertThreshold: z.number().min(70).max(95)
    }).optional(),
    connectionPool: z.object({
        minIdle: z.number().min(1).max(10),
        maxConnections: z.number().min(5).max(100)
    }).optional()
});

/**
 * Validates Redis configuration using Zod schema
 */
function validateRedisConfig(config: RedisConfig): boolean {
    try {
        redisConfigSchema.parse(config);
        return true;
    } catch (error) {
        console.error('Redis configuration validation failed:', error);
        return false;
    }
}

/**
 * Creates and configures Redis client instance with enhanced features
 */
function createRedisClient(config: RedisConfig): Redis {
    if (!validateRedisConfig(config)) {
        throw new Error('Invalid Redis configuration');
    }

    const redisOptions: Redis.RedisOptions = {
        retryStrategy: (times: number) => {
            const delay = Math.min(times * 50, 2000);
            return delay;
        },
        maxRetriesPerRequest: config.maxRetriesPerRequest || 3,
        enableReadyCheck: config.enableReadyCheck ?? true,
        connectTimeout: config.connectionTimeout || 10000,
        keyPrefix: config.keyPrefix,
        tls: config.sslEnabled ? {} : undefined,
        connectionName: 'tald_social_engine',
        lazyConnect: true
    };

    // Configure connection pooling if specified
    if (config.connectionPool) {
        redisOptions.minIdle = config.connectionPool.minIdle;
        redisOptions.maxConnections = config.connectionPool.maxConnections;
    }

    const client = new Redis(config.url, redisOptions);

    // Configure memory limits and eviction policy
    client.config('SET', 'maxmemory', config.maxMemory.toString());
    client.config('SET', 'maxmemory-policy', 'volatile-lru');

    // Setup persistence interval
    if (config.persistenceInterval > 0) {
        setInterval(() => {
            client.bgsave().catch(error => {
                console.error('Redis persistence failed:', error);
            });
        }, config.persistenceInterval);
    }

    // Setup monitoring if enabled
    if (config.monitoring?.enableMetrics) {
        const alertThreshold = config.monitoring.alertThreshold;
        setInterval(async () => {
            const info = await client.info('memory');
            const usedMemory = parseInt(info.match(/used_memory:(\d+)/)?.[1] || '0');
            const memoryUsagePercent = (usedMemory / config.maxMemory) * 100;
            
            if (memoryUsagePercent > alertThreshold) {
                console.warn(`Redis memory usage above threshold: ${memoryUsagePercent.toFixed(2)}%`);
            }
        }, 60000);
    }

    // Error handling
    client.on('error', (error) => {
        console.error('Redis client error:', error);
    });

    client.on('connect', () => {
        console.info('Redis client connected');
    });

    return client;
}

/**
 * Default Redis configuration for social engine
 */
const redisConfig: RedisConfig = {
    url: REDIS_URL,
    keyPrefix: REDIS_KEY_PREFIX,
    ttl: REDIS_TTL,
    maxRetriesPerRequest: 3,
    enableReadyCheck: true,
    connectionTimeout: 10000,
    clusterMode: false,
    maxMemory: REDIS_MAX_MEMORY,
    persistenceInterval: REDIS_PERSISTENCE_INTERVAL,
    sslEnabled: process.env.NODE_ENV === 'production',
    monitoring: {
        enableMetrics: true,
        alertThreshold: 80
    },
    connectionPool: {
        minIdle: 2,
        maxConnections: 50
    }
};

export { redisConfig, createRedisClient, RedisConfig };