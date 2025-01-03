import { PrismaClient } from '@prisma/client'; // v4.x
import Redis from 'ioredis'; // v5.x
import { z } from 'zod'; // v3.x

// Environment configuration with defaults
const DATABASE_URL = process.env.DATABASE_URL || 'file:./dev.db';
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
const DATABASE_CONNECTION_LIMIT = Number(process.env.DATABASE_CONNECTION_LIMIT || 10);
const REDIS_CONNECTION_LIMIT = Number(process.env.REDIS_CONNECTION_LIMIT || 20);

// Configuration validation schema
const DatabaseConfigSchema = z.object({
  databaseUrl: z.string().url(),
  redisUrl: z.string().url(),
  databaseConnectionLimit: z.number().min(1).max(100),
  redisConnectionLimit: z.number().min(1).max(100),
  retentionPeriod: z.number().min(1).max(90),
  backupInterval: z.number().min(15).max(1440),
});

type DatabaseConfig = z.infer<typeof DatabaseConfigSchema>;

// Validate configuration with detailed error messages
const validateConfig = (config: DatabaseConfig): Result<DatabaseConfig, ValidationError> => {
  try {
    return { success: true, data: DatabaseConfigSchema.parse(config) };
  } catch (error) {
    return {
      success: false,
      error: {
        message: 'Database configuration validation failed',
        details: error instanceof z.ZodError ? error.errors : [],
      },
    };
  }
};

// Retry strategy with exponential backoff
const createRetryStrategy = (maxAttempts: number, baseDelay: number) => {
  return function retryStrategy(retries: number): number | null {
    if (retries >= maxAttempts) {
      return null;
    }
    const delay = Math.min(baseDelay * Math.pow(2, retries), 30000);
    const jitter = Math.random() * 1000;
    return delay + jitter;
  };
};

// Metrics collector interface for monitoring
interface MetricsCollector {
  recordConnectionAttempt(): void;
  recordConnectionSuccess(): void;
  recordConnectionFailure(error: Error): void;
  recordQueryDuration(duration: number): void;
}

class DatabaseService {
  private static instance: DatabaseService;
  private prisma: PrismaClient;
  private redis: Redis;
  private metricsCollector: MetricsCollector;
  private connectionPool: Map<string, number>;

  private constructor() {
    const config: DatabaseConfig = {
      databaseUrl: DATABASE_URL,
      redisUrl: REDIS_URL,
      databaseConnectionLimit: DATABASE_CONNECTION_LIMIT,
      redisConnectionLimit: REDIS_CONNECTION_LIMIT,
      retentionPeriod: 7, // 7 days retention as per spec
      backupInterval: 15, // 15-minute backup interval as per spec
    };

    const validationResult = validateConfig(config);
    if (!validationResult.success) {
      throw new Error(`Invalid configuration: ${JSON.stringify(validationResult.error)}`);
    }

    // Initialize Prisma with connection pooling
    this.prisma = new PrismaClient({
      datasources: {
        db: {
          url: config.databaseUrl,
        },
      },
      log: ['error', 'warn'],
      errorFormat: 'pretty',
    });

    // Initialize Redis with retry strategy
    this.redis = new Redis(config.redisUrl, {
      maxRetriesPerRequest: 3,
      retryStrategy: createRetryStrategy(5, 1000),
      connectionName: 'social-engine-cache',
      maxLoadingRetryTime: 5000,
      enableReadyCheck: true,
      showFriendlyErrorStack: process.env.NODE_ENV !== 'production',
    });

    // Initialize connection pool tracking
    this.connectionPool = new Map();

    // Setup error handlers
    this.setupErrorHandlers();
  }

  private setupErrorHandlers(): void {
    this.redis.on('error', (error) => {
      this.metricsCollector.recordConnectionFailure(error);
      console.error('Redis connection error:', error);
    });

    this.redis.on('connect', () => {
      this.metricsCollector.recordConnectionSuccess();
    });

    process.on('SIGTERM', async () => {
      await this.disconnect();
    });
  }

  public static getInstance(): DatabaseService {
    if (!DatabaseService.instance) {
      DatabaseService.instance = new DatabaseService();
    }
    return DatabaseService.instance;
  }

  public async getPrisma(): Promise<PrismaClient> {
    try {
      await this.prisma.$queryRaw`SELECT 1`;
      return this.prisma;
    } catch (error) {
      this.metricsCollector.recordConnectionFailure(error as Error);
      throw new Error('Database connection failed');
    }
  }

  public getRedis(): Redis {
    if (!this.redis.status.includes('ready')) {
      throw new Error('Redis connection not ready');
    }
    return this.redis;
  }

  public async disconnect(): Promise<void> {
    try {
      await Promise.all([
        this.prisma.$disconnect(),
        this.redis.quit(),
      ]);
      
      this.connectionPool.clear();
      DatabaseService.instance = undefined as any;
    } catch (error) {
      console.error('Error during disconnect:', error);
      throw error;
    }
  }

  // Helper method to check connection health
  private async checkConnectionHealth(): Promise<boolean> {
    try {
      const [dbHealth, redisHealth] = await Promise.all([
        this.prisma.$queryRaw`SELECT 1`,
        this.redis.ping(),
      ]);
      return true;
    } catch (error) {
      return false;
    }
  }
}

// Export singleton instance and its methods
export const getDatabaseService = DatabaseService.getInstance;
export const getPrismaClient = async () => DatabaseService.getInstance().getPrisma();
export const getRedisClient = () => DatabaseService.getInstance().getRedis();
export const disconnectDatabase = () => DatabaseService.getInstance().disconnect();