import { Request, Response, NextFunction } from 'express'; // express@4.18.x
import { RedisClient } from 'redis'; // redis@7.0.x
import NodeCache from 'node-cache'; // node-cache@5.1.x
import { ApiError } from './error-handler';
import { v4 as uuidv4 } from 'uuid'; // uuid@9.0.x

// Constants for rate limiting configuration
const RATE_LIMIT_EXCEEDED_CODE = 4029;
const RATE_LIMIT_EXCEEDED_MESSAGE = 'Rate limit exceeded. Please try again later.';
const REDIS_RETRY_ATTEMPTS = 3;
const REDIS_RETRY_DELAY = 1000;
const LOCAL_CACHE_TTL = 60000;
const BYPASS_HEADER = 'X-RateLimit-Bypass';
const REQUEST_ID_HEADER = 'X-Request-ID';
const SECURITY_HEADERS = {
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
  'X-XSS-Protection': '1; mode=block'
};

// Predefined rate limits from technical specification
const ENDPOINT_LIMITS = {
  'fleet/discover': { windowMs: 60000, maxRequests: 10 },
  'fleet/connect': { windowMs: 60000, maxRequests: 5 },
  'environment/sync': { windowMs: 60000, maxRequests: 30 },
  'session/state': { windowMs: 60000, maxRequests: 60 },
  'user/profile': { windowMs: 60000, maxRequests: 20 }
};

interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
  keyPrefix: string;
  endpointPattern: string;
  bypassKey?: string;
  enableFallback: boolean;
  redisConfig: {
    host: string;
    port: number;
    password?: string;
    tls?: boolean;
  };
  metrics?: {
    enabled: boolean;
    prefix: string;
  };
}

interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset: number;
  requestId: string;
  endpoint: string;
  clientId: string;
  timestamp: number;
}

async function getRateLimitInfo(
  redis: RedisClient,
  key: string,
  endpoint: string,
  clientId: string,
  config: RateLimitConfig
): Promise<RateLimitInfo> {
  const requestId = uuidv4();
  const now = Date.now();
  const windowMs = ENDPOINT_LIMITS[endpoint]?.windowMs || config.windowMs;
  const maxRequests = ENDPOINT_LIMITS[endpoint]?.maxRequests || config.maxRequests;

  const count = await redis.get(key).catch(() => '0');
  const currentCount = parseInt(count || '0', 10);
  
  return {
    limit: maxRequests,
    remaining: Math.max(0, maxRequests - currentCount),
    reset: Math.ceil((now + windowMs) / 1000),
    requestId,
    endpoint,
    clientId,
    timestamp: now
  };
}

export function createRateLimiter(config: RateLimitConfig) {
  // Initialize Redis client with retry logic
  const redis = new RedisClient(config.redisConfig);
  redis.on('error', (err) => console.error('Redis error:', err));

  // Initialize local cache fallback
  const localCache = new NodeCache({ stdTTL: LOCAL_CACHE_TTL / 1000 });

  return async (req: Request, res: Response, next: NextFunction) => {
    try {
      // Extract client identifier (IP or authenticated user ID)
      const clientId = req.user?.id || req.ip;
      const endpoint = req.path.replace(/^\/+/, '');

      // Check bypass authorization
      if (config.bypassKey && req.header(BYPASS_HEADER) === config.bypassKey) {
        return next();
      }

      // Match endpoint pattern
      if (!endpoint.match(config.endpointPattern)) {
        return next();
      }

      // Generate rate limit key
      const key = `${config.keyPrefix}:${endpoint}:${clientId}`;

      let rateLimitInfo: RateLimitInfo;
      let useLocalCache = false;

      // Try Redis first
      for (let attempt = 0; attempt < REDIS_RETRY_ATTEMPTS; attempt++) {
        try {
          const multi = redis.multi();
          multi.incr(key);
          multi.pexpire(key, ENDPOINT_LIMITS[endpoint]?.windowMs || config.windowMs);
          await multi.exec();

          rateLimitInfo = await getRateLimitInfo(redis, key, endpoint, clientId, config);
          break;
        } catch (err) {
          if (attempt === REDIS_RETRY_ATTEMPTS - 1) {
            if (config.enableFallback) {
              useLocalCache = true;
            } else {
              throw err;
            }
          }
          await new Promise(resolve => setTimeout(resolve, REDIS_RETRY_DELAY));
        }
      }

      // Fallback to local cache if Redis fails
      if (useLocalCache) {
        const localKey = `${key}:local`;
        const count = (localCache.get(localKey) as number || 0) + 1;
        localCache.set(localKey, count);

        rateLimitInfo = {
          limit: config.maxRequests,
          remaining: Math.max(0, config.maxRequests - count),
          reset: Math.ceil((Date.now() + config.windowMs) / 1000),
          requestId: uuidv4(),
          endpoint,
          clientId,
          timestamp: Date.now()
        };
      }

      // Set rate limit headers
      res.setHeader('X-RateLimit-Limit', rateLimitInfo.limit);
      res.setHeader('X-RateLimit-Remaining', rateLimitInfo.remaining);
      res.setHeader('X-RateLimit-Reset', rateLimitInfo.reset);
      res.setHeader(REQUEST_ID_HEADER, rateLimitInfo.requestId);

      // Set security headers
      Object.entries(SECURITY_HEADERS).forEach(([header, value]) => {
        res.setHeader(header, value);
      });

      // Check if rate limit exceeded
      if (rateLimitInfo.remaining < 0) {
        throw new ApiError(
          429,
          RATE_LIMIT_EXCEEDED_MESSAGE,
          {
            retryAfter: rateLimitInfo.reset - Math.floor(Date.now() / 1000),
            requestId: rateLimitInfo.requestId
          }
        );
      }

      // Collect metrics if enabled
      if (config.metrics?.enabled) {
        req.app.get('metrics')?.incrementCounter(`${config.metrics.prefix}_requests_total`, {
          endpoint,
          status: 'allowed'
        });
      }

      next();
    } catch (error) {
      if (config.metrics?.enabled) {
        req.app.get('metrics')?.incrementCounter(`${config.metrics.prefix}_requests_total`, {
          endpoint: req.path.replace(/^\/+/, ''),
          status: 'blocked'
        });
      }
      next(error);
    }
  };
}