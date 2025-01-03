import { Router } from 'express'; // v4.18.x
import helmet from 'helmet'; // v7.0.x
import csrf from 'csurf'; // v1.11.x
import sanitizer from 'express-sanitizer'; // v1.0.x
import now from 'performance-now'; // v2.1.x

import { validateCreateUser, validateUpdateUser } from '../../validators/user.validator';
import { UserController } from '../../../../social_engine/src/controllers/user.controller';
import { createRateLimiter } from '../../middleware/rate-limiter';

// Constants for rate limiting windows and thresholds
const RATE_LIMIT_WINDOW = 60000; // 1 minute in ms
const PROFILE_RATE_LIMIT = 20;
const LOGIN_RATE_LIMIT = 10;
const REGISTER_RATE_LIMIT = 5;
const LOCATION_UPDATE_RATE_LIMIT = 30;
const PERFORMANCE_THRESHOLD_MS = 50;

/**
 * Initialize and configure user routes with comprehensive security middleware
 */
function initializeUserRoutes(): Router {
    const router = Router();
    const userController = new UserController();

    // Apply global security middleware
    router.use(helmet({
        contentSecurityPolicy: true,
        crossOriginEmbedderPolicy: true,
        crossOriginOpenerPolicy: true,
        crossOriginResourcePolicy: true,
        dnsPrefetchControl: true,
        frameguard: true,
        hidePoweredBy: true,
        hsts: true,
        ieNoOpen: true,
        noSniff: true,
        referrerPolicy: true,
        xssFilter: true
    }));

    // Performance monitoring middleware
    router.use((req, res, next) => {
        const start = now();
        res.on('finish', () => {
            const duration = now() - start;
            if (duration > PERFORMANCE_THRESHOLD_MS) {
                console.warn(`Route ${req.path} took ${duration}ms to process`);
            }
        });
        next();
    });

    // User registration endpoint
    router.post('/register',
        csrf(),
        sanitizer(),
        validateCreateUser,
        createRateLimiter({
            windowMs: RATE_LIMIT_WINDOW,
            maxRequests: REGISTER_RATE_LIMIT,
            keyPrefix: 'register',
            endpointPattern: 'register',
            enableFallback: true,
            redisConfig: {
                host: process.env.REDIS_HOST || 'localhost',
                port: Number(process.env.REDIS_PORT) || 6379
            },
            metrics: {
                enabled: true,
                prefix: 'user_registration'
            }
        }),
        userController.register
    );

    // User login endpoint
    router.post('/login',
        csrf(),
        sanitizer(),
        createRateLimiter({
            windowMs: RATE_LIMIT_WINDOW,
            maxRequests: LOGIN_RATE_LIMIT,
            keyPrefix: 'login',
            endpointPattern: 'login',
            enableFallback: true,
            redisConfig: {
                host: process.env.REDIS_HOST || 'localhost',
                port: Number(process.env.REDIS_PORT) || 6379
            },
            metrics: {
                enabled: true,
                prefix: 'user_login'
            }
        }),
        userController.login
    );

    // User logout endpoint
    router.post('/logout',
        csrf(),
        createRateLimiter({
            windowMs: RATE_LIMIT_WINDOW,
            maxRequests: REGISTER_RATE_LIMIT,
            keyPrefix: 'logout',
            endpointPattern: 'logout',
            enableFallback: true,
            redisConfig: {
                host: process.env.REDIS_HOST || 'localhost',
                port: Number(process.env.REDIS_PORT) || 6379
            }
        }),
        userController.logout
    );

    // Profile update endpoint
    router.put('/profile',
        helmet(),
        csrf(),
        sanitizer(),
        validateUpdateUser,
        createRateLimiter({
            windowMs: RATE_LIMIT_WINDOW,
            maxRequests: PROFILE_RATE_LIMIT,
            keyPrefix: 'profile_update',
            endpointPattern: 'profile',
            enableFallback: true,
            redisConfig: {
                host: process.env.REDIS_HOST || 'localhost',
                port: Number(process.env.REDIS_PORT) || 6379
            },
            metrics: {
                enabled: true,
                prefix: 'profile_updates'
            }
        }),
        userController.updateProfile
    );

    // Location update endpoint
    router.put('/location',
        helmet(),
        sanitizer(),
        createRateLimiter({
            windowMs: RATE_LIMIT_WINDOW,
            maxRequests: LOCATION_UPDATE_RATE_LIMIT,
            keyPrefix: 'location_update',
            endpointPattern: 'location',
            enableFallback: true,
            redisConfig: {
                host: process.env.REDIS_HOST || 'localhost',
                port: Number(process.env.REDIS_PORT) || 6379
            },
            metrics: {
                enabled: true,
                prefix: 'location_updates'
            }
        }),
        userController.updateLocation
    );

    // Nearby users discovery endpoint
    router.get('/nearby',
        helmet(),
        sanitizer(),
        createRateLimiter({
            windowMs: RATE_LIMIT_WINDOW,
            maxRequests: PROFILE_RATE_LIMIT,
            keyPrefix: 'nearby_users',
            endpointPattern: 'nearby',
            enableFallback: true,
            redisConfig: {
                host: process.env.REDIS_HOST || 'localhost',
                port: Number(process.env.REDIS_PORT) || 6379
            },
            metrics: {
                enabled: true,
                prefix: 'nearby_users'
            }
        }),
        userController.getNearbyUsers
    );

    return router;
}

// Export configured router
export const userRouter = initializeUserRoutes();