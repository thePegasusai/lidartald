import express, { Express, Request, Response, NextFunction } from 'express'; // v4.18.2
import cors from 'cors'; // v2.8.5
import helmet from 'helmet'; // v7.0.0
import compression from 'compression'; // v1.7.4
import morgan from 'morgan'; // v1.10.0
import { authenticate, authorize } from './middleware/auth.middleware';
import { errorHandler } from './middleware/error-handler';
import { createRateLimiter } from './middleware/rate-limiter';
import { User } from './types/user.types';

// Environment constants
const PORT = process.env.PORT || 3000;
const NODE_ENV = process.env.NODE_ENV || 'development';
const API_VERSION = 'v1';
const RATE_LIMIT_WINDOW = 60000;
const RATE_LIMIT_MAX_REQUESTS = 100;

/**
 * Creates and configures the Express application with comprehensive security features
 */
export function createApp(): Express {
    const app = express();

    // Security middleware configuration
    app.use(helmet({
        contentSecurityPolicy: {
            directives: {
                defaultSrc: ["'self'"],
                connectSrc: ["'self'", "wss://*"], // Allow WebSocket connections for mesh network
                scriptSrc: ["'self'"],
                styleSrc: ["'self'", "'unsafe-inline'"],
                imgSrc: ["'self'", "data:", "blob:"],
                workerSrc: ["'self'", "blob:"], // For WebWorkers in LiDAR processing
                frameSrc: ["'none'"],
                objectSrc: ["'none'"],
                upgradeInsecureRequests: []
            }
        },
        crossOriginEmbedderPolicy: true,
        crossOriginOpenerPolicy: { policy: "same-origin" },
        crossOriginResourcePolicy: { policy: "same-site" },
        dnsPrefetchControl: { allow: false },
        frameguard: { action: "deny" },
        hsts: {
            maxAge: 31536000,
            includeSubDomains: true,
            preload: true
        },
        referrerPolicy: { policy: "strict-origin-when-cross-origin" },
        xssFilter: true
    }));

    // CORS configuration for mesh network
    app.use(cors({
        origin: (origin, callback) => {
            // Allow requests with no origin (mobile apps, tools)
            if (!origin) return callback(null, true);
            
            // Validate origins for mesh network nodes
            const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || [];
            if (allowedOrigins.indexOf(origin) !== -1) {
                callback(null, true);
            } else {
                callback(new Error('Origin not allowed by CORS'));
            }
        },
        methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
        allowedHeaders: ['Content-Type', 'Authorization', 'X-Device-ID', 'X-Fleet-ID'],
        exposedHeaders: ['X-RateLimit-Limit', 'X-RateLimit-Remaining', 'X-RateLimit-Reset'],
        credentials: true,
        maxAge: 86400 // 24 hours
    }));

    // Compression middleware
    app.use(compression({
        level: 6,
        threshold: 1024,
        filter: (req, res) => {
            if (req.headers['x-no-compression']) return false;
            return compression.filter(req, res);
        }
    }));

    // Request logging with security event tracking
    app.use(morgan('combined', {
        skip: (req, res) => res.statusCode < 400,
        stream: {
            write: (message: string) => {
                console.log('[Security Event]', message.trim());
            }
        }
    }));

    // Body parsing middleware with size limits
    app.use(express.json({
        limit: '1mb',
        verify: (req: Request, res: Response, buf: Buffer) => {
            if (buf.length > 1024 * 1024) {
                throw new Error('Request payload too large');
            }
        }
    }));

    // Rate limiting configuration
    app.use(createRateLimiter({
        windowMs: RATE_LIMIT_WINDOW,
        maxRequests: RATE_LIMIT_MAX_REQUESTS,
        keyPrefix: 'rl',
        endpointPattern: '^/api/v1/',
        enableFallback: true,
        redisConfig: {
            host: process.env.REDIS_HOST || 'localhost',
            port: parseInt(process.env.REDIS_PORT || '6379'),
            tls: NODE_ENV === 'production'
        },
        metrics: {
            enabled: true,
            prefix: 'social_engine'
        }
    }));

    // API routes configuration
    setupRoutes(app);

    // Error handling middleware
    app.use(errorHandler);

    return app;
}

/**
 * Configures API routes with authentication and rate limiting
 */
function setupRoutes(app: Express): void {
    const apiRouter = express.Router();

    // Health check endpoint
    apiRouter.get('/health', (req: Request, res: Response) => {
        res.json({ status: 'healthy', timestamp: new Date().toISOString() });
    });

    // User routes
    apiRouter.use('/users',
        authenticate,
        authorize(['basic_user', 'premium_user', 'admin']),
        require('./routes/user.routes')
    );

    // Fleet routes
    apiRouter.use('/fleet',
        authenticate,
        authorize(['basic_user', 'premium_user', 'admin']),
        require('./routes/fleet.routes')
    );

    // Session routes
    apiRouter.use('/sessions',
        authenticate,
        authorize(['basic_user', 'premium_user', 'admin']),
        require('./routes/session.routes')
    );

    // Environment routes
    apiRouter.use('/environment',
        authenticate,
        authorize(['basic_user', 'premium_user', 'admin']),
        require('./routes/environment.routes')
    );

    // Mount API router with version prefix
    app.use(`/api/${API_VERSION}`, apiRouter);

    // 404 handler
    app.use((req: Request, res: Response) => {
        res.status(404).json({
            status: 'error',
            message: 'Resource not found',
            path: req.path
        });
    });
}

/**
 * Configures comprehensive error handling
 */
function setupErrorHandling(app: Express): void {
    // Global error handler
    app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
        errorHandler(err, req, res, next);
    });

    // Uncaught exception handler
    process.on('uncaughtException', (error: Error) => {
        console.error('Uncaught Exception:', error);
        process.exit(1);
    });

    // Unhandled rejection handler
    process.on('unhandledRejection', (reason: any) => {
        console.error('Unhandled Rejection:', reason);
        process.exit(1);
    });
}

// Export the factory function
export default createApp;