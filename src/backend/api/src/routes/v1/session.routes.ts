import { Router } from 'express'; // v4.18.x
import compression from 'compression'; // v1.7.x
import helmet from 'helmet'; // v4.x.x
import { z } from 'joi'; // v17.x.x
import { metrics } from '@opentelemetry/api'; // v1.x.x

import { SessionController } from '../../../social_engine/src/controllers/session.controller';
import { createRateLimiter } from '../../middleware/rate-limiter';
import { errorHandler } from '../../middleware/error-handler';

// Initialize metrics
const meter = metrics.getMeter('session-routes');
const requestDurationHistogram = meter.createHistogram('session_request_duration', {
    description: 'Duration of session requests',
    unit: 'ms'
});

// Session validation schemas
const sessionParamsSchema = z.object({
    sessionId: z.string().uuid()
});

// Rate limiter configuration with Redis failover
const sessionRateLimiter = createRateLimiter({
    windowMs: 60000, // 1 minute
    maxRequests: 60, // 60 requests per minute as per spec
    keyPrefix: 'session',
    endpointPattern: '^/v1/session',
    enableFallback: true,
    redisConfig: {
        host: process.env.REDIS_HOST || 'localhost',
        port: Number(process.env.REDIS_PORT) || 6379,
        tls: process.env.NODE_ENV === 'production'
    },
    metrics: {
        enabled: true,
        prefix: 'session_ratelimit'
    }
});

// Performance monitoring middleware
const monitorPerformance = (routeName: string) => (req: any, res: any, next: any) => {
    const startTime = process.hrtime();
    
    res.on('finish', () => {
        const [seconds, nanoseconds] = process.hrtime(startTime);
        const duration = seconds * 1000 + nanoseconds / 1000000;
        
        requestDurationHistogram.record(duration, {
            route: routeName,
            method: req.method,
            status: res.statusCode.toString()
        });
    });
    
    next();
};

// Configure session routes
const router = Router();
const sessionController = SessionController.getInstance();

// Apply common middleware to all routes
router.use(helmet());
router.use(compression());
router.use(sessionRateLimiter);

// Create new gaming session
router.post('/create',
    monitorPerformance('create_session'),
    async (req, res, next) => {
        try {
            const response = await sessionController.createSession(req, res, next);
            return response;
        } catch (error) {
            next(error);
        }
    }
);

// Join existing session
router.put('/join/:sessionId',
    monitorPerformance('join_session'),
    async (req, res, next) => {
        try {
            const { sessionId } = await sessionParamsSchema.validateAsync(req.params);
            const response = await sessionController.joinSession(req, res, next);
            return response;
        } catch (error) {
            next(error);
        }
    }
);

// Update session state
router.patch('/:sessionId/state',
    monitorPerformance('update_session_state'),
    async (req, res, next) => {
        try {
            const { sessionId } = await sessionParamsSchema.validateAsync(req.params);
            const response = await sessionController.updateSessionState(req, res, next);
            return response;
        } catch (error) {
            next(error);
        }
    }
);

// End session
router.post('/:sessionId/end',
    monitorPerformance('end_session'),
    async (req, res, next) => {
        try {
            const { sessionId } = await sessionParamsSchema.validateAsync(req.params);
            const response = await sessionController.endSession(req, res, next);
            return response;
        } catch (error) {
            next(error);
        }
    }
);

// Get session state
router.get('/:sessionId/state',
    monitorPerformance('get_session_state'),
    async (req, res, next) => {
        try {
            const { sessionId } = await sessionParamsSchema.validateAsync(req.params);
            const response = await sessionController.getSessionState(req, res, next);
            return response;
        } catch (error) {
            next(error);
        }
    }
);

// Error handling middleware
router.use(errorHandler);

export default router;