import express, { Router } from 'express'; // v4.18.2
import { 
    validateCreateFleet, 
    validateUpdateFleet, 
    validateAddDevice 
} from '../../validators/fleet.validator';
import { createRateLimiter } from '../../middleware/rate-limiter';
import FleetController from '../../../../social_engine/src/controllers/fleet.controller';
import { ApiError } from '../../middleware/error-handler';

// Rate limit configurations based on technical specifications
const FLEET_RATE_LIMITS = {
    discover: {
        windowMs: 60000, // 1 minute
        maxRequests: 10,
        keyPrefix: 'fleet:discover'
    },
    connect: {
        windowMs: 60000,
        maxRequests: 5,
        keyPrefix: 'fleet:connect'
    },
    sync: {
        windowMs: 60000,
        maxRequests: 30,
        keyPrefix: 'fleet:sync'
    },
    state: {
        windowMs: 60000,
        maxRequests: 60,
        keyPrefix: 'fleet:state'
    }
} as const;

// Initialize router
const router: Router = express.Router();

// Create fleet rate limiters
const discoverLimiter = createRateLimiter(FLEET_RATE_LIMITS.discover);
const connectLimiter = createRateLimiter(FLEET_RATE_LIMITS.connect);
const syncLimiter = createRateLimiter(FLEET_RATE_LIMITS.sync);
const stateLimiter = createRateLimiter(FLEET_RATE_LIMITS.state);

/**
 * POST /fleets
 * Create new fleet with initial device
 * Rate limit: 10 requests per minute
 */
router.post('/fleets',
    discoverLimiter,
    async (req, res, next) => {
        try {
            await validateCreateFleet(req.body);
            return FleetController.createFleet(req, res);
        } catch (error) {
            next(error);
        }
    }
);

/**
 * POST /fleets/:fleetId/devices
 * Add device to existing fleet
 * Rate limit: 5 requests per minute
 */
router.post('/fleets/:fleetId/devices',
    connectLimiter,
    async (req, res, next) => {
        try {
            await validateAddDevice(req.body);
            return FleetController.joinFleet(req, res);
        } catch (error) {
            next(error);
        }
    }
);

/**
 * DELETE /fleets/:fleetId/devices
 * Remove device from fleet
 * Rate limit: 5 requests per minute
 */
router.delete('/fleets/:fleetId/devices',
    connectLimiter,
    async (req, res, next) => {
        try {
            if (!req.params.fleetId) {
                throw new ApiError(400, 'Fleet ID is required');
            }
            return FleetController.leaveFleet(req, res);
        } catch (error) {
            next(error);
        }
    }
);

/**
 * GET /fleets/:fleetId/state
 * Get current fleet state
 * Rate limit: 60 requests per minute (supports 30Hz updates)
 */
router.get('/fleets/:fleetId/state',
    stateLimiter,
    async (req, res, next) => {
        try {
            if (!req.params.fleetId) {
                throw new ApiError(400, 'Fleet ID is required');
            }
            return FleetController.getFleetState(req, res);
        } catch (error) {
            next(error);
        }
    }
);

/**
 * PATCH /fleets/:fleetId/sync
 * Synchronize fleet state
 * Rate limit: 30 requests per minute
 */
router.patch('/fleets/:fleetId/sync',
    syncLimiter,
    async (req, res, next) => {
        try {
            if (!req.params.fleetId) {
                throw new ApiError(400, 'Fleet ID is required');
            }
            return FleetController.syncFleetState(req, res);
        } catch (error) {
            next(error);
        }
    }
);

// Error handling middleware
router.use((error: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
    if (error instanceof ApiError) {
        return res.status(error.statusCode).json({
            status: 'error',
            message: error.message,
            details: error.details
        });
    }
    next(error);
});

export default router;