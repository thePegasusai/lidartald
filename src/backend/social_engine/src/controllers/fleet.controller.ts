import { Request, Response } from 'express'; // v4.18.x
import asyncHandler from 'express-async-handler'; // v1.2.x
import cache from 'express-cache-middleware'; // v1.0.x
import { FleetService } from '../services/fleet.service';
import { validateBody, validateParams, validateQuery } from '../middleware/validation.middleware';
import { ApiError } from '../../api/src/middleware/error-handler';
import { 
    Fleet, 
    FleetStatus, 
    fleetSchema, 
    deviceCapabilitiesSchema 
} from '../types/fleet.types';
import { PERMISSIONS } from '../../security/src/rbac';

/**
 * Controller handling fleet-related HTTP endpoints with 30Hz state updates
 * and enhanced error handling for the TALD UNIA social engine
 */
export class FleetController {
    private readonly fleetService: FleetService;
    private readonly MAX_FLEET_SIZE = 32;
    private readonly STATE_UPDATE_RATE = 30; // 30Hz as per spec
    private readonly RATE_LIMIT_WINDOW = 1000; // 1 second window

    constructor() {
        this.fleetService = FleetService.getInstance();
        this.setupStateUpdateInterval();
    }

    /**
     * Create new fleet with initial device and validation
     */
    @asyncHandler
    public async createFleet(req: Request, res: Response): Promise<Response> {
        const validatedData = await validateBody(fleetSchema)(req, res, () => {});
        
        // Validate device capabilities
        const deviceCapabilities = await validateBody(deviceCapabilitiesSchema)(
            req.body.deviceCapabilities
        );

        // Ensure fleet size limit
        if (validatedData.maxDevices > this.MAX_FLEET_SIZE) {
            throw new ApiError(400, 'Fleet size exceeds maximum limit', {
                limit: this.MAX_FLEET_SIZE,
                requested: validatedData.maxDevices
            });
        }

        const fleet = await this.fleetService.createFleet({
            ...validatedData,
            hostDeviceId: req.body.deviceId,
            status: FleetStatus.INITIALIZING,
            devices: [{
                deviceId: req.body.deviceId,
                userId: req.user.id,
                status: FleetStatus.INITIALIZING,
                capabilities: deviceCapabilities,
                lastSeen: new Date()
            }]
        });

        return res.status(201).json({
            status: 'success',
            data: fleet,
            updateRate: this.STATE_UPDATE_RATE
        });
    }

    /**
     * Join existing fleet with device validation
     */
    @asyncHandler
    public async joinFleet(req: Request, res: Response): Promise<Response> {
        const { fleetId } = await validateParams({ fleetId: fleetSchema.shape.id })(req, res, () => {});
        const deviceCapabilities = await validateBody(deviceCapabilitiesSchema)(
            req.body.deviceCapabilities
        );

        const fleet = await this.fleetService.joinFleet(fleetId, {
            deviceId: req.body.deviceId,
            userId: req.user.id,
            status: FleetStatus.INITIALIZING,
            capabilities: deviceCapabilities,
            lastSeen: new Date()
        });

        return res.status(200).json({
            status: 'success',
            data: fleet,
            updateRate: this.STATE_UPDATE_RATE
        });
    }

    /**
     * Leave fleet with cleanup
     */
    @asyncHandler
    public async leaveFleet(req: Request, res: Response): Promise<Response> {
        const { fleetId } = await validateParams({ fleetId: fleetSchema.shape.id })(req, res, () => {});
        
        await this.fleetService.leaveFleet(fleetId, req.user.id);

        return res.status(204).send();
    }

    /**
     * Synchronize fleet state at 30Hz
     */
    @asyncHandler
    public async syncFleetState(req: Request, res: Response): Promise<Response> {
        const { fleetId } = await validateParams({ fleetId: fleetSchema.shape.id })(req, res, () => {});
        
        // Validate update rate compliance
        const timestamp = Date.now();
        const lastUpdate = req.get('X-Last-Update');
        if (lastUpdate && timestamp - parseInt(lastUpdate) < (1000 / this.STATE_UPDATE_RATE)) {
            throw new ApiError(429, 'Update rate exceeded', {
                minInterval: 1000 / this.STATE_UPDATE_RATE
            });
        }

        const state = await this.fleetService.syncFleetState(fleetId, {
            fleetId,
            deviceId: req.body.deviceId,
            timestamp,
            data: req.body.state,
            version: req.body.version
        });

        return res.status(200)
            .set('X-Last-Update', timestamp.toString())
            .json({
                status: 'success',
                data: state,
                timestamp
            });
    }

    /**
     * Get current fleet state
     */
    @asyncHandler
    public async getFleetState(req: Request, res: Response): Promise<Response> {
        const { fleetId } = await validateParams({ fleetId: fleetSchema.shape.id })(req, res, () => {});
        
        const state = await this.fleetService.getFleetState(fleetId);

        // Enable caching for GET requests
        res.set('Cache-Control', 'private, max-age=1'); // 1 second cache due to 30Hz updates

        return res.status(200).json({
            status: 'success',
            data: state,
            timestamp: Date.now()
        });
    }

    /**
     * Setup periodic state synchronization
     */
    private setupStateUpdateInterval(): void {
        setInterval(() => {
            this.fleetService.broadcastStateUpdates()
                .catch(error => {
                    console.error('State broadcast failed:', error);
                });
        }, 1000 / this.STATE_UPDATE_RATE);
    }
}

export default new FleetController();