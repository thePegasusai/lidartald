import { EventEmitter } from 'events'; // node built-in
import Redis from 'ioredis'; // v5.x
import { z } from 'zod'; // v3.x

import { 
    Fleet, 
    FleetStatus, 
    fleetSchema 
} from '../types/fleet.types';
import { FleetModel } from '../models/fleet.model';
import { DatabaseService } from '../config/database';

// Constants for fleet configuration
const FLEET_SYNC_INTERVAL = 33; // 30Hz update rate as per spec
const MAX_FLEET_SIZE = 32;
const MAX_SYNC_RETRIES = 3;
const SYNC_TIMEOUT = 50; // 50ms max latency as per spec

// Custom types for fleet operations
interface RetryPolicy {
    maxAttempts: number;
    backoffMs: number;
    timeout: number;
}

interface StateUpdate {
    fleetId: string;
    deviceId: string;
    timestamp: number;
    data: Record<string, any>;
    version: number;
}

// State update validation schema
const stateUpdateSchema = z.object({
    fleetId: z.string().uuid(),
    deviceId: z.string().uuid(),
    timestamp: z.number(),
    data: z.record(z.any()),
    version: z.number().positive()
});

/**
 * Singleton service managing fleet operations with real-time state synchronization
 * and mesh network coordination supporting up to 32 devices with <50ms latency
 */
export class FleetService {
    private static instance: FleetService;
    private readonly redisClient: Redis;
    private readonly fleetModel: FleetModel;
    private readonly eventEmitter: EventEmitter;
    private readonly retryPolicies: Map<string, RetryPolicy>;
    private readonly stateCache: Map<string, Map<string, any>>;
    private readonly syncTimeouts: Map<string, NodeJS.Timeout>;

    private constructor() {
        this.redisClient = DatabaseService.getInstance().getRedis();
        this.fleetModel = FleetModel.getInstance();
        this.eventEmitter = new EventEmitter();
        this.retryPolicies = new Map();
        this.stateCache = new Map();
        this.syncTimeouts = new Map();

        this.setupEventHandlers();
        this.initializeRetryPolicies();
    }

    /**
     * Initialize retry policies for different operations
     */
    private initializeRetryPolicies(): void {
        this.retryPolicies.set('sync', {
            maxAttempts: MAX_SYNC_RETRIES,
            backoffMs: 10,
            timeout: SYNC_TIMEOUT
        });
        this.retryPolicies.set('join', {
            maxAttempts: 3,
            backoffMs: 100,
            timeout: 1000
        });
    }

    /**
     * Setup event handlers for fleet operations
     */
    private setupEventHandlers(): void {
        this.eventEmitter.on('stateUpdate', async (update: StateUpdate) => {
            try {
                await this.broadcastStateUpdate(update);
            } catch (error) {
                console.error('State update broadcast failed:', error);
            }
        });

        this.eventEmitter.on('fleetDisconnect', async (fleetId: string) => {
            try {
                await this.handleFleetDisconnect(fleetId);
            } catch (error) {
                console.error('Fleet disconnect handling failed:', error);
            }
        });
    }

    /**
     * Get singleton instance
     */
    public static getInstance(): FleetService {
        if (!FleetService.instance) {
            FleetService.instance = new FleetService();
        }
        return FleetService.instance;
    }

    /**
     * Create new fleet with real-time state management
     */
    public async createFleet(fleetData: z.infer<typeof fleetSchema>): Promise<Fleet> {
        try {
            // Validate fleet data
            const validatedData = fleetSchema.parse(fleetData);

            // Create fleet in database
            const fleet = await this.fleetModel.createFleet({
                ...validatedData,
                maxDevices: Math.min(validatedData.maxDevices, MAX_FLEET_SIZE)
            });

            // Initialize fleet state cache
            this.stateCache.set(fleet.id, new Map());

            // Setup periodic state synchronization
            this.setupStateSyncInterval(fleet.id);

            return fleet;
        } catch (error) {
            throw new Error(`Fleet creation failed: ${error.message}`);
        }
    }

    /**
     * Synchronize fleet state with conflict resolution
     */
    public async syncFleetState(fleetId: string, stateUpdate: StateUpdate): Promise<void> {
        try {
            // Validate state update
            const validatedUpdate = stateUpdateSchema.parse(stateUpdate);

            // Get retry policy
            const retryPolicy = this.retryPolicies.get('sync')!;
            let attempts = 0;

            while (attempts < retryPolicy.maxAttempts) {
                try {
                    await this.processSyncUpdate(validatedUpdate);
                    break;
                } catch (error) {
                    attempts++;
                    if (attempts === retryPolicy.maxAttempts) {
                        throw error;
                    }
                    await new Promise(resolve => setTimeout(resolve, retryPolicy.backoffMs * attempts));
                }
            }
        } catch (error) {
            throw new Error(`Fleet state sync failed: ${error.message}`);
        }
    }

    /**
     * Process and merge state updates with CRDT conflict resolution
     */
    private async processSyncUpdate(update: StateUpdate): Promise<void> {
        const fleetState = this.stateCache.get(update.fleetId);
        if (!fleetState) {
            throw new Error('Fleet state not found');
        }

        // Apply CRDT merge strategy
        const currentState = fleetState.get(update.deviceId) || { version: 0 };
        if (update.version > currentState.version) {
            fleetState.set(update.deviceId, {
                ...update.data,
                version: update.version,
                timestamp: update.timestamp
            });

            // Broadcast update to other fleet members
            this.eventEmitter.emit('stateUpdate', update);

            // Persist state to Redis
            await this.persistStateToRedis(update.fleetId, fleetState);
        }
    }

    /**
     * Setup periodic state synchronization for a fleet
     */
    private setupStateSyncInterval(fleetId: string): void {
        const interval = setInterval(async () => {
            try {
                const fleetState = this.stateCache.get(fleetId);
                if (fleetState) {
                    await this.persistStateToRedis(fleetId, fleetState);
                }
            } catch (error) {
                console.error(`State sync failed for fleet ${fleetId}:`, error);
            }
        }, FLEET_SYNC_INTERVAL);

        this.syncTimeouts.set(fleetId, interval);
    }

    /**
     * Persist fleet state to Redis
     */
    private async persistStateToRedis(fleetId: string, state: Map<string, any>): Promise<void> {
        const stateObj = Object.fromEntries(state);
        await this.redisClient.set(
            `fleet:${fleetId}:state`,
            JSON.stringify(stateObj),
            'EX',
            300 // 5 minute expiry
        );
    }

    /**
     * Handle fleet disconnection and cleanup
     */
    private async handleFleetDisconnect(fleetId: string): Promise<void> {
        try {
            // Clear sync interval
            const interval = this.syncTimeouts.get(fleetId);
            if (interval) {
                clearInterval(interval);
                this.syncTimeouts.delete(fleetId);
            }

            // Clear state cache
            this.stateCache.delete(fleetId);

            // Update fleet status
            await this.fleetModel.updateFleet(fleetId, {
                status: FleetStatus.DISCONNECTED
            });

            // Clean up Redis state
            await this.redisClient.del(`fleet:${fleetId}:state`);
        } catch (error) {
            console.error(`Fleet disconnect cleanup failed for ${fleetId}:`, error);
        }
    }

    /**
     * Cleanup resources
     */
    public async disconnect(): Promise<void> {
        // Clear all sync intervals
        for (const interval of this.syncTimeouts.values()) {
            clearInterval(interval);
        }
        this.syncTimeouts.clear();

        // Clear state cache
        this.stateCache.clear();

        // Remove all listeners
        this.eventEmitter.removeAllListeners();
    }
}

// Export singleton instance
export const getFleetService = FleetService.getInstance;