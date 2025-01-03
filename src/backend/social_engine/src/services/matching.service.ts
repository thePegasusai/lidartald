import { Subject, debounceTime, filter } from 'rxjs'; // v7.x
import Redis from 'ioredis'; // v5.x
import { EventEmitter } from 'events';
import { User, UserProfile, UserPreferences } from '../types/user.types';
import { Fleet, FleetStatus, FleetDevice, FleetCapabilities } from '../types/fleet.types';
import { UserModel } from '../models/user.model';

// Types for location and proximity tracking
interface LocationUpdate {
    latitude: number;
    longitude: number;
    accuracy: number;
    timestamp: number;
}

interface ProximityOptions {
    range?: number;
    updateFrequency?: number;
    includeFleetData?: boolean;
}

type ProximityCallback = (nearbyUsers: User[]) => void;

/**
 * High-performance service for managing user proximity matching and fleet formation
 * with real-time updates and optimized spatial indexing
 */
export class MatchingService {
    private static instance: MatchingService;
    private redis: Redis;
    private userModel: UserModel;
    private eventEmitter: EventEmitter;
    private locationSubject: Subject<{ userId: string; update: LocationUpdate }>;

    // Configuration constants
    private readonly LOCATION_UPDATE_DEBOUNCE = 100; // ms
    private readonly PROXIMITY_RANGE = 5; // meters
    private readonly MAX_FLEET_SIZE = 32;
    private readonly LOCATION_KEY_PREFIX = 'location:';
    private readonly FLEET_KEY_PREFIX = 'fleet:';

    private constructor() {
        // Initialize Redis with optimized configuration
        this.redis = new Redis({
            host: process.env.REDIS_HOST || 'localhost',
            port: Number(process.env.REDIS_PORT) || 6379,
            maxRetriesPerRequest: 3,
            enableReadyCheck: true,
            autoResubscribe: true,
            retryStrategy: (times) => Math.min(times * 50, 2000)
        });

        this.userModel = new UserModel();
        this.eventEmitter = new EventEmitter();
        this.eventEmitter.setMaxListeners(100);

        // Initialize RxJS Subject for debounced location updates
        this.locationSubject = new Subject();
        this.setupLocationUpdates();
    }

    /**
     * Get singleton instance with lazy initialization
     */
    public static getInstance(): MatchingService {
        if (!MatchingService.instance) {
            MatchingService.instance = new MatchingService();
        }
        return MatchingService.instance;
    }

    /**
     * Setup location update processing with debouncing
     */
    private setupLocationUpdates(): void {
        this.locationSubject.pipe(
            debounceTime(this.LOCATION_UPDATE_DEBOUNCE),
            filter(update => this.validateLocationUpdate(update.update))
        ).subscribe(async ({ userId, update }) => {
            try {
                await this.processLocationUpdate(userId, update);
            } catch (error) {
                console.error('Location update processing failed:', error);
            }
        });
    }

    /**
     * Validate location update data
     */
    private validateLocationUpdate(update: LocationUpdate): boolean {
        return (
            typeof update.latitude === 'number' &&
            typeof update.longitude === 'number' &&
            update.latitude >= -90 && update.latitude <= 90 &&
            update.longitude >= -180 && update.longitude <= 180 &&
            update.accuracy > 0 &&
            update.timestamp <= Date.now()
        );
    }

    /**
     * Process and store location update with spatial indexing
     */
    private async processLocationUpdate(userId: string, location: LocationUpdate): Promise<void> {
        const locationKey = `${this.LOCATION_KEY_PREFIX}${userId}`;
        const geoKey = 'user_locations';

        await Promise.all([
            this.redis.geoadd(
                geoKey,
                location.longitude,
                location.latitude,
                userId
            ),
            this.redis.hmset(locationKey, {
                ...location,
                lastUpdate: Date.now()
            }),
            this.redis.expire(locationKey, 300) // 5 minute TTL
        ]);

        await this.userModel.updateLastActive(userId);
    }

    /**
     * Update user's location with optimized batch processing
     */
    public async updateUserLocation(userId: string, location: LocationUpdate): Promise<void> {
        this.locationSubject.next({ userId, update: location });
    }

    /**
     * Find users within range using spatial indexing
     */
    public async findNearbyUsers(userId: string, range: number = this.PROXIMITY_RANGE): Promise<User[]> {
        const geoKey = 'user_locations';
        const userLocation = await this.redis.geopos(geoKey, userId);

        if (!userLocation || !userLocation[0]) {
            throw new Error('User location not found');
        }

        const nearbyUsers = await this.redis.georadius(
            geoKey,
            userLocation[0][0],
            userLocation[0][1],
            range,
            'm',
            'WITHCOORD'
        );

        const userIds = nearbyUsers
            .map(user => typeof user[0] === 'string' ? user[0] : null)
            .filter(id => id && id !== userId) as string[];

        const users = await Promise.all(
            userIds.map(id => this.userModel.getUserById(id))
        );

        return users.filter(user => user !== null) as User[];
    }

    /**
     * Suggest optimal fleet formation based on multiple criteria
     */
    public async suggestFleetFormation(
        userId: string,
        preferences: UserPreferences
    ): Promise<Fleet | null> {
        const nearbyUsers = await this.findNearbyUsers(userId);
        
        if (nearbyUsers.length === 0) {
            return null;
        }

        const compatibleUsers = nearbyUsers.filter(user => 
            this.checkUserCompatibility(user, preferences)
        );

        if (compatibleUsers.length === 0) {
            return null;
        }

        const optimalSize = Math.min(
            compatibleUsers.length + 1,
            this.MAX_FLEET_SIZE
        );

        const fleetDevices: FleetDevice[] = await this.createFleetDevices(
            [userId, ...compatibleUsers.slice(0, optimalSize - 1).map(u => u.id)]
        );

        return {
            id: crypto.randomUUID(),
            name: `Fleet-${Date.now()}`,
            hostDeviceId: fleetDevices[0].deviceId,
            status: FleetStatus.INITIALIZING,
            devices: fleetDevices,
            maxDevices: optimalSize,
            createdAt: new Date(),
            updatedAt: new Date()
        };
    }

    /**
     * Subscribe to real-time proximity events with optimized updates
     */
    public subscribeToProximityEvents(
        userId: string,
        callback: ProximityCallback,
        options: ProximityOptions = {}
    ): () => void {
        const {
            range = this.PROXIMITY_RANGE,
            updateFrequency = 1000
        } = options;

        const intervalId = setInterval(async () => {
            try {
                const nearbyUsers = await this.findNearbyUsers(userId, range);
                callback(nearbyUsers);
            } catch (error) {
                console.error('Proximity event error:', error);
            }
        }, updateFrequency);

        return () => {
            clearInterval(intervalId);
        };
    }

    /**
     * Check compatibility between users for fleet formation
     */
    private checkUserCompatibility(user: User, preferences: UserPreferences): boolean {
        const userPrefs = user.preferences;
        return (
            userPrefs.autoJoinFleet &&
            userPrefs.scanRange >= preferences.scanRange &&
            userPrefs.scanResolution <= preferences.scanResolution &&
            userPrefs.privacySettings?.fleetDiscoverable !== false
        );
    }

    /**
     * Create fleet devices with capabilities
     */
    private async createFleetDevices(userIds: string[]): Promise<FleetDevice[]> {
        return Promise.all(userIds.map(async (userId) => {
            const user = await this.userModel.getUserById(userId);
            if (!user) {
                throw new Error(`User not found: ${userId}`);
            }

            return {
                deviceId: crypto.randomUUID(),
                userId,
                status: FleetStatus.INITIALIZING,
                lastSeen: new Date(),
                capabilities: {
                    lidarResolution: user.preferences.scanResolution,
                    scanRange: user.preferences.scanRange,
                    processingPower: 100 // Default to maximum
                }
            };
        }));
    }
}

export default MatchingService;