import { Request, Response } from 'express'; // v4.18.x
import { z } from 'zod'; // v3.21.x
import Redis from 'ioredis'; // v5.3.x
import { RateLimiter } from 'rate-limiter-flexible'; // v2.4.x
import { UserModel } from '../models/user.model';
import { AuthService } from '../services/auth.service';
import { MatchingService } from '../services/matching.service';
import { 
    createUserSchema, 
    updateUserSchema, 
    User, 
    UserProfile 
} from '../types/user.types';
import { FleetStatus } from '../types/fleet.types';

/**
 * Enhanced controller handling user-related HTTP endpoints with fleet awareness
 * and real-time social features
 */
export class UserController {
    private readonly rateLimiter: RateLimiter;
    private readonly redisClient: Redis;

    constructor(
        private readonly userModel: UserModel,
        private readonly authService: AuthService,
        private readonly matchingService: MatchingService,
        redisOptions: Redis.RedisOptions
    ) {
        // Initialize Redis client
        this.redisClient = new Redis({
            ...redisOptions,
            enableAutoPipelining: true,
            maxRetriesPerRequest: 3
        });

        // Configure rate limiting
        this.rateLimiter = new RateLimiter({
            storeClient: this.redisClient,
            points: 100,
            duration: 60,
            blockDuration: 300
        });
    }

    /**
     * Register new user with device binding and fleet preferences
     */
    public register = async (req: Request, res: Response): Promise<void> => {
        try {
            // Rate limit check
            await this.rateLimiter.consume(req.ip);

            // Validate request data
            const validatedData = createUserSchema.parse(req.body);

            // Verify device hardware token
            const deviceContext = await this.authService.validateHardwareToken(
                req.body.deviceToken,
                {
                    deviceId: req.body.deviceId,
                    hardwareId: req.body.hardwareId,
                    firmwareVersion: req.body.firmwareVersion,
                    securityLevel: 0
                }
            );

            // Create user with fleet preferences
            const user = await this.userModel.createUser({
                ...validatedData,
                initialPreferences: {
                    ...validatedData.initialPreferences,
                    scanResolution: 0.01,
                    scanRange: 5,
                    autoJoinFleet: true,
                    privacySettings: {
                        profileVisibility: 'public',
                        locationSharing: true,
                        fleetDiscoverable: true
                    }
                }
            });

            // Generate auth tokens
            const authResponse = await this.authService.login({
                email: validatedData.email,
                password: validatedData.password,
                deviceId: req.body.deviceId,
                hardwareId: req.body.hardwareId,
                firmwareVersion: req.body.firmwareVersion
            });

            res.status(201).json({
                user,
                auth: authResponse,
                deviceContext
            });

        } catch (error) {
            if (error instanceof z.ZodError) {
                res.status(400).json({
                    error: 'Validation failed',
                    details: error.errors
                });
                return;
            }

            res.status(500).json({
                error: 'Registration failed',
                message: error.message
            });
        }
    };

    /**
     * Update user location and calculate proximity data
     */
    public updateLocation = async (req: Request, res: Response): Promise<void> => {
        try {
            const userId = req.user.id;
            const { latitude, longitude, accuracy } = req.body;

            // Validate location data
            if (!latitude || !longitude || !accuracy) {
                res.status(400).json({ error: 'Invalid location data' });
                return;
            }

            // Update user location
            await this.matchingService.updateUserLocation(userId, {
                latitude,
                longitude,
                accuracy,
                timestamp: Date.now()
            });

            // Find nearby users
            const nearbyUsers = await this.matchingService.findNearbyUsers(
                userId,
                req.user.preferences.scanRange
            );

            // Calculate fleet proximity if user is in fleet
            const fleetProximity = req.user.fleetId ? 
                await this.matchingService.calculateFleetProximity(
                    userId,
                    req.user.fleetId
                ) : null;

            // Update fleet status if needed
            if (fleetProximity && fleetProximity.averageDistance > 5) {
                await this.userModel.updateFleetStatus(
                    userId,
                    req.user.fleetId,
                    FleetStatus.DISCONNECTED
                );
            }

            res.status(200).json({
                nearbyUsers: nearbyUsers.map(user => ({
                    id: user.id,
                    username: user.username,
                    distance: user.distance,
                    fleetStatus: user.fleetStatus
                })),
                fleetProximity
            });

        } catch (error) {
            res.status(500).json({
                error: 'Location update failed',
                message: error.message
            });
        }
    };

    /**
     * Update user profile with fleet preferences
     */
    public updateProfile = async (req: Request, res: Response): Promise<void> => {
        try {
            const userId = req.user.id;
            const validatedData = updateUserSchema.parse(req.body);

            // Update user profile
            const updatedProfile = await this.userModel.updateUserProfile(
                userId,
                {
                    ...validatedData,
                    preferences: validatedData.preferences ? {
                        ...req.user.preferences,
                        ...validatedData.preferences,
                        // Ensure fleet-critical settings are preserved
                        scanResolution: Math.min(
                            validatedData.preferences.scanResolution || 0.01,
                            req.user.preferences.scanResolution
                        ),
                        scanRange: Math.min(
                            validatedData.preferences.scanRange || 5,
                            req.user.preferences.scanRange
                        )
                    } : undefined
                }
            );

            res.status(200).json(updatedProfile);

        } catch (error) {
            if (error instanceof z.ZodError) {
                res.status(400).json({
                    error: 'Validation failed',
                    details: error.errors
                });
                return;
            }

            res.status(500).json({
                error: 'Profile update failed',
                message: error.message
            });
        }
    };

    /**
     * Get nearby users with fleet status
     */
    public getNearbyUsers = async (req: Request, res: Response): Promise<void> => {
        try {
            const userId = req.user.id;
            const range = Number(req.query.range) || req.user.preferences.scanRange;

            const nearbyUsers = await this.matchingService.findNearbyUsers(
                userId,
                range
            );

            // Enrich user data with fleet information
            const enrichedUsers = await Promise.all(
                nearbyUsers.map(async user => {
                    const fleetStatus = await this.redisClient.get(
                        `fleet:${user.id}`
                    );
                    return {
                        ...user,
                        fleetStatus: fleetStatus || null
                    };
                })
            );

            res.status(200).json(enrichedUsers);

        } catch (error) {
            res.status(500).json({
                error: 'Failed to fetch nearby users',
                message: error.message
            });
        }
    };

    /**
     * Handle user logout with fleet cleanup
     */
    public logout = async (req: Request, res: Response): Promise<void> => {
        try {
            const userId = req.user.id;
            const sessionId = req.sessionID;

            // Update fleet status if user is in fleet
            if (req.user.fleetId) {
                await this.userModel.updateFleetStatus(
                    userId,
                    req.user.fleetId,
                    'left'
                );
            }

            // Perform logout
            await this.authService.logout(sessionId);

            // Clear location data
            await this.redisClient.del(`location:${userId}`);

            res.status(200).json({ message: 'Logout successful' });

        } catch (error) {
            res.status(500).json({
                error: 'Logout failed',
                message: error.message
            });
        }
    };
}

export default UserController;