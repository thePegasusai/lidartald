import { Request, Response, NextFunction } from 'express'; // v4.18.x
import { z } from 'zod'; // v3.x
import Redis from 'ioredis'; // v5.x
import { 
    Session, 
    SessionType, 
    SessionStatus, 
    sessionSchema 
} from '../types/session.types';
import { FleetService } from '../services/fleet.service';
import { MatchingService } from '../services/matching.service';

/**
 * Controller handling gaming session management with real-time fleet coordination
 * and environment synchronization supporting up to 32 connected devices
 */
export class SessionController {
    private static instance: SessionController;
    private readonly redis: Redis;
    private readonly fleetService: FleetService;
    private readonly matchingService: MatchingService;

    private constructor() {
        // Initialize Redis with cluster support
        this.redis = new Redis({
            host: process.env.REDIS_HOST || 'localhost',
            port: Number(process.env.REDIS_PORT) || 6379,
            maxRetriesPerRequest: 3,
            retryStrategy: (times) => Math.min(times * 50, 2000),
            enableReadyCheck: true,
            autoResubscribe: true
        });

        // Initialize services
        this.fleetService = FleetService.getInstance();
        this.matchingService = MatchingService.getInstance();

        // Setup event handlers
        this.setupEventHandlers();
    }

    /**
     * Get singleton instance
     */
    public static getInstance(): SessionController {
        if (!SessionController.instance) {
            SessionController.instance = new SessionController();
        }
        return SessionController.instance;
    }

    /**
     * Setup event handlers for real-time updates
     */
    private setupEventHandlers(): void {
        this.redis.on('error', (error) => {
            console.error('Redis connection error:', error);
        });

        process.on('SIGTERM', async () => {
            await this.cleanup();
        });
    }

    /**
     * Create new gaming session with fleet coordination
     */
    public async createSession(
        req: Request,
        res: Response,
        next: NextFunction
    ): Promise<Response> {
        try {
            // Validate request body
            const validatedData = sessionSchema.parse(req.body);

            // Check fleet capacity
            const fleet = await this.fleetService.getFleetState(validatedData.fleetId);
            if (fleet.devices.length > 32) {
                throw new Error('Fleet exceeds maximum capacity of 32 devices');
            }

            // Verify proximity of participants
            const nearbyUsers = await this.matchingService.findNearbyUsers(
                req.user.id,
                validatedData.settings.scanRange
            );

            if (!this.verifyParticipantProximity(fleet.devices, nearbyUsers)) {
                throw new Error('Not all participants are within range');
            }

            // Create session with environment data
            const session: Session = {
                id: crypto.randomUUID(),
                type: validatedData.type,
                status: SessionStatus.INITIALIZING,
                fleetId: validatedData.fleetId,
                participants: fleet.devices.map(device => ({
                    userId: device.userId,
                    deviceId: device.deviceId,
                    joinedAt: new Date(),
                    status: SessionStatus.INITIALIZING,
                    score: 0,
                    lastSyncTime: new Date(),
                    position: { x: 0, y: 0, z: 0 }
                })),
                environmentId: crypto.randomUUID(),
                startedAt: new Date(),
                endedAt: null,
                settings: validatedData.settings,
                environmentData: validatedData.environmentData,
                lastSyncTimestamp: new Date()
            };

            // Store session in Redis with TTL
            await this.redis.setex(
                `session:${session.id}`,
                3600, // 1 hour TTL
                JSON.stringify(session)
            );

            // Initialize fleet sync
            await this.fleetService.syncFleetState(session.fleetId, {
                fleetId: session.fleetId,
                deviceId: req.device.id,
                timestamp: Date.now(),
                data: { sessionId: session.id },
                version: 1
            });

            return res.status(201).json(session);

        } catch (error) {
            next(error);
        }
    }

    /**
     * Join existing session with proximity validation
     */
    public async joinSession(
        req: Request,
        res: Response,
        next: NextFunction
    ): Promise<Response> {
        try {
            const { sessionId } = req.params;
            const session = await this.getSessionFromRedis(sessionId);

            if (!session) {
                throw new Error('Session not found');
            }

            if (session.participants.length >= 32) {
                throw new Error('Session has reached maximum capacity');
            }

            // Verify participant proximity
            const isInRange = await this.verifyParticipantRange(
                req.user.id,
                session.participants,
                session.settings.scanRange
            );

            if (!isInRange) {
                throw new Error('Participant not within session range');
            }

            // Add participant
            session.participants.push({
                userId: req.user.id,
                deviceId: req.device.id,
                joinedAt: new Date(),
                status: SessionStatus.INITIALIZING,
                score: 0,
                lastSyncTime: new Date(),
                position: { x: 0, y: 0, z: 0 }
            });

            // Update session in Redis
            await this.redis.setex(
                `session:${session.id}`,
                3600,
                JSON.stringify(session)
            );

            // Sync fleet state
            await this.fleetService.syncFleetState(session.fleetId, {
                fleetId: session.fleetId,
                deviceId: req.device.id,
                timestamp: Date.now(),
                data: { participantJoined: req.user.id },
                version: Date.now()
            });

            return res.json(session);

        } catch (error) {
            next(error);
        }
    }

    /**
     * Update session state with real-time synchronization
     */
    public async updateSessionState(
        req: Request,
        res: Response,
        next: NextFunction
    ): Promise<Response> {
        try {
            const { sessionId } = req.params;
            const session = await this.getSessionFromRedis(sessionId);

            if (!session) {
                throw new Error('Session not found');
            }

            // Validate participant
            if (!this.isSessionParticipant(session, req.user.id)) {
                throw new Error('Not a session participant');
            }

            // Update session state
            Object.assign(session, req.body);
            session.lastSyncTimestamp = new Date();

            // Store updated state
            await this.redis.setex(
                `session:${session.id}`,
                3600,
                JSON.stringify(session)
            );

            // Sync fleet state (<50ms latency)
            await this.fleetService.syncFleetState(session.fleetId, {
                fleetId: session.fleetId,
                deviceId: req.device.id,
                timestamp: Date.now(),
                data: req.body,
                version: Date.now()
            });

            return res.json(session);

        } catch (error) {
            next(error);
        }
    }

    /**
     * End session and cleanup resources
     */
    public async endSession(
        req: Request,
        res: Response,
        next: NextFunction
    ): Promise<Response> {
        try {
            const { sessionId } = req.params;
            const session = await this.getSessionFromRedis(sessionId);

            if (!session) {
                throw new Error('Session not found');
            }

            // Update session status
            session.status = SessionStatus.COMPLETED;
            session.endedAt = new Date();

            // Archive session data
            await this.archiveSession(session);

            // Cleanup Redis
            await this.redis.del(`session:${session.id}`);

            // Notify fleet
            await this.fleetService.syncFleetState(session.fleetId, {
                fleetId: session.fleetId,
                deviceId: req.device.id,
                timestamp: Date.now(),
                data: { sessionEnded: true },
                version: Date.now()
            });

            return res.json({ success: true });

        } catch (error) {
            next(error);
        }
    }

    /**
     * Get current session state
     */
    public async getSessionState(
        req: Request,
        res: Response,
        next: NextFunction
    ): Promise<Response> {
        try {
            const { sessionId } = req.params;
            const session = await this.getSessionFromRedis(sessionId);

            if (!session) {
                throw new Error('Session not found');
            }

            // Get real-time fleet state
            const fleetState = await this.fleetService.getFleetState(session.fleetId);

            // Merge states
            const currentState = {
                ...session,
                fleetState
            };

            return res.json(currentState);

        } catch (error) {
            next(error);
        }
    }

    /**
     * Helper method to get session from Redis
     */
    private async getSessionFromRedis(sessionId: string): Promise<Session | null> {
        const sessionData = await this.redis.get(`session:${sessionId}`);
        return sessionData ? JSON.parse(sessionData) : null;
    }

    /**
     * Verify participant proximity
     */
    private verifyParticipantProximity(
        fleetDevices: any[],
        nearbyUsers: any[]
    ): boolean {
        const nearbyUserIds = new Set(nearbyUsers.map(u => u.id));
        return fleetDevices.every(device => nearbyUserIds.has(device.userId));
    }

    /**
     * Verify if user is session participant
     */
    private isSessionParticipant(session: Session, userId: string): boolean {
        return session.participants.some(p => p.userId === userId);
    }

    /**
     * Archive completed session
     */
    private async archiveSession(session: Session): Promise<void> {
        // Implementation for session archival
        // This would typically involve storing in a persistent database
    }

    /**
     * Cleanup resources
     */
    private async cleanup(): Promise<void> {
        await this.redis.quit();
    }
}

export default SessionController;