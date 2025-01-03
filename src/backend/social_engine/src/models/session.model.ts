import { PrismaClient } from '@prisma/client'; // v4.x
import { z } from 'zod'; // v3.x
import Redis from 'ioredis'; // v5.x
import { Session, SessionStatus, SessionType, sessionSchema } from '../types/session.types';
import { DatabaseService } from '../config/database';

/**
 * Retry policy configuration for session operations
 */
interface RetryPolicy {
    maxAttempts: number;
    baseDelay: number;
    maxDelay: number;
}

/**
 * Session model class for managing gaming sessions with real-time synchronization
 */
export class SessionModel {
    private static instance: SessionModel;
    private prisma: PrismaClient;
    private redis: Redis;
    private retryPolicies: Map<string, RetryPolicy>;

    private constructor() {
        this.initializeServices();
        this.setupRetryPolicies();
        this.setupEventListeners();
    }

    /**
     * Initialize database and cache services
     */
    private async initializeServices(): Promise<void> {
        const dbService = DatabaseService.getInstance();
        this.prisma = await dbService.getPrisma();
        this.redis = dbService.getRedis();
    }

    /**
     * Configure retry policies for different operations
     */
    private setupRetryPolicies(): void {
        this.retryPolicies = new Map([
            ['create', { maxAttempts: 3, baseDelay: 1000, maxDelay: 5000 }],
            ['update', { maxAttempts: 5, baseDelay: 500, maxDelay: 3000 }],
            ['sync', { maxAttempts: 10, baseDelay: 100, maxDelay: 1000 }]
        ]);
    }

    /**
     * Setup event listeners for session state changes
     */
    private setupEventListeners(): void {
        this.redis.subscribe('session:state:change', 'session:sync:request');
        this.redis.on('message', this.handleRedisMessage.bind(this));
    }

    /**
     * Handle Redis pub/sub messages
     */
    private async handleRedisMessage(channel: string, message: string): Promise<void> {
        try {
            const data = JSON.parse(message);
            switch (channel) {
                case 'session:state:change':
                    await this.handleStateChange(data);
                    break;
                case 'session:sync:request':
                    await this.handleSyncRequest(data);
                    break;
            }
        } catch (error) {
            console.error('Redis message handling error:', error);
        }
    }

    /**
     * Get singleton instance of SessionModel
     */
    public static getInstance(): SessionModel {
        if (!SessionModel.instance) {
            SessionModel.instance = new SessionModel();
        }
        return SessionModel.instance;
    }

    /**
     * Create a new gaming session
     */
    public async createSession(sessionData: z.infer<typeof sessionSchema>): Promise<Session> {
        try {
            // Validate session data
            const validatedData = sessionSchema.parse(sessionData);

            // Create session with optimistic locking
            const session = await this.prisma.$transaction(async (tx) => {
                const created = await tx.session.create({
                    data: {
                        type: validatedData.type,
                        status: SessionStatus.INITIALIZING,
                        fleetId: validatedData.fleetId,
                        environmentData: validatedData.environmentData,
                        settings: validatedData.settings,
                        startedAt: new Date(),
                        lastSyncTimestamp: new Date()
                    }
                });

                // Cache session state
                await this.redis.setex(
                    `session:${created.id}`,
                    300, // 5 minutes TTL
                    JSON.stringify(created)
                );

                return created;
            });

            // Publish session creation event
            await this.redis.publish('session:state:change', JSON.stringify({
                type: 'created',
                sessionId: session.id
            }));

            return session;
        } catch (error) {
            console.error('Session creation error:', error);
            throw new Error('Failed to create session');
        }
    }

    /**
     * Update session state with optimistic locking
     */
    public async updateSession(
        sessionId: string,
        updateData: Partial<Session>
    ): Promise<Session> {
        try {
            const session = await this.prisma.$transaction(async (tx) => {
                // Get current session with lock
                const current = await tx.session.findUnique({
                    where: { id: sessionId },
                    select: { id: true, version: true }
                });

                if (!current) {
                    throw new Error('Session not found');
                }

                // Update session with version check
                const updated = await tx.session.update({
                    where: {
                        id: sessionId,
                        version: current.version
                    },
                    data: {
                        ...updateData,
                        version: { increment: 1 },
                        updatedAt: new Date()
                    }
                });

                // Invalidate cache
                await this.redis.del(`session:${sessionId}`);

                return updated;
            });

            // Publish update event
            await this.redis.publish('session:state:change', JSON.stringify({
                type: 'updated',
                sessionId,
                changes: updateData
            }));

            return session;
        } catch (error) {
            console.error('Session update error:', error);
            throw new Error('Failed to update session');
        }
    }

    /**
     * Handle session state changes
     */
    private async handleStateChange(data: any): Promise<void> {
        try {
            const { sessionId, type, changes } = data;
            const cacheKey = `session:${sessionId}`;

            if (type === 'updated') {
                // Update cache with new state
                const currentState = await this.redis.get(cacheKey);
                if (currentState) {
                    const updated = {
                        ...JSON.parse(currentState),
                        ...changes
                    };
                    await this.redis.setex(cacheKey, 300, JSON.stringify(updated));
                }
            }
        } catch (error) {
            console.error('State change handling error:', error);
        }
    }

    /**
     * Handle session sync requests
     */
    private async handleSyncRequest(data: any): Promise<void> {
        try {
            const { sessionId, deviceId } = data;
            const session = await this.prisma.session.findUnique({
                where: { id: sessionId }
            });

            if (session) {
                await this.redis.publish('session:sync:response', JSON.stringify({
                    sessionId,
                    deviceId,
                    state: session
                }));
            }
        } catch (error) {
            console.error('Sync request handling error:', error);
        }
    }

    /**
     * Get session by ID with caching
     */
    public async getSession(sessionId: string): Promise<Session | null> {
        try {
            // Check cache first
            const cached = await this.redis.get(`session:${sessionId}`);
            if (cached) {
                return JSON.parse(cached);
            }

            // Fetch from database
            const session = await this.prisma.session.findUnique({
                where: { id: sessionId }
            });

            if (session) {
                // Cache the result
                await this.redis.setex(
                    `session:${sessionId}`,
                    300,
                    JSON.stringify(session)
                );
            }

            return session;
        } catch (error) {
            console.error('Session fetch error:', error);
            throw new Error('Failed to fetch session');
        }
    }
}

// Export singleton instance
export const getSessionModel = SessionModel.getInstance;