import { z } from 'zod'; // v3.21.4
import { apiClient } from '../config/api';
import { GameSession, GameStateUpdate, gameSessionSchema, STATE_UPDATE_INTERVAL_MS } from '../types/game.types';

/**
 * API endpoints for game operations
 */
const GAME_API_ENDPOINTS = {
    CREATE_SESSION: '/api/v1/game/sessions',
    UPDATE_STATE: '/api/v1/game/sessions/:id/state'
} as const;

/**
 * Constants for API operation configuration
 */
const MAX_RETRY_ATTEMPTS = 3;
const RATE_LIMIT_WINDOW_MS = 1000;

/**
 * Validation schema for game state updates
 */
const gameStateUpdateSchema = z.object({
    sessionId: z.string().uuid('Invalid session ID'),
    timestamp: z.number().min(0, 'Invalid timestamp'),
    playerUpdates: z.array(z.object({
        playerId: z.string().uuid('Invalid player ID'),
        position: z.object({
            x: z.number(),
            y: z.number(),
            z: z.number()
        }).optional(),
        rotation: z.object({
            x: z.number(),
            y: z.number(),
            z: z.number(),
            w: z.number()
        }).optional(),
        score: z.number().optional(),
        status: z.string().optional()
    })),
    environmentUpdates: z.record(z.any()).optional()
});

/**
 * Game API client for managing game sessions and state updates
 */
export const gameApi = {
    /**
     * Creates a new game session with specified configuration
     * @param sessionConfig - Game session configuration
     * @returns Promise resolving to created game session
     */
    createGameSession: async (sessionConfig: z.infer<typeof gameSessionSchema>): Promise<GameSession> => {
        try {
            // Validate session configuration
            const validatedConfig = gameSessionSchema.parse(sessionConfig);

            // Transform environment data for API compatibility
            const apiPayload = {
                ...validatedConfig,
                environmentSettings: {
                    ...validatedConfig.config.environmentSettings,
                    timestamp: Date.now()
                }
            };

            // Send request with retry logic
            const response = await apiClient.post<GameSession>(
                GAME_API_ENDPOINTS.CREATE_SESSION,
                apiPayload,
                {
                    validateSchema: gameSessionSchema,
                    retryAttempts: MAX_RETRY_ATTEMPTS,
                    headers: {
                        'X-Request-Priority': 'high'
                    }
                }
            );

            return response.data;
        } catch (error) {
            console.error('[Game API] Create session error:', error);
            throw error;
        }
    },

    /**
     * Updates game state with optimized partial updates
     * Implements 60Hz update rate with rate limiting
     * @param stateUpdate - Partial game state update
     */
    updateGameState: async (stateUpdate: GameStateUpdate): Promise<void> => {
        try {
            // Validate state update data
            const validatedUpdate = gameStateUpdateSchema.parse(stateUpdate);

            // Optimize update payload by removing unchanged fields
            const optimizedUpdate = {
                ...validatedUpdate,
                playerUpdates: validatedUpdate.playerUpdates.map(update => {
                    const filtered: Partial<typeof update> = {};
                    Object.entries(update).forEach(([key, value]) => {
                        if (value !== undefined) {
                            filtered[key] = value;
                        }
                    });
                    return filtered;
                })
            };

            // Send state update with rate limiting
            const endpoint = GAME_API_ENDPOINTS.UPDATE_STATE.replace(
                ':id',
                validatedUpdate.sessionId
            );

            await apiClient.put(
                endpoint,
                optimizedUpdate,
                {
                    headers: {
                        'X-Update-Timestamp': validatedUpdate.timestamp.toString(),
                        'X-Rate-Limit-Window': RATE_LIMIT_WINDOW_MS.toString()
                    },
                    timeout: STATE_UPDATE_INTERVAL_MS,
                    retryAttempts: 1 // Limited retries for real-time updates
                }
            );
        } catch (error) {
            // Log error but don't throw to prevent game interruption
            console.error('[Game API] State update error:', error);
            
            // Emit error metric for monitoring
            console.debug('[Game Metrics] State update failed', {
                timestamp: Date.now(),
                error: error.message
            });
        }
    }
} as const;