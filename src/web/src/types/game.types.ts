import { z } from 'zod'; // v3.x
import { THREE } from 'three'; // v0.150.0
import { EnvironmentMap } from './environment.types';
import { Fleet } from './fleet.types';

/**
 * Constants for game performance and scaling
 */
export const TARGET_FRAME_RATE = 60; // 60 FPS UI responsiveness
export const MAX_PLAYERS = 32; // Maximum players per session
export const STATE_UPDATE_INTERVAL_MS = 16; // ~60Hz state updates

/**
 * Enumeration of available game modes
 */
export enum GameMode {
    BATTLE_ARENA = 'BATTLE_ARENA',
    CAPTURE_POINT = 'CAPTURE_POINT',
    EXPLORATION = 'EXPLORATION'
}

/**
 * Enumeration of possible game session states
 */
export enum GameStatus {
    INITIALIZING = 'INITIALIZING',
    WAITING_FOR_PLAYERS = 'WAITING_FOR_PLAYERS',
    IN_PROGRESS = 'IN_PROGRESS',
    PAUSED = 'PAUSED',
    COMPLETED = 'COMPLETED'
}

/**
 * Interface representing real-time player state during gameplay
 */
export interface PlayerState {
    /** Unique player identifier */
    playerId: string;

    /** Associated device identifier */
    deviceId: string;

    /** 3D position in game space */
    position: THREE.Vector3;

    /** Player orientation */
    rotation: THREE.Quaternion;

    /** Current score */
    score: number;

    /** Player status (active/inactive/eliminated) */
    status: string;
}

/**
 * Interface for game session configuration
 */
export interface GameConfig {
    /** Maximum number of players (2-32) */
    maxPlayers: number;

    /** Session duration in seconds */
    duration: number;

    /** Score required to win */
    scoreLimit: number;

    /** Environment-specific game settings */
    environmentSettings: Record<string, any>;
}

/**
 * Core interface representing an active game session
 */
export interface GameSession {
    /** Unique session identifier */
    id: string;

    /** Associated fleet identifier */
    fleetId: string;

    /** Environment map identifier */
    environmentMapId: string;

    /** Selected game mode */
    mode: GameMode;

    /** Current session status */
    status: GameStatus;

    /** Array of active player states */
    players: PlayerState[];

    /** Session start timestamp */
    startTime: Date;

    /** Session end timestamp (null if ongoing) */
    endTime: Date | null;

    /** Session configuration */
    config: GameConfig;
}

/**
 * Interface for real-time game state updates
 */
export interface GameStateUpdate {
    /** Associated session identifier */
    sessionId: string;

    /** Update timestamp */
    timestamp: number;

    /** Array of player state updates */
    playerUpdates: Partial<PlayerState>[];

    /** Environment state changes */
    environmentUpdates: Record<string, any>;
}

/**
 * Zod validation schema for game session creation/updates
 */
export const gameSessionSchema = z.object({
    fleetId: z.string().uuid('Invalid fleet ID format'),
    mode: z.nativeEnum(GameMode, {
        errorMap: () => ({ message: 'Invalid game mode selected' })
    }),
    config: z.object({
        maxPlayers: z.number()
            .min(2, 'Minimum 2 players required')
            .max(MAX_PLAYERS, `Maximum ${MAX_PLAYERS} players allowed`),
        duration: z.number()
            .min(60, 'Minimum duration 60 seconds')
            .max(3600, 'Maximum duration 1 hour'),
        scoreLimit: z.number()
            .min(1, 'Score limit must be positive'),
        environmentSettings: z.record(z.any())
    })
});

/**
 * Type for game session creation payload
 */
export type CreateGameSessionPayload = z.infer<typeof gameSessionSchema>;

/**
 * Interface for game performance metrics
 */
export interface GameMetrics {
    /** Current frame rate */
    fps: number;

    /** State update latency in milliseconds */
    updateLatency: number;

    /** Number of active players */
    playerCount: number;

    /** Environment processing load (0-1) */
    environmentLoad: number;

    /** Network sync status */
    networkStatus: {
        connected: boolean;
        latency: number;
        packetLoss: number;
    };
}

/**
 * Interface for game event data
 */
export interface GameEvent {
    /** Event type identifier */
    type: string;

    /** Event timestamp */
    timestamp: number;

    /** Associated player ID (if applicable) */
    playerId?: string;

    /** Event-specific data */
    data: Record<string, any>;

    /** Event sequence number */
    sequence: number;
}