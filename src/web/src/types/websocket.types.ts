import { z } from 'zod'; // v3.21.4
import { Fleet } from './fleet.types';
import { GameStateUpdate } from './game.types';
import { UserLocation } from './user.types';

/**
 * Enumeration of all possible WebSocket event types
 * Covers all real-time communication scenarios in the TALD UNIA platform
 */
export enum WebSocketEventType {
    FLEET_UPDATE = 'FLEET_UPDATE',
    GAME_STATE = 'GAME_STATE',
    PROXIMITY_UPDATE = 'PROXIMITY_UPDATE',
    ENVIRONMENT_SYNC = 'ENVIRONMENT_SYNC',
    CONNECTION_STATE = 'CONNECTION_STATE',
    ERROR = 'ERROR'
}

/**
 * Interface for WebSocket error information
 * Provides detailed error context for handling and recovery
 */
export interface WebSocketError {
    code: number;
    message: string;
    timestamp: number;
    recoverable: boolean;
    retryCount?: number;
}

/**
 * Generic WebSocket message structure
 * Implements consistent message format across all event types
 */
export interface WebSocketMessage<T = unknown> {
    type: WebSocketEventType;
    payload: T;
    timestamp: number;
    deviceId: string;
    error: WebSocketError | null;
    messageId: string;
}

/**
 * Interface for fleet state update events
 * Implements CRDT-based state synchronization
 */
export interface FleetUpdateEvent {
    fleetId: string;
    status: Fleet['status'];
    deltaUpdates: Record<string, any>;
    version: number;
}

/**
 * Interface for proximity update events
 * Handles real-time user location updates with accuracy metrics
 */
export interface ProximityEvent {
    userId: string;
    distance: number;
    lastUpdate: number;
    accuracy: number;
}

/**
 * Interface for environment synchronization events
 * Manages real-time environment data sharing
 */
export interface EnvironmentSyncEvent {
    mapId: string;
    timestamp: number;
    deltaPoints: Array<{ x: number; y: number; z: number }>;
    compressionType: 'none' | 'lz4' | 'zstd';
    sequenceNumber: number;
}

/**
 * Interface for connection state events
 * Implements connection management and recovery strategy
 */
export interface ConnectionStateEvent {
    connected: boolean;
    latency: number;
    reconnectAttempt: number;
    backoffDelay: number;
    lastSuccessfulConnection: number;
}

/**
 * Constants for WebSocket configuration
 * Based on technical specifications and performance requirements
 */
export const WS_RECONNECT_ATTEMPTS = 5;
export const WS_RECONNECT_INTERVAL_MS = 1000;
export const WS_HEARTBEAT_INTERVAL_MS = 30000;
export const WS_MAX_MESSAGE_SIZE = 1048576; // 1MB
export const WS_COMPRESSION_THRESHOLD = 512; // Bytes

/**
 * Zod validation schema for WebSocket messages
 * Implements comprehensive runtime validation
 */
export const websocketMessageSchema = z.object({
    type: z.nativeEnum(WebSocketEventType, {
        errorMap: () => ({ message: 'Invalid WebSocket event type' })
    }),
    timestamp: z.number()
        .min(0, 'Timestamp must be positive')
        .max(Date.now() + 60000, 'Timestamp cannot be in the future'),
    deviceId: z.string()
        .uuid('Invalid device ID format'),
    messageId: z.string()
        .uuid('Invalid message ID format'),
    payload: z.any(),
    error: z.object({
        code: z.number(),
        message: z.string(),
        timestamp: z.number(),
        recoverable: z.boolean(),
        retryCount: z.number().optional()
    }).nullable()
});

/**
 * Type guard for checking WebSocket message validity
 */
export const isValidWebSocketMessage = (
    message: unknown
): message is WebSocketMessage => {
    try {
        websocketMessageSchema.parse(message);
        return true;
    } catch {
        return false;
    }
};

/**
 * Type guard for checking fleet update event validity
 */
export const isFleetUpdateEvent = (
    payload: unknown
): payload is FleetUpdateEvent => {
    return typeof payload === 'object' &&
        payload !== null &&
        'fleetId' in payload &&
        'status' in payload &&
        'version' in payload;
};

/**
 * Type guard for checking proximity event validity
 */
export const isProximityEvent = (
    payload: unknown
): payload is ProximityEvent => {
    return typeof payload === 'object' &&
        payload !== null &&
        'userId' in payload &&
        'distance' in payload &&
        'accuracy' in payload;
};

/**
 * Type guard for checking environment sync event validity
 */
export const isEnvironmentSyncEvent = (
    payload: unknown
): payload is EnvironmentSyncEvent => {
    return typeof payload === 'object' &&
        payload !== null &&
        'mapId' in payload &&
        'deltaPoints' in payload &&
        'sequenceNumber' in payload;
};

/**
 * Type guard for checking connection state event validity
 */
export const isConnectionStateEvent = (
    payload: unknown
): payload is ConnectionStateEvent => {
    return typeof payload === 'object' &&
        payload !== null &&
        'connected' in payload &&
        'latency' in payload &&
        'reconnectAttempt' in payload;
};