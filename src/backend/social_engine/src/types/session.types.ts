import { z } from 'zod'; // v3.x
import { User } from './user.types';
import { Fleet } from './fleet.types';

/**
 * Enumeration of possible session states
 */
export enum SessionStatus {
    INITIALIZING = 'INITIALIZING',  // Initial setup and environment scanning
    SCANNING = 'SCANNING',          // Active environment scanning
    ACTIVE = 'ACTIVE',             // Game session in progress
    PAUSED = 'PAUSED',             // Session temporarily paused
    COMPLETED = 'COMPLETED',        // Session finished successfully
    ERROR = 'ERROR'                // Session encountered an error
}

/**
 * Types of gaming sessions available
 */
export enum SessionType {
    SOLO = 'SOLO',           // Single player session
    FLEET = 'FLEET',         // Multi-device fleet session
    TOURNAMENT = 'TOURNAMENT' // Competitive tournament session
}

/**
 * Interface for tracking individual session participants
 */
export interface SessionParticipant {
    userId: string;                 // User identifier
    deviceId: string;              // Device identifier
    joinedAt: Date;               // Participation start time
    status: SessionStatus;         // Current participant status
    score: number;                // Participant's current score
    lastSyncTime: Date;           // Last state synchronization
    position: {                   // Real-time position in environment
        x: number;
        y: number;
        z: number;
    };
}

/**
 * Configuration settings for gaming sessions
 */
export interface SessionSettings {
    maxParticipants: number;        // Maximum allowed participants (2-32)
    scanResolution: number;         // LiDAR scan resolution (0.01-1.00)
    scanRange: number;              // Scan range in meters (1-5)
    autoSync: boolean;              // Auto-sync environment data
    syncInterval: number;           // Sync frequency in milliseconds
    environmentUpdateRate: number;  // Environment refresh rate in Hz
}

/**
 * Environment mapping and feature data
 */
export interface EnvironmentData {
    pointCloud: Buffer;            // Raw point cloud data
    features: Feature[];           // Detected environmental features
    boundaries: Boundary[];        // Environment boundaries
    lastUpdate: Date;             // Last environment update
}

/**
 * Environmental feature detection
 */
interface Feature {
    id: string;                   // Feature identifier
    type: string;                 // Feature classification
    coordinates: {                // 3D coordinates
        x: number;
        y: number;
        z: number;
    };
    confidence: number;           // Detection confidence (0-1)
}

/**
 * Environment boundary definition
 */
interface Boundary {
    id: string;                   // Boundary identifier
    points: Array<{               // Boundary vertices
        x: number;
        y: number;
        z: number;
    }>;
    type: 'wall' | 'floor' | 'ceiling' | 'obstacle';
}

/**
 * Core session data structure
 */
export interface Session {
    id: string;                   // Session identifier
    type: SessionType;            // Session type
    status: SessionStatus;        // Current session status
    fleetId: string;             // Associated fleet identifier
    participants: SessionParticipant[]; // Session participants
    environmentId: string;        // Environment identifier
    startedAt: Date;             // Session start time
    endedAt: Date | null;        // Session end time
    settings: SessionSettings;    // Session configuration
    environmentData: EnvironmentData; // Environment state
    lastSyncTimestamp: Date;     // Last sync timestamp
}

/**
 * Zod validation schema for session settings
 */
export const sessionSettingsSchema = z.object({
    maxParticipants: z.number()
        .min(2, 'Must allow at least 2 participants')
        .max(32, 'Cannot exceed 32 participants'),
    scanResolution: z.number()
        .min(0.01, 'Resolution must be at least 0.01')
        .max(1.00, 'Resolution cannot exceed 1.00'),
    scanRange: z.number()
        .min(1, 'Range must be at least 1 meter')
        .max(5, 'Range cannot exceed 5 meters'),
    autoSync: z.boolean(),
    syncInterval: z.number()
        .min(100, 'Sync interval must be at least 100ms')
        .max(5000, 'Sync interval cannot exceed 5000ms'),
    environmentUpdateRate: z.number()
        .min(1, 'Update rate must be at least 1Hz')
        .max(30, 'Update rate cannot exceed 30Hz')
});

/**
 * Zod validation schema for session data
 */
export const sessionSchema = z.object({
    type: z.nativeEnum(SessionType),
    fleetId: z.string().uuid('Fleet ID must be a valid UUID'),
    settings: sessionSettingsSchema,
    environmentData: z.object({
        pointCloud: z.instanceof(Buffer),
        features: z.array(z.object({
            id: z.string().uuid(),
            type: z.string(),
            coordinates: z.object({
                x: z.number(),
                y: z.number(),
                z: z.number()
            }),
            confidence: z.number().min(0).max(1)
        })),
        boundaries: z.array(z.object({
            id: z.string().uuid(),
            points: z.array(z.object({
                x: z.number(),
                y: z.number(),
                z: z.number()
            })),
            type: z.enum(['wall', 'floor', 'ceiling', 'obstacle'])
        })),
        lastUpdate: z.date()
    })
});

/**
 * Type guard to check if an object is a valid Session
 */
export function isSession(obj: any): obj is Session {
    return (
        typeof obj === 'object' &&
        typeof obj.id === 'string' &&
        Object.values(SessionType).includes(obj.type) &&
        Object.values(SessionStatus).includes(obj.status) &&
        typeof obj.fleetId === 'string' &&
        Array.isArray(obj.participants) &&
        obj.participants.every(isSessionParticipant) &&
        typeof obj.environmentId === 'string' &&
        obj.startedAt instanceof Date &&
        (obj.endedAt === null || obj.endedAt instanceof Date) &&
        isSessionSettings(obj.settings) &&
        isEnvironmentData(obj.environmentData) &&
        obj.lastSyncTimestamp instanceof Date
    );
}

/**
 * Type guard for session participant validation
 */
function isSessionParticipant(obj: any): obj is SessionParticipant {
    return (
        typeof obj === 'object' &&
        typeof obj.userId === 'string' &&
        typeof obj.deviceId === 'string' &&
        obj.joinedAt instanceof Date &&
        Object.values(SessionStatus).includes(obj.status) &&
        typeof obj.score === 'number' &&
        obj.lastSyncTime instanceof Date &&
        typeof obj.position === 'object' &&
        typeof obj.position.x === 'number' &&
        typeof obj.position.y === 'number' &&
        typeof obj.position.z === 'number'
    );
}

/**
 * Type guard for session settings validation
 */
function isSessionSettings(obj: any): obj is SessionSettings {
    return (
        typeof obj === 'object' &&
        typeof obj.maxParticipants === 'number' &&
        typeof obj.scanResolution === 'number' &&
        typeof obj.scanRange === 'number' &&
        typeof obj.autoSync === 'boolean' &&
        typeof obj.syncInterval === 'number' &&
        typeof obj.environmentUpdateRate === 'number'
    );
}

/**
 * Type guard for environment data validation
 */
function isEnvironmentData(obj: any): obj is EnvironmentData {
    return (
        typeof obj === 'object' &&
        obj.pointCloud instanceof Buffer &&
        Array.isArray(obj.features) &&
        obj.features.every(isFeature) &&
        Array.isArray(obj.boundaries) &&
        obj.boundaries.every(isBoundary) &&
        obj.lastUpdate instanceof Date
    );
}

/**
 * Type guard for feature validation
 */
function isFeature(obj: any): obj is Feature {
    return (
        typeof obj === 'object' &&
        typeof obj.id === 'string' &&
        typeof obj.type === 'string' &&
        typeof obj.coordinates === 'object' &&
        typeof obj.coordinates.x === 'number' &&
        typeof obj.coordinates.y === 'number' &&
        typeof obj.coordinates.z === 'number' &&
        typeof obj.confidence === 'number' &&
        obj.confidence >= 0 &&
        obj.confidence <= 1
    );
}

/**
 * Type guard for boundary validation
 */
function isBoundary(obj: any): obj is Boundary {
    return (
        typeof obj === 'object' &&
        typeof obj.id === 'string' &&
        Array.isArray(obj.points) &&
        obj.points.every((point: any) =>
            typeof point === 'object' &&
            typeof point.x === 'number' &&
            typeof point.y === 'number' &&
            typeof point.z === 'number'
        ) &&
        ['wall', 'floor', 'ceiling', 'obstacle'].includes(obj.type)
    );
}