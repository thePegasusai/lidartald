import { z } from 'zod'; // v3.x
import { EnvironmentMap } from './environment.types';

/**
 * Enumeration of possible roles within a fleet
 */
export enum FleetRole {
    HOST = 'HOST',
    MEMBER = 'MEMBER'
}

/**
 * Enumeration of possible fleet states
 */
export enum FleetStatus {
    INITIALIZING = 'INITIALIZING',
    ACTIVE = 'ACTIVE',
    SYNCING = 'SYNCING',
    DISCONNECTED = 'DISCONNECTED'
}

/**
 * Interface defining hardware capabilities of a device
 * Used for fleet coordination and task distribution
 */
export interface DeviceCapabilities {
    /** LiDAR scanning resolution in centimeters */
    lidarResolution: number;
    
    /** Maximum scanning range in meters */
    scanRange: number;
    
    /** Relative processing power score (1-100) */
    processingPower: number;
    
    /** Scanning rate in Hz */
    scanRate: number;
}

/**
 * Interface for fleet participant information including proximity data
 */
export interface FleetParticipant {
    /** Unique identifier for the participant */
    participantId: string;
    
    /** Display name for the participant */
    displayName: string;
    
    /** Real-time proximity information */
    proximityData: {
        /** Distance in meters */
        distance: number;
        /** Last proximity update timestamp */
        lastUpdate: number;
    };
}

/**
 * Interface for fleet member information
 */
export interface FleetMember {
    /** Unique identifier for the member */
    memberId: string;
    
    /** Member's role in the fleet */
    role: FleetRole;
    
    /** Associated device ID */
    deviceId: string;
    
    /** Join timestamp */
    joinedAt: Date;
}

/**
 * Interface for device information within a fleet
 */
export interface FleetDevice {
    /** Unique device identifier */
    deviceId: string;
    
    /** Associated participant ID */
    participantId: string;
    
    /** Current device status */
    status: FleetStatus;
    
    /** Last seen timestamp */
    lastSeen: Date;
    
    /** Device capabilities */
    capabilities: DeviceCapabilities;
    
    /** Current network latency in milliseconds */
    networkLatency: number;
}

/**
 * Core fleet interface containing comprehensive fleet management data
 */
export interface Fleet {
    /** Unique fleet identifier */
    id: string;
    
    /** Fleet display name */
    name: string;
    
    /** Host device identifier */
    hostDeviceId: string;
    
    /** Current fleet status */
    status: FleetStatus;
    
    /** Connected devices */
    devices: FleetDevice[];
    
    /** Fleet members */
    members: FleetMember[];
    
    /** Active participants */
    participants: FleetParticipant[];
    
    /** Associated environment map ID */
    environmentMapId: string;
    
    /** Last synchronization timestamp */
    lastSyncTimestamp: number;
    
    /** Maximum allowed devices (up to 32) */
    maxDevices: number;
    
    /** Mesh network status */
    meshNetworkStatus: {
        connected: boolean;
        latency: number;
    };
    
    /** Fleet creation timestamp */
    createdAt: Date;
    
    /** Last update timestamp */
    updatedAt: Date;
}

/**
 * Zod validation schema for fleet creation/updates
 */
export const fleetSchema = z.object({
    name: z.string()
        .min(3, 'Fleet name must be at least 3 characters')
        .max(50, 'Fleet name cannot exceed 50 characters'),
    
    hostDeviceId: z.string().uuid('Invalid device ID format'),
    
    maxDevices: z.number()
        .min(2, 'Fleet must allow at least 2 devices')
        .max(32, 'Fleet cannot exceed 32 devices'),
    
    environmentMapId: z.string().uuid('Invalid environment map ID').optional()
});

/**
 * Type for fleet creation payload
 */
export type CreateFleetPayload = z.infer<typeof fleetSchema>;

/**
 * Interface for fleet synchronization state
 */
export interface FleetSyncState {
    /** Last successful sync timestamp */
    lastSync: number;
    
    /** Pending updates count */
    pendingUpdates: number;
    
    /** Sync progress (0-100) */
    syncProgress: number;
    
    /** Current sync status */
    status: FleetStatus;
}

/**
 * Interface for fleet mesh network metrics
 */
export interface FleetNetworkMetrics {
    /** Average latency across all devices */
    averageLatency: number;
    
    /** Number of active connections */
    activeConnections: number;
    
    /** Network topology type */
    topology: 'mesh' | 'star' | 'hybrid';
    
    /** Network health score (0-100) */
    healthScore: number;
}