import { z } from 'zod'; // v3.x
import { User } from './user.types';

/**
 * Enumeration of possible fleet states
 */
export enum FleetStatus {
    INITIALIZING = 'INITIALIZING',  // Initial fleet setup
    ACTIVE = 'ACTIVE',              // Fleet is operational
    SYNCING = 'SYNCING',            // Fleet is synchronizing state
    DISCONNECTED = 'DISCONNECTED'   // Fleet connection lost
}

/**
 * Device hardware capabilities interface
 */
export interface DeviceCapabilities {
    lidarResolution: number;    // Range: 0.01-1.00 cm
    scanRange: number;          // Range: 1-5 meters
    processingPower: number;    // Relative processing capability score
}

/**
 * Device information within a fleet
 */
export interface FleetDevice {
    deviceId: string;           // Unique device identifier
    userId: string;             // Associated user ID
    status: FleetStatus;        // Current device status
    lastSeen: Date;            // Last activity timestamp
    capabilities: DeviceCapabilities;  // Device hardware specs
}

/**
 * Core fleet data structure
 */
export interface Fleet {
    id: string;                 // Unique fleet identifier
    name: string;               // Fleet display name
    hostDeviceId: string;       // Host device identifier
    status: FleetStatus;        // Current fleet status
    devices: FleetDevice[];     // Connected devices
    maxDevices: number;         // Maximum allowed devices (up to 32)
    createdAt: Date;           // Fleet creation timestamp
    updatedAt: Date;           // Last update timestamp
}

/**
 * Data transfer object for fleet creation
 */
export interface CreateFleetDTO {
    name: string;              // Fleet name
    hostDeviceId: string;      // Initial host device
    maxDevices: number;        // Maximum device limit
}

/**
 * Data transfer object for fleet updates
 */
export interface UpdateFleetDTO {
    name?: string;             // Updated fleet name
    status?: FleetStatus;      // Updated fleet status
}

/**
 * Zod validation schema for fleet data
 */
export const fleetSchema = z.object({
    name: z.string()
        .min(3, 'Fleet name must be at least 3 characters')
        .max(50, 'Fleet name cannot exceed 50 characters'),
    hostDeviceId: z.string()
        .uuid('Host device ID must be a valid UUID'),
    maxDevices: z.number()
        .min(2, 'Fleet must allow at least 2 devices')
        .max(32, 'Fleet cannot exceed 32 devices'),
});

/**
 * Zod validation schema for device capabilities
 */
export const deviceCapabilitiesSchema = z.object({
    lidarResolution: z.number()
        .min(0.01, 'LiDAR resolution must be at least 0.01cm')
        .max(1.00, 'LiDAR resolution cannot exceed 1.00cm'),
    scanRange: z.number()
        .min(1, 'Scan range must be at least 1 meter')
        .max(5, 'Scan range cannot exceed 5 meters'),
    processingPower: z.number()
        .min(0, 'Processing power cannot be negative')
        .max(100, 'Processing power cannot exceed 100')
});

/**
 * Zod validation schema for fleet device
 */
export const fleetDeviceSchema = z.object({
    deviceId: z.string().uuid('Device ID must be a valid UUID'),
    userId: z.string().uuid('User ID must be a valid UUID'),
    status: z.nativeEnum(FleetStatus),
    lastSeen: z.date(),
    capabilities: deviceCapabilitiesSchema
});

/**
 * Type guard to check if an object is a valid Fleet
 */
export function isFleet(obj: any): obj is Fleet {
    return (
        typeof obj === 'object' &&
        typeof obj.id === 'string' &&
        typeof obj.name === 'string' &&
        typeof obj.hostDeviceId === 'string' &&
        Object.values(FleetStatus).includes(obj.status) &&
        Array.isArray(obj.devices) &&
        obj.devices.every((device: any) => isFleetDevice(device)) &&
        typeof obj.maxDevices === 'number' &&
        obj.maxDevices >= 2 &&
        obj.maxDevices <= 32 &&
        obj.createdAt instanceof Date &&
        obj.updatedAt instanceof Date
    );
}

/**
 * Type guard to check if an object is a valid FleetDevice
 */
export function isFleetDevice(obj: any): obj is FleetDevice {
    return (
        typeof obj === 'object' &&
        typeof obj.deviceId === 'string' &&
        typeof obj.userId === 'string' &&
        Object.values(FleetStatus).includes(obj.status) &&
        obj.lastSeen instanceof Date &&
        isDeviceCapabilities(obj.capabilities)
    );
}

/**
 * Type guard to check if an object has valid DeviceCapabilities
 */
export function isDeviceCapabilities(obj: any): obj is DeviceCapabilities {
    return (
        typeof obj === 'object' &&
        typeof obj.lidarResolution === 'number' &&
        obj.lidarResolution >= 0.01 &&
        obj.lidarResolution <= 1.00 &&
        typeof obj.scanRange === 'number' &&
        obj.scanRange >= 1 &&
        obj.scanRange <= 5 &&
        typeof obj.processingPower === 'number' &&
        obj.processingPower >= 0 &&
        obj.processingPower <= 100
    );
}