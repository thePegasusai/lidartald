import { z } from 'zod'; // v3.21.4
import { FleetStatus } from '../../social_engine/src/types/fleet.types';
import { ApiError } from '../middleware/error-handler';

/**
 * Schema for validating device capabilities in fleet operations
 */
const deviceCapabilitiesSchema = z.object({
    lidarVersion: z.string()
        .regex(/^\d+\.\d+\.\d+$/, 'Invalid LiDAR version format'),
    meshProtocol: z.string()
        .regex(/^(WebRTC|LibP2P)-\d+\.\d+$/, 'Invalid mesh protocol format'),
    bandwidth: z.number()
        .min(1, 'Minimum bandwidth requirement not met')
        .max(1000, 'Bandwidth exceeds maximum limit'),
    scanRate: z.number()
        .min(30, 'Minimum scan rate of 30Hz required')
        .max(120, 'Scan rate cannot exceed 120Hz'),
    lidarResolution: z.number()
        .min(0.01, 'LiDAR resolution must be at least 0.01cm')
        .max(1.00, 'LiDAR resolution cannot exceed 1.00cm'),
    scanRange: z.number()
        .min(1, 'Scan range must be at least 1 meter')
        .max(5, 'Scan range cannot exceed 5 meters')
});

/**
 * Schema for fleet creation requests
 */
export const createFleetSchema = z.object({
    name: z.string()
        .min(3, 'Fleet name must be at least 3 characters')
        .max(50, 'Fleet name cannot exceed 50 characters')
        .regex(/^[a-zA-Z0-9-_\s]+$/, 'Fleet name contains invalid characters'),
    hostDeviceId: z.string()
        .uuid('Host device ID must be a valid UUID'),
    maxDevices: z.number()
        .int('Maximum devices must be an integer')
        .min(2, 'Fleet must allow at least 2 devices')
        .max(32, 'Fleet cannot exceed 32 devices'),
    capabilities: deviceCapabilitiesSchema
});

/**
 * Schema for fleet update requests
 */
export const updateFleetSchema = z.object({
    name: z.string()
        .min(3, 'Fleet name must be at least 3 characters')
        .max(50, 'Fleet name cannot exceed 50 characters')
        .regex(/^[a-zA-Z0-9-_\s]+$/, 'Fleet name contains invalid characters')
        .optional(),
    status: z.nativeEnum(FleetStatus)
        .optional(),
    settings: z.object({
        scanInterval: z.number()
            .min(100, 'Scan interval must be at least 100ms')
            .max(1000, 'Scan interval cannot exceed 1000ms'),
        syncMode: z.enum(['real-time', 'batch']),
        meshTopology: z.enum(['full', 'star', 'ring'])
            .optional(),
        autoReconnect: z.boolean()
            .optional()
    }).optional()
});

/**
 * Schema for adding devices to a fleet
 */
export const addDeviceSchema = z.object({
    deviceId: z.string()
        .uuid('Device ID must be a valid UUID'),
    capabilities: deviceCapabilitiesSchema,
    meshPosition: z.object({
        distance: z.number()
            .max(5, 'Device must be within 5 meters range'),
        signalStrength: z.number()
            .min(-90, 'Signal strength too weak')
            .max(-30, 'Signal strength exceeds maximum')
    })
});

/**
 * Validates fleet creation request payload
 */
export async function validateCreateFleet(payload: unknown): Promise<void> {
    try {
        const validatedData = await createFleetSchema.parseAsync(payload);
        
        // Additional validation for host device capabilities
        if (validatedData.capabilities.scanRate < 30) {
            throw new ApiError(400, 'Host device must support minimum 30Hz scan rate', {
                field: 'capabilities.scanRate',
                required: 30,
                provided: validatedData.capabilities.scanRate
            });
        }

        // Validate mesh network requirements
        if (!validatedData.capabilities.meshProtocol.startsWith('WebRTC')) {
            throw new ApiError(400, 'Host device must support WebRTC mesh protocol', {
                field: 'capabilities.meshProtocol',
                required: 'WebRTC',
                provided: validatedData.capabilities.meshProtocol
            });
        }
    } catch (error) {
        if (error instanceof z.ZodError) {
            throw new ApiError(400, 'Invalid fleet creation parameters', {
                validation_errors: error.errors
            });
        }
        throw error;
    }
}

/**
 * Validates fleet update request payload
 */
export async function validateUpdateFleet(payload: unknown): Promise<void> {
    try {
        const validatedData = await updateFleetSchema.parseAsync(payload);

        // Validate state transitions
        if (validatedData.status) {
            const validTransitions: Record<FleetStatus, FleetStatus[]> = {
                [FleetStatus.INITIALIZING]: [FleetStatus.ACTIVE, FleetStatus.DISCONNECTED],
                [FleetStatus.ACTIVE]: [FleetStatus.SYNCING, FleetStatus.DISCONNECTED],
                [FleetStatus.SYNCING]: [FleetStatus.ACTIVE, FleetStatus.DISCONNECTED],
                [FleetStatus.DISCONNECTED]: [FleetStatus.INITIALIZING]
            };

            // Note: Current fleet status would be retrieved from the fleet service
            // This is a placeholder for demonstration
            const currentStatus = FleetStatus.ACTIVE;

            if (!validTransitions[currentStatus].includes(validatedData.status)) {
                throw new ApiError(400, 'Invalid fleet status transition', {
                    current: currentStatus,
                    requested: validatedData.status,
                    allowed: validTransitions[currentStatus]
                });
            }
        }
    } catch (error) {
        if (error instanceof z.ZodError) {
            throw new ApiError(400, 'Invalid fleet update parameters', {
                validation_errors: error.errors
            });
        }
        throw error;
    }
}

/**
 * Validates device addition request payload
 */
export async function validateAddDevice(payload: unknown): Promise<void> {
    try {
        const validatedData = await addDeviceSchema.parseAsync(payload);

        // Validate device proximity
        if (validatedData.meshPosition.distance > 5) {
            throw new ApiError(400, 'Device is out of fleet range', {
                field: 'meshPosition.distance',
                maximum: 5,
                provided: validatedData.meshPosition.distance
            });
        }

        // Validate signal strength for reliable mesh networking
        if (validatedData.meshPosition.signalStrength < -70) {
            throw new ApiError(400, 'Device signal strength too weak for reliable operation', {
                field: 'meshPosition.signalStrength',
                minimum: -70,
                provided: validatedData.meshPosition.signalStrength
            });
        }

        // Validate device capabilities compatibility
        if (validatedData.capabilities.scanRate < 30) {
            throw new ApiError(400, 'Device must support minimum 30Hz scan rate', {
                field: 'capabilities.scanRate',
                required: 30,
                provided: validatedData.capabilities.scanRate
            });
        }
    } catch (error) {
        if (error instanceof z.ZodError) {
            throw new ApiError(400, 'Invalid device addition parameters', {
                validation_errors: error.errors
            });
        }
        throw error;
    }
}