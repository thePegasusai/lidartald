import { z } from 'zod'; // v3.21.4
import { Fleet, FleetStatus, fleetSchema } from '../types/fleet.types';
import { apiClient, apiEndpoints } from '../config/api';
import { FLEET_CONSTANTS } from '../config/constants';

/**
 * Validation schema for nearby fleet search parameters
 */
const nearbyFleetSchema = z.object({
    range: z.number()
        .min(0.1, 'Range must be at least 0.1 meters')
        .max(FLEET_CONSTANTS.MAX_RANGE_M, `Range cannot exceed ${FLEET_CONSTANTS.MAX_RANGE_M} meters`),
    maxResults: z.number()
        .min(1)
        .max(FLEET_CONSTANTS.MAX_DEVICES)
});

/**
 * Creates a new fleet with the specified configuration
 * @param fleetData Fleet creation parameters
 * @returns Promise resolving to created Fleet
 */
export const createFleet = async (fleetData: z.infer<typeof fleetSchema>): Promise<Fleet> => {
    try {
        // Validate fleet data against schema
        const validatedData = fleetSchema.parse(fleetData);

        const response = await apiClient.post<Fleet>(
            apiEndpoints.fleet.create,
            validatedData,
            {
                validateSchema: fleetSchema,
                headers: {
                    'X-Fleet-Operation': 'create',
                    'X-Max-Devices': validatedData.maxDevices.toString()
                }
            }
        );

        return response.data;
    } catch (error) {
        console.error('[Fleet Creation Error]', error);
        throw error;
    }
};

/**
 * Joins an existing fleet with the current device
 * @param fleetId Target fleet identifier
 * @returns Promise resolving to updated Fleet
 */
export const joinFleet = async (fleetId: string): Promise<Fleet> => {
    try {
        const response = await apiClient.put<Fleet>(
            `${apiEndpoints.fleet.join}/${fleetId}`,
            {},
            {
                validateSchema: fleetSchema,
                headers: {
                    'X-Fleet-Operation': 'join',
                    'X-Device-Capabilities': JSON.stringify({
                        lidarResolution: FLEET_CONSTANTS.VALIDATION_RANGES.MIN_RESOLUTION_CM[0],
                        scanRange: FLEET_CONSTANTS.VALIDATION_RANGES.MAX_RANGE_M[0],
                        scanRate: FLEET_CONSTANTS.SYNC_INTERVAL_MS
                    })
                }
            }
        );

        return response.data;
    } catch (error) {
        console.error('[Fleet Join Error]', error);
        throw error;
    }
};

/**
 * Leaves the current fleet safely
 * @returns Promise resolving when fleet is left
 */
export const leaveFleet = async (): Promise<void> => {
    try {
        await apiClient.delete(apiEndpoints.fleet.status, {
            headers: {
                'X-Fleet-Operation': 'leave'
            }
        });
    } catch (error) {
        console.error('[Fleet Leave Error]', error);
        throw error;
    }
};

/**
 * Retrieves current fleet state with caching
 * @returns Promise resolving to current Fleet state
 */
export const getFleetState = async (): Promise<Fleet> => {
    try {
        const response = await apiClient.get<Fleet>(
            apiEndpoints.fleet.status,
            {
                validateSchema: fleetSchema,
                headers: {
                    'X-Fleet-Operation': 'status'
                }
            }
        );

        return response.data;
    } catch (error) {
        console.error('[Fleet State Error]', error);
        throw error;
    }
};

/**
 * Synchronizes fleet state with mesh network
 * @param fleetId Target fleet identifier
 * @returns Promise resolving to synchronized Fleet state
 */
export const syncFleetState = async (fleetId: string): Promise<Fleet> => {
    try {
        const response = await apiClient.post<Fleet>(
            `${apiEndpoints.fleet.sync}/${fleetId}`,
            {},
            {
                validateSchema: fleetSchema,
                headers: {
                    'X-Fleet-Operation': 'sync',
                    'X-Sync-Timestamp': Date.now().toString()
                },
                timeout: FLEET_CONSTANTS.SYNC_INTERVAL_MS
            }
        );

        return response.data;
    } catch (error) {
        console.error('[Fleet Sync Error]', error);
        throw error;
    }
};

/**
 * Retrieves nearby fleets within specified range
 * @param range Search range in meters
 * @param maxResults Maximum number of results to return
 * @returns Promise resolving to array of nearby Fleets
 */
export const getNearbyFleets = async (
    range: number = FLEET_CONSTANTS.MAX_RANGE_M,
    maxResults: number = FLEET_CONSTANTS.MAX_DEVICES
): Promise<Fleet[]> => {
    try {
        // Validate search parameters
        const validatedParams = nearbyFleetSchema.parse({ range, maxResults });

        const response = await apiClient.get<Fleet[]>(
            apiEndpoints.fleet.status,
            {
                params: validatedParams,
                headers: {
                    'X-Fleet-Operation': 'nearby',
                    'X-Search-Range': validatedParams.range.toString()
                }
            }
        );

        return response.data.filter(fleet => 
            fleet.status === FleetStatus.ACTIVE && 
            fleet.devices.length < fleet.maxDevices
        );
    } catch (error) {
        console.error('[Nearby Fleets Error]', error);
        throw error;
    }
};

/**
 * Retrieves fleet performance metrics
 * @param fleetId Target fleet identifier
 * @returns Promise resolving to fleet metrics
 */
export const getFleetMetrics = async (fleetId: string) => {
    try {
        const response = await apiClient.get(
            `${apiEndpoints.fleet.metrics}/${fleetId}`,
            {
                headers: {
                    'X-Fleet-Operation': 'metrics'
                }
            }
        );

        return response.data;
    } catch (error) {
        console.error('[Fleet Metrics Error]', error);
        throw error;
    }
};