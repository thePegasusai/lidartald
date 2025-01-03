import { describe, it, beforeEach, afterEach, expect, jest } from 'jest';
import MockRedis from 'ioredis-mock';
import { FleetService } from '../services/fleet.service';
import { 
    Fleet,
    FleetStatus,
    FleetDevice,
    CreateFleetDTO,
    NetworkCondition
} from '../types/fleet.types';

// Constants for performance testing
const LATENCY_THRESHOLD = 50; // 50ms max latency as per spec
const SYNC_RATE = 30; // 30Hz update rate
const MAX_FLEET_SIZE = 32;

// Mock device data for testing
const mockDeviceCapabilities = {
    lidarResolution: 0.01,
    scanRange: 5,
    processingPower: 85
};

const mockDeviceData: FleetDevice = {
    deviceId: '123e4567-e89b-12d3-a456-426614174000',
    userId: '123e4567-e89b-12d3-a456-426614174001',
    status: FleetStatus.INITIALIZING,
    lastSeen: new Date(),
    capabilities: mockDeviceCapabilities
};

// Mock fleet data
const mockFleetData: CreateFleetDTO = {
    name: 'Test Fleet',
    hostDeviceId: mockDeviceData.deviceId,
    maxDevices: MAX_FLEET_SIZE
};

describe('FleetService Integration Tests', () => {
    let fleetService: FleetService;
    let mockRedis: MockRedis;
    let createdFleet: Fleet;

    beforeEach(async () => {
        // Initialize mock Redis
        mockRedis = new MockRedis({
            data: new Map(),
            keyPrefix: 'test:',
        });

        // Reset FleetService singleton with mocked dependencies
        (FleetService as any).instance = undefined;
        fleetService = FleetService.getInstance();
        (fleetService as any).redisClient = mockRedis;

        // Create test fleet
        createdFleet = await fleetService.createFleet(mockFleetData);
    });

    afterEach(async () => {
        // Cleanup
        await mockRedis.flushall();
        await fleetService.disconnect();
        jest.clearAllMocks();
    });

    describe('Fleet Creation and Management', () => {
        it('should create a fleet with correct initial state', async () => {
            expect(createdFleet).toBeDefined();
            expect(createdFleet.name).toBe(mockFleetData.name);
            expect(createdFleet.hostDeviceId).toBe(mockFleetData.hostDeviceId);
            expect(createdFleet.status).toBe(FleetStatus.INITIALIZING);
            expect(createdFleet.devices).toHaveLength(0);
            expect(createdFleet.maxDevices).toBeLessThanOrEqual(MAX_FLEET_SIZE);
        });

        it('should enforce maximum fleet size limit', async () => {
            const oversizedFleet: CreateFleetDTO = {
                ...mockFleetData,
                maxDevices: MAX_FLEET_SIZE + 1
            };

            const fleet = await fleetService.createFleet(oversizedFleet);
            expect(fleet.maxDevices).toBe(MAX_FLEET_SIZE);
        });

        it('should handle fleet state transitions correctly', async () => {
            await fleetService.syncFleetState(createdFleet.id, {
                fleetId: createdFleet.id,
                deviceId: mockDeviceData.deviceId,
                timestamp: Date.now(),
                data: { status: FleetStatus.ACTIVE },
                version: 1
            });

            const updatedFleet = await fleetService.getFleetState(createdFleet.id);
            expect(updatedFleet.status).toBe(FleetStatus.ACTIVE);
        });
    });

    describe('Fleet Network Performance', () => {
        it('should maintain sync rate of 30Hz', async () => {
            const startTime = Date.now();
            let updateCount = 0;
            const testDuration = 1000; // 1 second test

            while (Date.now() - startTime < testDuration) {
                await fleetService.syncFleetState(createdFleet.id, {
                    fleetId: createdFleet.id,
                    deviceId: mockDeviceData.deviceId,
                    timestamp: Date.now(),
                    data: { test: 'data' },
                    version: updateCount + 1
                });
                updateCount++;
                await new Promise(resolve => setTimeout(resolve, 1000 / SYNC_RATE));
            }

            expect(updateCount).toBeGreaterThanOrEqual(SYNC_RATE - 1);
        });

        it('should maintain latency under 50ms', async () => {
            const latencies: number[] = [];

            for (let i = 0; i < 100; i++) {
                const startTime = Date.now();
                await fleetService.syncFleetState(createdFleet.id, {
                    fleetId: createdFleet.id,
                    deviceId: mockDeviceData.deviceId,
                    timestamp: startTime,
                    data: { test: 'latency' },
                    version: i + 1
                });
                latencies.push(Date.now() - startTime);
            }

            const maxLatency = Math.max(...latencies);
            expect(maxLatency).toBeLessThanOrEqual(LATENCY_THRESHOLD);
        });

        it('should handle network degradation gracefully', async () => {
            const poorNetworkCondition: NetworkCondition = {
                latency: 100,
                packetLoss: 0.1,
                jitter: 20
            };

            await fleetService.simulateNetworkConditions(poorNetworkCondition);
            const latency = await fleetService.measureLatency(createdFleet.id);

            expect(latency).toBeDefined();
            expect(typeof latency).toBe('number');
        });
    });

    describe('Fleet Capacity and Scaling', () => {
        it('should handle maximum fleet capacity', async () => {
            const devices: FleetDevice[] = [];

            // Add maximum number of devices
            for (let i = 0; i < MAX_FLEET_SIZE; i++) {
                const device: FleetDevice = {
                    ...mockDeviceData,
                    deviceId: `device-${i}`,
                    userId: `user-${i}`
                };
                devices.push(device);
                await fleetService.joinFleet(createdFleet.id, device);
            }

            const fleet = await fleetService.getFleetState(createdFleet.id);
            expect(fleet.devices).toHaveLength(MAX_FLEET_SIZE);

            // Attempt to add one more device
            await expect(
                fleetService.joinFleet(createdFleet.id, {
                    ...mockDeviceData,
                    deviceId: 'overflow-device',
                    userId: 'overflow-user'
                })
            ).rejects.toThrow();
        });

        it('should maintain performance with maximum fleet size', async () => {
            // Add maximum number of devices
            for (let i = 0; i < MAX_FLEET_SIZE; i++) {
                await fleetService.joinFleet(createdFleet.id, {
                    ...mockDeviceData,
                    deviceId: `device-${i}`,
                    userId: `user-${i}`
                });
            }

            const startTime = Date.now();
            await fleetService.syncFleetState(createdFleet.id, {
                fleetId: createdFleet.id,
                deviceId: mockDeviceData.deviceId,
                timestamp: Date.now(),
                data: { test: 'scaling' },
                version: 1
            });

            const syncTime = Date.now() - startTime;
            expect(syncTime).toBeLessThanOrEqual(LATENCY_THRESHOLD);
        });
    });

    describe('Error Handling and Recovery', () => {
        it('should handle device disconnection gracefully', async () => {
            await fleetService.joinFleet(createdFleet.id, mockDeviceData);
            await fleetService.leaveFleet(createdFleet.id, mockDeviceData.deviceId);

            const fleet = await fleetService.getFleetState(createdFleet.id);
            expect(fleet.devices).not.toContainEqual(
                expect.objectContaining({ deviceId: mockDeviceData.deviceId })
            );
        });

        it('should recover from sync failures', async () => {
            // Simulate sync failure
            jest.spyOn(mockRedis, 'set').mockRejectedValueOnce(new Error('Sync failed'));

            await expect(
                fleetService.syncFleetState(createdFleet.id, {
                    fleetId: createdFleet.id,
                    deviceId: mockDeviceData.deviceId,
                    timestamp: Date.now(),
                    data: { test: 'recovery' },
                    version: 1
                })
            ).resolves.not.toThrow();
        });
    });
});