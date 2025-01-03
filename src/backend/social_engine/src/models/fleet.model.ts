import { PrismaClient } from '@prisma/client'; // v4.x
import { z } from 'zod'; // v3.x
import { 
    Fleet, 
    FleetStatus, 
    FleetDevice, 
    CreateFleetDTO, 
    UpdateFleetDTO, 
    fleetSchema 
} from '../types/fleet.types';

/**
 * Connection pool configuration for database operations
 */
interface ConnectionPool {
    min: number;
    max: number;
    idle: number;
}

/**
 * Singleton data access layer implementing comprehensive fleet management
 * with transaction support, validation, and proper error handling
 */
export class FleetModel {
    private static instance: FleetModel;
    private readonly prisma: PrismaClient;
    private readonly MAX_FLEET_SIZE = 32;
    private readonly connectionPool: ConnectionPool = {
        min: 5,
        max: 20,
        idle: 10000
    };

    /**
     * Private constructor implementing singleton pattern with connection pooling
     */
    private constructor() {
        this.prisma = new PrismaClient({
            log: ['error', 'warn'],
            errorFormat: 'minimal',
            connectionTimeout: 5000,
            pool: this.connectionPool
        });

        // Setup error handlers
        this.prisma.$on('error', (error) => {
            console.error('Database error:', error);
        });

        // Verify database connection
        this.verifyConnection();
    }

    /**
     * Verifies database connection is healthy
     */
    private async verifyConnection(): Promise<void> {
        try {
            await this.prisma.$connect();
        } catch (error) {
            console.error('Failed to connect to database:', error);
            throw new Error('Database connection failed');
        }
    }

    /**
     * Gets or creates singleton instance
     */
    public static getInstance(): FleetModel {
        if (!FleetModel.instance) {
            FleetModel.instance = new FleetModel();
        }
        return FleetModel.instance;
    }

    /**
     * Creates new fleet with validation and transaction support
     */
    public async createFleet(fleetData: CreateFleetDTO): Promise<Fleet> {
        try {
            // Validate fleet data
            const validatedData = fleetSchema.parse(fleetData);

            return await this.prisma.$transaction(async (tx) => {
                // Create fleet record
                const fleet = await tx.fleet.create({
                    data: {
                        name: validatedData.name,
                        hostDeviceId: validatedData.hostDeviceId,
                        status: FleetStatus.INITIALIZING,
                        maxDevices: Math.min(validatedData.maxDevices, this.MAX_FLEET_SIZE),
                        devices: {
                            create: [] // Initial empty devices array
                        }
                    },
                    include: {
                        devices: true
                    }
                });

                return fleet as Fleet;
            });
        } catch (error) {
            if (error instanceof z.ZodError) {
                throw new Error(`Validation error: ${error.message}`);
            }
            throw new Error(`Failed to create fleet: ${error.message}`);
        }
    }

    /**
     * Updates fleet with validation and state transition checks
     */
    public async updateFleet(fleetId: string, updateData: UpdateFleetDTO): Promise<Fleet> {
        try {
            // Verify fleet exists
            const existingFleet = await this.getFleetById(fleetId);
            if (!existingFleet) {
                throw new Error('Fleet not found');
            }

            // Validate state transition if status is being updated
            if (updateData.status) {
                this.validateStateTransition(existingFleet.status, updateData.status);
            }

            return await this.prisma.$transaction(async (tx) => {
                const updatedFleet = await tx.fleet.update({
                    where: { id: fleetId },
                    data: {
                        ...(updateData.name && { name: updateData.name }),
                        ...(updateData.status && { status: updateData.status })
                    },
                    include: {
                        devices: true
                    }
                });

                return updatedFleet as Fleet;
            });
        } catch (error) {
            throw new Error(`Failed to update fleet: ${error.message}`);
        }
    }

    /**
     * Adds member to fleet with capacity and state validation
     */
    public async addMember(fleetId: string, userId: string, deviceData: FleetDevice): Promise<Fleet> {
        try {
            const fleet = await this.getFleetById(fleetId);
            if (!fleet) {
                throw new Error('Fleet not found');
            }

            // Check fleet capacity
            if (fleet.devices.length >= this.MAX_FLEET_SIZE) {
                throw new Error('Fleet has reached maximum capacity');
            }

            // Verify fleet state allows new members
            if (fleet.status === FleetStatus.DISCONNECTED) {
                throw new Error('Cannot add members to disconnected fleet');
            }

            return await this.prisma.$transaction(async (tx) => {
                const updatedFleet = await tx.fleet.update({
                    where: { id: fleetId },
                    data: {
                        devices: {
                            create: {
                                deviceId: deviceData.deviceId,
                                userId: userId,
                                status: FleetStatus.INITIALIZING,
                                lastSeen: new Date(),
                                capabilities: deviceData.capabilities
                            }
                        }
                    },
                    include: {
                        devices: true
                    }
                });

                return updatedFleet as Fleet;
            });
        } catch (error) {
            throw new Error(`Failed to add fleet member: ${error.message}`);
        }
    }

    /**
     * Removes member with cleanup and state management
     */
    public async removeMember(fleetId: string, userId: string): Promise<Fleet> {
        try {
            const fleet = await this.getFleetById(fleetId);
            if (!fleet) {
                throw new Error('Fleet not found');
            }

            return await this.prisma.$transaction(async (tx) => {
                // Remove member
                const updatedFleet = await tx.fleet.update({
                    where: { id: fleetId },
                    data: {
                        devices: {
                            deleteMany: {
                                userId: userId
                            }
                        }
                    },
                    include: {
                        devices: true
                    }
                });

                // Update fleet status if no members remain
                if (updatedFleet.devices.length === 0) {
                    await tx.fleet.update({
                        where: { id: fleetId },
                        data: {
                            status: FleetStatus.DISCONNECTED
                        }
                    });
                }

                return updatedFleet as Fleet;
            });
        } catch (error) {
            throw new Error(`Failed to remove fleet member: ${error.message}`);
        }
    }

    /**
     * Retrieves fleet by ID with related data
     */
    public async getFleetById(fleetId: string): Promise<Fleet | null> {
        try {
            const fleet = await this.prisma.fleet.findUnique({
                where: { id: fleetId },
                include: {
                    devices: true
                }
            });

            return fleet as Fleet | null;
        } catch (error) {
            throw new Error(`Failed to retrieve fleet: ${error.message}`);
        }
    }

    /**
     * Validates fleet state transitions
     */
    private validateStateTransition(currentState: FleetStatus, newState: FleetStatus): void {
        const validTransitions = {
            [FleetStatus.INITIALIZING]: [FleetStatus.ACTIVE, FleetStatus.DISCONNECTED],
            [FleetStatus.ACTIVE]: [FleetStatus.SYNCING, FleetStatus.DISCONNECTED],
            [FleetStatus.SYNCING]: [FleetStatus.ACTIVE, FleetStatus.DISCONNECTED],
            [FleetStatus.DISCONNECTED]: [FleetStatus.INITIALIZING]
        };

        if (!validTransitions[currentState].includes(newState)) {
            throw new Error(`Invalid state transition from ${currentState} to ${newState}`);
        }
    }

    /**
     * Cleanup resources on application shutdown
     */
    public async disconnect(): Promise<void> {
        await this.prisma.$disconnect();
    }
}