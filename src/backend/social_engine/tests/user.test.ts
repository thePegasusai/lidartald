import { jest } from '@jest/globals'; // v29.5.x
import supertest from 'supertest'; // v6.3.x
import { faker } from '@faker-js/faker'; // v8.0.x
import Redis from 'redis-mock'; // v0.56.x
import { mockHardwareSecurity } from '@testing-library/mock-hardware-security'; // v1.2.x

import { UserController } from '../src/controllers/user.controller';
import { UserModel } from '../src/models/user.model';
import { AuthService } from '../src/services/auth.service';
import { FleetStatus } from '../types/fleet.types';
import { User, UserProfile, CreateUserDTO } from '../types/user.types';

describe('User Management Tests', () => {
    let userController: UserController;
    let userModel: UserModel;
    let authService: AuthService;
    let redisClient: Redis.RedisClient;
    let hardwareMock: any;

    // Test data storage
    let testUsers: Map<string, User> = new Map();
    let testProfiles: Map<string, UserProfile> = new Map();

    beforeAll(async () => {
        // Initialize mocks
        hardwareMock = mockHardwareSecurity({
            securityLevel: 2,
            hardwareTokens: true,
            secureEnclave: true
        });

        // Setup Redis mock
        redisClient = Redis.createClient();

        // Initialize services with mocks
        userModel = new UserModel();
        authService = new AuthService(
            {} as any, // JWT service mock
            {} as any, // OAuth service mock
            {} as any, // RBAC service mock
            {} as any  // Security monitor mock
        );

        // Initialize controller
        userController = new UserController(
            userModel,
            authService,
            {} as any, // Matching service mock
            { host: 'localhost', port: 6379 }
        );

        // Setup test data helpers
        jest.spyOn(userModel, 'createUser');
        jest.spyOn(userModel, 'updateUserProfile');
        jest.spyOn(authService, 'validateHardwareToken');
    });

    afterAll(async () => {
        await redisClient.quit();
        hardwareMock.restore();
        jest.clearAllMocks();
    });

    const generateTestUser = (overrides: Partial<CreateUserDTO> = {}): CreateUserDTO => ({
        email: faker.internet.email(),
        username: faker.internet.userName(),
        password: 'Test123!@#',
        initialPreferences: {
            scanResolution: 0.01,
            scanRange: 5,
            autoJoinFleet: true,
            defaultGameMode: 'standard',
            privacySettings: {
                profileVisibility: 'public',
                locationSharing: true,
                fleetDiscoverable: true
            }
        },
        ...overrides
    });

    describe('User Registration Tests', () => {
        it('should register user with hardware binding', async () => {
            const testUser = generateTestUser();
            const deviceToken = hardwareMock.generateToken();

            const response = await supertest(userController)
                .post('/register')
                .send({
                    ...testUser,
                    deviceToken,
                    deviceId: faker.string.uuid(),
                    hardwareId: faker.string.alphanumeric(16),
                    firmwareVersion: '1.0.0'
                });

            expect(response.status).toBe(201);
            expect(response.body.user).toBeDefined();
            expect(response.body.auth).toBeDefined();
            expect(response.body.deviceContext).toBeDefined();
            expect(authService.validateHardwareToken).toHaveBeenCalled();
        });

        it('should enforce password security requirements', async () => {
            const testUser = generateTestUser({ password: 'weak' });

            const response = await supertest(userController)
                .post('/register')
                .send(testUser);

            expect(response.status).toBe(400);
            expect(response.body.error).toBe('Validation failed');
        });

        it('should validate LiDAR preferences', async () => {
            const testUser = generateTestUser({
                initialPreferences: {
                    scanResolution: 2.0, // Invalid: > 1.0
                    scanRange: 10 // Invalid: > 5
                }
            });

            const response = await supertest(userController)
                .post('/register')
                .send(testUser);

            expect(response.status).toBe(400);
            expect(response.body.error).toBe('Validation failed');
        });
    });

    describe('Hardware Authentication Tests', () => {
        let testUser: User;
        let deviceToken: string;

        beforeEach(async () => {
            // Create test user with hardware binding
            testUser = await userModel.createUser(generateTestUser());
            deviceToken = hardwareMock.generateToken();
            await authService.bindDevice(testUser.id, {
                deviceId: faker.string.uuid(),
                hardwareId: faker.string.alphanumeric(16),
                deviceToken
            });
        });

        it('should authenticate with valid hardware token', async () => {
            const response = await supertest(userController)
                .post('/login')
                .send({
                    email: testUser.email,
                    password: 'Test123!@#',
                    deviceToken
                });

            expect(response.status).toBe(200);
            expect(response.body.auth).toBeDefined();
            expect(response.body.deviceBinding).toBeDefined();
        });

        it('should reject invalid hardware tokens', async () => {
            const response = await supertest(userController)
                .post('/login')
                .send({
                    email: testUser.email,
                    password: 'Test123!@#',
                    deviceToken: 'invalid-token'
                });

            expect(response.status).toBe(401);
            expect(response.body.error).toBe('Invalid hardware token');
        });

        it('should handle device binding changes', async () => {
            const newDeviceToken = hardwareMock.generateToken();

            const response = await supertest(userController)
                .post('/devices/bind')
                .set('Authorization', `Bearer ${testUser.id}`)
                .send({
                    deviceId: faker.string.uuid(),
                    hardwareId: faker.string.alphanumeric(16),
                    deviceToken: newDeviceToken
                });

            expect(response.status).toBe(200);
            expect(response.body.deviceBinding).toBeDefined();
        });
    });

    describe('Fleet-Aware Profile Tests', () => {
        let testUser: User;
        let fleetId: string;

        beforeEach(async () => {
            testUser = await userModel.createUser(generateTestUser());
            fleetId = faker.string.uuid();
            await userModel.updateFleetStatus(testUser.id, fleetId, 'joined');
        });

        it('should update profile with fleet context', async () => {
            const response = await supertest(userController)
                .patch('/profile')
                .set('Authorization', `Bearer ${testUser.id}`)
                .send({
                    displayName: faker.internet.userName(),
                    preferences: {
                        autoJoinFleet: true,
                        scanRange: 4
                    }
                });

            expect(response.status).toBe(200);
            expect(response.body.fleetHistory).toContain(fleetId);
        });

        it('should maintain fleet critical settings', async () => {
            const response = await supertest(userController)
                .patch('/profile')
                .set('Authorization', `Bearer ${testUser.id}`)
                .send({
                    preferences: {
                        scanResolution: 0.5, // Should not increase from 0.01
                        scanRange: 3 // Should maintain minimum for fleet
                    }
                });

            expect(response.status).toBe(200);
            expect(response.body.preferences.scanResolution).toBe(0.01);
            expect(response.body.preferences.scanRange).toBeGreaterThanOrEqual(3);
        });
    });

    describe('Proximity Features Tests', () => {
        let testUser: User;
        let nearbyUsers: User[];

        beforeEach(async () => {
            testUser = await userModel.createUser(generateTestUser());
            
            // Create nearby users
            nearbyUsers = await Promise.all([
                userModel.createUser(generateTestUser()),
                userModel.createUser(generateTestUser()),
                userModel.createUser(generateTestUser())
            ]);

            // Set user locations
            await Promise.all([
                userController.updateLocation(testUser.id, {
                    latitude: 0,
                    longitude: 0,
                    accuracy: 1,
                    timestamp: Date.now()
                }),
                ...nearbyUsers.map((user, i) => 
                    userController.updateLocation(user.id, {
                        latitude: 0.0001 * (i + 1),
                        longitude: 0.0001 * (i + 1),
                        accuracy: 1,
                        timestamp: Date.now()
                    })
                )
            ]);
        });

        it('should find users within 5m range', async () => {
            const response = await supertest(userController)
                .get('/nearby')
                .set('Authorization', `Bearer ${testUser.id}`)
                .query({ range: 5 });

            expect(response.status).toBe(200);
            expect(response.body.length).toBeGreaterThan(0);
            expect(response.body[0].distance).toBeLessThanOrEqual(5);
        });

        it('should include fleet status in nearby users', async () => {
            const fleetId = faker.string.uuid();
            await userModel.updateFleetStatus(nearbyUsers[0].id, fleetId, 'joined');

            const response = await supertest(userController)
                .get('/nearby')
                .set('Authorization', `Bearer ${testUser.id}`)
                .query({ range: 5 });

            expect(response.status).toBe(200);
            expect(response.body[0].fleetStatus).toBeDefined();
        });

        it('should handle location updates with rate limiting', async () => {
            const updates = Array(10).fill(null).map(() => ({
                latitude: faker.location.latitude(),
                longitude: faker.location.longitude(),
                accuracy: 1,
                timestamp: Date.now()
            }));

            const responses = await Promise.all(
                updates.map(update =>
                    supertest(userController)
                        .post('/location')
                        .set('Authorization', `Bearer ${testUser.id}`)
                        .send(update)
                )
            );

            expect(responses.some(r => r.status === 429)).toBe(true);
        });
    });
});