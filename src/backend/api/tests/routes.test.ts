import request from 'supertest'; // v6.3.x
import { describe, test, expect, beforeAll, afterAll, beforeEach, jest } from '@jest/globals'; // v29.5.x
import express from 'express'; // v4.18.x
import WebSocket from 'ws'; // v8.13.x
import Redis from 'redis-mock'; // v0.56.x
import { userRouter } from '../src/routes/v1/user.routes';
import { fleetRouter } from '../src/routes/v1/fleet.routes';
import { ApiError } from '../src/middleware/error-handler';
import { ROLES, PERMISSIONS } from '../../security/src/rbac';

// Test configuration constants
const TEST_PORT = 3001;
const WS_PORT = 3002;
const RATE_LIMIT_WINDOW = 60000;
const TEST_TIMEOUT = 30000;

// Test data constants
const testUser = {
    email: 'test@tald.dev',
    username: 'testuser',
    password: 'TestPass123!@#',
    deviceId: 'test-device-001',
    hardwareId: 'hw-001',
    firmwareVersion: '1.0.0',
    lidarCapabilities: {
        resolution: 0.01,
        range: 5,
        scanRate: 30
    }
};

// Mock setup functions
async function setupTestApp() {
    const app = express();
    app.use(express.json());
    app.use('/api/v1/users', userRouter);
    app.use('/api/v1/fleets', fleetRouter);
    
    // Add security middleware
    app.use((req, res, next) => {
        res.setHeader('X-Content-Type-Options', 'nosniff');
        res.setHeader('X-Frame-Options', 'DENY');
        res.setHeader('X-XSS-Protection', '1; mode=block');
        next();
    });

    return app;
}

async function setupTestEnvironment() {
    const redisClient = new Redis();
    const wsServer = new WebSocket.Server({ port: WS_PORT });

    return { redisClient, wsServer };
}

// Test suites
describe('User Management Routes', () => {
    let app: express.Application;
    let redisClient: Redis.RedisClient;
    let wsServer: WebSocket.Server;

    beforeAll(async () => {
        app = await setupTestApp();
        const env = await setupTestEnvironment();
        redisClient = env.redisClient;
        wsServer = env.wsServer;
    });

    afterAll(async () => {
        await redisClient.quit();
        await wsServer.close();
    });

    beforeEach(async () => {
        await redisClient.flushall();
    });

    describe('POST /api/v1/users/register', () => {
        test('should register new user with valid data', async () => {
            const response = await request(app)
                .post('/api/v1/users/register')
                .send(testUser)
                .expect('Content-Type', /json/)
                .expect(201);

            expect(response.body).toHaveProperty('user');
            expect(response.body.user.email).toBe(testUser.email);
            expect(response.body).toHaveProperty('auth.accessToken');
            expect(response.body).toHaveProperty('deviceContext');
        });

        test('should enforce password complexity', async () => {
            const weakPassword = { ...testUser, password: 'weak' };
            const response = await request(app)
                .post('/api/v1/users/register')
                .send(weakPassword)
                .expect(400);

            expect(response.body.error).toMatch(/password/i);
        });

        test('should validate LiDAR capabilities', async () => {
            const invalidCapabilities = {
                ...testUser,
                lidarCapabilities: { resolution: 2, range: 10 }
            };
            const response = await request(app)
                .post('/api/v1/users/register')
                .send(invalidCapabilities)
                .expect(400);

            expect(response.body.error).toMatch(/capabilities/i);
        });
    });

    describe('POST /api/v1/users/login', () => {
        beforeEach(async () => {
            await request(app)
                .post('/api/v1/users/register')
                .send(testUser);
        });

        test('should authenticate valid credentials', async () => {
            const response = await request(app)
                .post('/api/v1/users/login')
                .send({
                    email: testUser.email,
                    password: testUser.password,
                    deviceId: testUser.deviceId
                })
                .expect(200);

            expect(response.body).toHaveProperty('auth.accessToken');
            expect(response.body).toHaveProperty('auth.refreshToken');
        });

        test('should enforce rate limits', async () => {
            const attempts = Array(11).fill(null);
            const responses = await Promise.all(
                attempts.map(() => 
                    request(app)
                        .post('/api/v1/users/login')
                        .send({
                            email: testUser.email,
                            password: 'wrong'
                        })
                )
            );

            const lastResponse = responses[responses.length - 1];
            expect(lastResponse.status).toBe(429);
        });
    });
});

describe('Fleet Management Routes', () => {
    let app: express.Application;
    let redisClient: Redis.RedisClient;
    let wsServer: WebSocket.Server;
    let authToken: string;

    beforeAll(async () => {
        app = await setupTestApp();
        const env = await setupTestEnvironment();
        redisClient = env.redisClient;
        wsServer = env.wsServer;

        // Setup authenticated user
        const response = await request(app)
            .post('/api/v1/users/register')
            .send(testUser);
        authToken = response.body.auth.accessToken;
    });

    afterAll(async () => {
        await redisClient.quit();
        await wsServer.close();
    });

    describe('POST /api/v1/fleets', () => {
        test('should create fleet with valid data', async () => {
            const fleetData = {
                name: 'Test Fleet',
                maxDevices: 5,
                deviceId: testUser.deviceId,
                capabilities: testUser.lidarCapabilities
            };

            const response = await request(app)
                .post('/api/v1/fleets')
                .set('Authorization', `Bearer ${authToken}`)
                .send(fleetData)
                .expect(201);

            expect(response.body.data).toHaveProperty('id');
            expect(response.body.data.name).toBe(fleetData.name);
            expect(response.body.data.devices).toHaveLength(1);
        });

        test('should validate fleet size limits', async () => {
            const oversizedFleet = {
                name: 'Large Fleet',
                maxDevices: 50,
                deviceId: testUser.deviceId,
                capabilities: testUser.lidarCapabilities
            };

            const response = await request(app)
                .post('/api/v1/fleets')
                .set('Authorization', `Bearer ${authToken}`)
                .send(oversizedFleet)
                .expect(400);

            expect(response.body.error).toMatch(/size/i);
        });
    });

    describe('PATCH /api/v1/fleets/:fleetId/sync', () => {
        let fleetId: string;

        beforeEach(async () => {
            const createResponse = await request(app)
                .post('/api/v1/fleets')
                .set('Authorization', `Bearer ${authToken}`)
                .send({
                    name: 'Sync Test Fleet',
                    maxDevices: 5,
                    deviceId: testUser.deviceId,
                    capabilities: testUser.lidarCapabilities
                });
            fleetId = createResponse.body.data.id;
        });

        test('should sync fleet state in real-time', async () => {
            const ws = new WebSocket(`ws://localhost:${WS_PORT}/fleets/${fleetId}`);
            
            const updatePromise = new Promise((resolve) => {
                ws.on('message', (data) => {
                    const message = JSON.parse(data.toString());
                    resolve(message);
                });
            });

            const syncResponse = await request(app)
                .patch(`/api/v1/fleets/${fleetId}/sync`)
                .set('Authorization', `Bearer ${authToken}`)
                .send({
                    state: { position: { x: 1, y: 2, z: 3 } },
                    version: 1
                })
                .expect(200);

            const wsMessage = await updatePromise;
            expect(wsMessage).toHaveProperty('state.position');
            expect(syncResponse.body.timestamp).toBeDefined();
        });

        test('should enforce update frequency limits', async () => {
            const updates = Array(31).fill(null);
            const responses = await Promise.all(
                updates.map(() => 
                    request(app)
                        .patch(`/api/v1/fleets/${fleetId}/sync`)
                        .set('Authorization', `Bearer ${authToken}`)
                        .send({
                            state: { timestamp: Date.now() },
                            version: 1
                        })
                )
            );

            const lastResponse = responses[responses.length - 1];
            expect(lastResponse.status).toBe(429);
        });
    });
});

describe('Security Validation', () => {
    let app: express.Application;

    beforeAll(async () => {
        app = await setupTestApp();
    });

    test('should enforce CSRF protection', async () => {
        const response = await request(app)
            .post('/api/v1/users/register')
            .send(testUser)
            .set('X-CSRF-Token', 'invalid')
            .expect(403);

        expect(response.body.error).toMatch(/csrf/i);
    });

    test('should validate security headers', async () => {
        const response = await request(app)
            .get('/api/v1/users/profile')
            .set('Authorization', 'Bearer invalid');

        expect(response.headers['x-content-type-options']).toBe('nosniff');
        expect(response.headers['x-frame-options']).toBe('DENY');
        expect(response.headers['x-xss-protection']).toBe('1; mode=block');
    });
});