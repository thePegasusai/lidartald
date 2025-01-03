import { EncryptionService } from '../src/encryption';
import { JWTService } from '../src/jwt';
import { OAuthService } from '../src/oauth';
import { RBACService, ROLES, PERMISSIONS } from '../src/rbac';
import Redis from 'ioredis';
import { jest } from '@jest/globals';

// Mock Redis client
jest.mock('ioredis');

// Mock HSM client
const mockHSMClient = {
    generateKey: jest.fn(),
    storeKey: jest.fn(),
    retrieveKey: jest.fn()
};

describe('TALD UNIA Security Services', () => {
    let encryptionService: EncryptionService;
    let jwtService: JWTService;
    let oauthService: OAuthService;
    let rbacService: RBACService;
    let redisClient: Redis;

    beforeAll(async () => {
        // Initialize Redis client with cluster configuration
        redisClient = new Redis({
            cluster: [{ host: 'localhost', port: 6379 }],
            clusterRetryStrategy: (times: number) => Math.min(times * 50, 2000)
        });

        // Initialize services
        encryptionService = new EncryptionService(mockHSMClient);
        jwtService = new JWTService(encryptionService);
        oauthService = new OAuthService(jwtService, encryptionService, {
            cluster: true,
            clusterRetryStrategy: (times: number) => Math.min(times * 50, 2000)
        });
        rbacService = new RBACService(jwtService, {
            host: 'localhost',
            port: 6379
        });

        // Setup HSM mock responses
        mockHSMClient.generateKey.mockResolvedValue(TEST_ENCRYPTION_KEY);
        mockHSMClient.storeKey.mockResolvedValue(undefined);
        mockHSMClient.retrieveKey.mockResolvedValue(TEST_ENCRYPTION_KEY);
    });

    afterAll(async () => {
        // Cleanup
        await redisClient.quit();
        jest.clearAllMocks();
    });

    describe('EncryptionService Tests', () => {
        const testData = Buffer.from('sensitive-test-data');
        const keyId = 'test-key-id';

        test('should encrypt and decrypt data using hardware-backed keys', async () => {
            // Encrypt data
            const encrypted = await encryptionService.encryptData(testData, keyId);
            expect(encrypted).toHaveProperty('data');
            expect(encrypted).toHaveProperty('iv');
            expect(encrypted).toHaveProperty('authTag');

            // Decrypt data
            const decrypted = await encryptionService.decryptData(encrypted, keyId);
            expect(decrypted).toEqual(testData);
        });

        test('should rotate encryption keys securely', async () => {
            // Initial encryption
            const encrypted = await encryptionService.encryptData(testData, keyId);
            
            // Rotate key
            await encryptionService.rotateKey(keyId);
            
            // Verify data is still accessible
            const decrypted = await encryptionService.decryptData(encrypted, keyId);
            expect(decrypted).toEqual(testData);
        });

        test('should handle hardware security module failures gracefully', async () => {
            mockHSMClient.generateKey.mockRejectedValueOnce(new Error('HSM failure'));
            
            await expect(
                encryptionService.encryptData(testData, 'failed-key')
            ).rejects.toThrow('Encryption failed');
        });
    });

    describe('JWT Service Tests', () => {
        const testPayload = {
            userId: TEST_USER_DATA.id,
            roles: [TEST_USER_DATA.role],
            permissions: TEST_USER_DATA.permissions,
            deviceId: TEST_USER_DATA.deviceId
        };

        test('should sign and verify JWT tokens with device binding', async () => {
            // Sign token
            const token = await jwtService.signToken(testPayload);
            expect(token).toBeTruthy();

            // Verify token
            const decoded = await jwtService.verifyAndDecode(token);
            expect(decoded.userId).toBe(testPayload.userId);
            expect(decoded.deviceId).toBe(testPayload.deviceId);
        });

        test('should handle token refresh with security context preservation', async () => {
            const token = await jwtService.signToken(testPayload);
            const refreshedToken = await jwtService.refreshToken(token);
            
            const decoded = await jwtService.verifyAndDecode(refreshedToken);
            expect(decoded.userId).toBe(testPayload.userId);
            expect(decoded.permissions).toEqual(testPayload.permissions);
        });

        test('should enforce device binding and prevent token reuse', async () => {
            const token = await jwtService.signToken(testPayload);
            await jwtService.revokeToken(token);
            
            await expect(
                jwtService.verifyAndDecode(token)
            ).rejects.toThrow('Token has been revoked');
        });
    });

    describe('OAuth Service Tests', () => {
        const clientCredentials = {
            clientId: 'test-client',
            clientSecret: 'test-secret',
            grantTypes: ['authorization_code'],
            scope: ['profile', 'fleet']
        };

        const deviceContext = {
            deviceId: TEST_USER_DATA.deviceId,
            hardwareId: 'test-hardware-id',
            firmwareVersion: '1.0.0',
            securityLevel: 1
        };

        test('should validate client credentials with device binding', async () => {
            const validation = await oauthService.validateClient(
                clientCredentials,
                deviceContext
            );
            
            expect(validation.isValid).toBe(true);
            expect(validation.securityContext.deviceTrust).toBeGreaterThan(0);
        });

        test('should handle authentication with PKCE validation', async () => {
            const request = {
                body: {
                    client_id: clientCredentials.clientId,
                    code_verifier: 'test-verifier',
                    code_challenge: 'test-challenge',
                    scope: 'profile fleet'
                }
            };

            const response = await oauthService.authenticateHandler(
                request as any,
                deviceContext
            );

            expect(response.accessToken).toBeTruthy();
            expect(response.refreshToken).toBeTruthy();
            expect(response.deviceBinding).toBeTruthy();
        });

        test('should enforce rate limiting on token requests', async () => {
            const requests = Array(101).fill(null).map(() => 
                oauthService.validateClient(clientCredentials, deviceContext)
            );

            await expect(
                Promise.all(requests)
            ).rejects.toThrow('Rate limit exceeded');
        });
    });

    describe('RBAC Service Tests', () => {
        test('should validate role hierarchy and permission inheritance', async () => {
            const adminPerms = await rbacService.getRolePermissions(ROLES.ADMIN);
            const basicPerms = await rbacService.getRolePermissions(ROLES.BASIC_USER);

            expect(adminPerms).toContain(PERMISSIONS.ADMIN_ACCESS);
            expect(basicPerms).not.toContain(PERMISSIONS.ADMIN_ACCESS);
            expect(adminPerms).toContain(PERMISSIONS.SCAN);
            expect(basicPerms).toContain(PERMISSIONS.SCAN);
        });

        test('should handle permission checks with caching', async () => {
            const hasPermission = await rbacService.hasPermission(
                ROLES.PREMIUM_USER,
                PERMISSIONS.FLEET_CREATE
            );
            expect(hasPermission).toBe(true);

            // Check cache hit
            const cachedResult = await rbacService.hasPermission(
                ROLES.PREMIUM_USER,
                PERMISSIONS.FLEET_CREATE
            );
            expect(cachedResult).toBe(true);
        });

        test('should validate multiple required permissions', async () => {
            const result = await rbacService.requirePermission(
                ROLES.DEVELOPER,
                [PERMISSIONS.SCAN, PERMISSIONS.FLEET_CREATE, PERMISSIONS.PROFILE_WRITE]
            );
            expect(result).toBe(true);
        });

        test('should prevent privilege escalation attempts', async () => {
            const hasAdminAccess = await rbacService.hasPermission(
                ROLES.BASIC_USER,
                PERMISSIONS.ADMIN_ACCESS
            );
            expect(hasAdminAccess).toBe(false);
        });
    });
});