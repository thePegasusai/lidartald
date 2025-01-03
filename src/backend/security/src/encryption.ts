import { injectable, singleton } from 'tsyringe';
import { promisify } from 'util';
import * as crypto from 'crypto'; // v16.0.0+

// Constants for encryption configuration
const ENCRYPTION_ALGORITHM = 'aes-256-gcm';
const IV_LENGTH = 16;
const AUTH_TAG_LENGTH = 16;
const KEY_LENGTH = 32;
const KEY_ITERATION_COUNT = 100000;
const SALT_LENGTH = 32;

// Type definitions
interface EncryptedData {
    data: Buffer;
    iv: Buffer;
    authTag: Buffer;
    salt?: Buffer;
}

interface HSMClient {
    generateKey(length: number): Promise<Buffer>;
    storeKey(identifier: string, key: Buffer): Promise<void>;
    retrieveKey(identifier: string): Promise<Buffer>;
}

// Decorators
function validateKeyGeneration(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    descriptor.value = async function(...args: any[]) {
        const key = await originalMethod.apply(this, args);
        if (!Buffer.isBuffer(key) || key.length !== KEY_LENGTH) {
            throw new Error('Invalid key generated');
        }
        return key;
    };
}

function validateEncryption(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    descriptor.value = async function(...args: any[]) {
        if (!args[0]) throw new Error('Data to encrypt is required');
        if (!args[1]) throw new Error('Key identifier is required');
        return originalMethod.apply(this, args);
    };
}

function validateDecryption(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    descriptor.value = async function(...args: any[]) {
        const [encryptedData] = args;
        if (!encryptedData?.data || !encryptedData?.iv || !encryptedData?.authTag) {
            throw new Error('Invalid encrypted data structure');
        }
        return originalMethod.apply(this, args);
    };
}

function measurePerformance(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    descriptor.value = async function(...args: any[]) {
        const start = process.hrtime();
        const result = await originalMethod.apply(this, args);
        const [seconds, nanoseconds] = process.hrtime(start);
        console.debug(`${propertyKey} execution time: ${seconds}s ${nanoseconds}ns`);
        return result;
    };
}

// Utility functions
export async function generateKey(useHardwareBackedKey: boolean = false): Promise<Buffer> {
    if (useHardwareBackedKey) {
        throw new Error('Hardware-backed key generation requires HSM client');
    }
    return crypto.randomBytes(KEY_LENGTH);
}

export function generateIV(): Buffer {
    return crypto.randomBytes(IV_LENGTH);
}

export async function deriveKey(password: string, salt?: Buffer): Promise<Buffer> {
    const useSalt = salt || crypto.randomBytes(SALT_LENGTH);
    const pbkdf2 = promisify(crypto.pbkdf2);
    return pbkdf2(password, useSalt, KEY_ITERATION_COUNT, KEY_LENGTH, 'sha512');
}

@injectable()
@singleton()
export class EncryptionService {
    private masterKey: Buffer;
    private keyCache: Map<string, Buffer>;
    private readonly hsmClient: HSMClient;

    constructor(hsmClient: HSMClient) {
        this.hsmClient = hsmClient;
        this.keyCache = new Map();
        this.initializeService();
    }

    private async initializeService(): Promise<void> {
        try {
            this.masterKey = await this.hsmClient.generateKey(KEY_LENGTH);
            this.setupKeyRotation();
            this.setupSecureMemory();
        } catch (error) {
            throw new Error('Failed to initialize encryption service');
        }
    }

    private setupKeyRotation(): void {
        // Setup automated key rotation every 30 days
        setInterval(() => {
            this.keyCache.forEach(async (_, keyId) => {
                await this.rotateKey(keyId);
            });
        }, 30 * 24 * 60 * 60 * 1000);
    }

    private setupSecureMemory(): void {
        // Ensure sensitive data is cleared from memory when process exits
        process.on('exit', () => {
            this.masterKey.fill(0);
            this.keyCache.forEach(key => key.fill(0));
            this.keyCache.clear();
        });
    }

    @validateEncryption
    @measurePerformance
    public async encryptData(data: Buffer | string, keyIdentifier: string): Promise<EncryptedData> {
        try {
            const key = await this.getOrGenerateKey(keyIdentifier);
            const iv = generateIV();
            const cipher = crypto.createCipheriv(ENCRYPTION_ALGORITHM, key, iv);
            
            const inputData = Buffer.isBuffer(data) ? data : Buffer.from(data);
            const encryptedData = Buffer.concat([
                cipher.update(inputData),
                cipher.final()
            ]);

            const authTag = cipher.getAuthTag();

            return {
                data: encryptedData,
                iv,
                authTag
            };
        } catch (error) {
            throw new Error(`Encryption failed: ${error.message}`);
        }
    }

    @validateDecryption
    @measurePerformance
    public async decryptData(encryptedData: EncryptedData, keyIdentifier: string): Promise<Buffer> {
        try {
            const key = await this.getKey(keyIdentifier);
            const decipher = crypto.createDecipheriv(ENCRYPTION_ALGORITHM, key, encryptedData.iv);
            decipher.setAuthTag(encryptedData.authTag);

            return Buffer.concat([
                decipher.update(encryptedData.data),
                decipher.final()
            ]);
        } catch (error) {
            throw new Error(`Decryption failed: ${error.message}`);
        }
    }

    @measurePerformance
    public async rotateKey(keyIdentifier: string): Promise<boolean> {
        try {
            const newKey = await this.hsmClient.generateKey(KEY_LENGTH);
            const oldKey = await this.getKey(keyIdentifier);

            // Store new key in HSM
            await this.hsmClient.storeKey(keyIdentifier, newKey);

            // Update cache
            this.keyCache.set(keyIdentifier, newKey);

            // Securely clear old key
            oldKey.fill(0);

            return true;
        } catch (error) {
            throw new Error(`Key rotation failed: ${error.message}`);
        }
    }

    private async getOrGenerateKey(keyIdentifier: string): Promise<Buffer> {
        if (this.keyCache.has(keyIdentifier)) {
            return this.keyCache.get(keyIdentifier)!;
        }

        const key = await this.hsmClient.generateKey(KEY_LENGTH);
        this.keyCache.set(keyIdentifier, key);
        return key;
    }

    private async getKey(keyIdentifier: string): Promise<Buffer> {
        const key = this.keyCache.get(keyIdentifier) || 
                   await this.hsmClient.retrieveKey(keyIdentifier);
        
        if (!key) {
            throw new Error(`Key not found for identifier: ${keyIdentifier}`);
        }

        return key;
    }
}