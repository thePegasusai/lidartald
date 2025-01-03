import { Subject, interval, fromEvent } from 'rxjs'; // v7.8
import { filter, map, takeUntil } from 'rxjs/operators'; // v7.8
import { z } from 'zod'; // v3.x
import { WebSocketEventType } from '../types/websocket.types';
import { webSocketConfig } from '../config/websocket';

// Constants for WebSocket management
const WEBSOCKET_CLOSE_NORMAL = 1000;
const WEBSOCKET_CLOSE_ABNORMAL = 1006;
const MAX_MESSAGE_SIZE = 16384; // 16KB
const MAX_RECONNECT_ATTEMPTS = 5;
const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 32000;
const COMPRESSION_THRESHOLD = 1024; // 1KB
const CONNECTION_POOL_SIZE = 32;

/**
 * Message validation schema using Zod
 * Ensures type safety and data integrity for WebSocket messages
 */
const messageSchema = z.object({
    type: z.nativeEnum(WebSocketEventType),
    payload: z.unknown(),
    timestamp: z.number(),
    deviceId: z.string().uuid(),
    messageId: z.string().uuid(),
    error: z.object({
        code: z.number(),
        message: z.string(),
        timestamp: z.number(),
        recoverable: z.boolean()
    }).nullable()
});

/**
 * Enhanced WebSocket client implementation for TALD UNIA platform
 * Handles fleet coordination, game state sync, and social interactions
 */
export class WebSocketClient {
    private ws: WebSocket | null = null;
    private messageSubject = new Subject<any>();
    private closeSubject = new Subject<void>();
    private reconnectAttempts = 0;
    private heartbeatInterval: number;
    private messageQueue: Array<{ data: any; timestamp: number }> = [];
    private metrics = {
        latency: 0,
        messageCount: 0,
        errorCount: 0,
        lastHeartbeat: 0
    };

    constructor() {
        this.heartbeatInterval = webSocketConfig.heartbeatInterval;
        this.setupMetricsCollection();
    }

    /**
     * Establishes WebSocket connection with automatic reconnection
     * Implements exponential backoff strategy for reliability
     */
    public async connect(): Promise<void> {
        try {
            this.ws = new WebSocket(webSocketConfig.url);
            this.ws.binaryType = 'arraybuffer';

            this.setupEventHandlers();
            this.startHeartbeat();
            this.initializeMetrics();

            await this.waitForConnection();
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            await this.handleReconnection();
        }
    }

    /**
     * Gracefully closes WebSocket connection and cleans up resources
     */
    public disconnect(): void {
        if (this.ws) {
            clearInterval(this.heartbeatInterval);
            this.ws.close(WEBSOCKET_CLOSE_NORMAL);
            this.closeSubject.next();
            this.closeSubject.complete();
            this.messageSubject.complete();
            this.ws = null;
            this.reconnectAttempts = 0;
        }
    }

    /**
     * Sends message through WebSocket with compression and validation
     * Implements message queuing for reliability
     */
    public async sendMessage(message: any): Promise<void> {
        try {
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                throw new Error('WebSocket not connected');
            }

            const validatedMessage = await this.validateMessage(message);
            const compressedMessage = this.compressMessageIfNeeded(validatedMessage);

            if (this.shouldQueueMessage()) {
                this.queueMessage(compressedMessage);
                return;
            }

            await this.sendWithRetry(compressedMessage);
            this.updateMetrics('send');
        } catch (error) {
            console.error('Failed to send message:', error);
            this.updateMetrics('error');
            throw error;
        }
    }

    /**
     * Subscribes to WebSocket messages with optional filtering
     * Returns RxJS Observable for message stream
     */
    public onMessage<T>(eventType?: WebSocketEventType) {
        let stream = this.messageSubject.asObservable();
        
        if (eventType) {
            stream = stream.pipe(
                filter(message => message.type === eventType),
                map(message => message.payload as T)
            );
        }

        return stream.pipe(takeUntil(this.closeSubject));
    }

    private setupEventHandlers(): void {
        if (!this.ws) return;

        this.ws.onopen = () => {
            this.reconnectAttempts = 0;
            this.processQueuedMessages();
            this.updateMetrics('connection');
        };

        this.ws.onmessage = (event) => {
            this.handleMessage(event.data);
        };

        this.ws.onclose = (event) => {
            this.handleClose(event);
        };

        this.ws.onerror = (error) => {
            this.handleError(error);
        };
    }

    private async handleMessage(data: any): Promise<void> {
        try {
            const message = typeof data === 'string' ? JSON.parse(data) : data;
            const validatedMessage = await this.validateMessage(message);
            this.messageSubject.next(validatedMessage);
            this.updateMetrics('receive');
        } catch (error) {
            console.error('Message handling failed:', error);
            this.updateMetrics('error');
        }
    }

    private async validateMessage(message: unknown): Promise<any> {
        try {
            return messageSchema.parse(message);
        } catch (error) {
            throw new Error(`Message validation failed: ${error}`);
        }
    }

    private compressMessageIfNeeded(message: any): any {
        const messageSize = new TextEncoder().encode(JSON.stringify(message)).length;
        
        if (messageSize > COMPRESSION_THRESHOLD) {
            // Implement compression logic here
            return message; // Placeholder for actual compression
        }

        return message;
    }

    private async handleReconnection(): Promise<void> {
        if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
            throw new Error('Maximum reconnection attempts reached');
        }

        const backoffTime = Math.min(
            INITIAL_BACKOFF_MS * Math.pow(2, this.reconnectAttempts),
            MAX_BACKOFF_MS
        );

        this.reconnectAttempts++;
        await new Promise(resolve => setTimeout(resolve, backoffTime));
        await this.connect();
    }

    private startHeartbeat(): void {
        interval(this.heartbeatInterval)
            .pipe(takeUntil(this.closeSubject))
            .subscribe(() => {
                this.sendHeartbeat();
            });
    }

    private async sendHeartbeat(): Promise<void> {
        try {
            await this.sendMessage({
                type: WebSocketEventType.CONNECTION_STATE,
                payload: { timestamp: Date.now() },
                deviceId: 'system',
                messageId: crypto.randomUUID(),
                error: null
            });
            this.metrics.lastHeartbeat = Date.now();
        } catch (error) {
            console.error('Heartbeat failed:', error);
        }
    }

    private setupMetricsCollection(): void {
        interval(1000)
            .pipe(takeUntil(this.closeSubject))
            .subscribe(() => {
                this.calculateMetrics();
            });
    }

    private calculateMetrics(): void {
        // Calculate and update metrics
        const currentLatency = Date.now() - this.metrics.lastHeartbeat;
        this.metrics.latency = currentLatency;

        if (currentLatency > 50) { // >50ms latency threshold
            console.warn('High WebSocket latency detected:', currentLatency);
        }
    }

    private updateMetrics(type: 'send' | 'receive' | 'error' | 'connection'): void {
        switch (type) {
            case 'send':
            case 'receive':
                this.metrics.messageCount++;
                break;
            case 'error':
                this.metrics.errorCount++;
                break;
            case 'connection':
                this.metrics = {
                    latency: 0,
                    messageCount: 0,
                    errorCount: 0,
                    lastHeartbeat: Date.now()
                };
                break;
        }
    }

    private shouldQueueMessage(): boolean {
        return this.messageQueue.length > 0 || 
               (this.ws?.bufferedAmount || 0) > MAX_MESSAGE_SIZE;
    }

    private queueMessage(message: any): void {
        this.messageQueue.push({
            data: message,
            timestamp: Date.now()
        });
    }

    private async processQueuedMessages(): Promise<void> {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            if (message) {
                await this.sendWithRetry(message.data);
            }
        }
    }

    private async sendWithRetry(data: any, attempts = 0): Promise<void> {
        try {
            if (!this.ws) throw new Error('WebSocket not connected');
            
            this.ws.send(JSON.stringify(data));
        } catch (error) {
            if (attempts < 3) {
                await new Promise(resolve => setTimeout(resolve, 1000));
                return this.sendWithRetry(data, attempts + 1);
            }
            throw error;
        }
    }

    private async waitForConnection(): Promise<void> {
        if (!this.ws) throw new Error('WebSocket not initialized');

        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Connection timeout'));
            }, webSocketConfig.messageTimeout);

            this.ws!.onopen = () => {
                clearTimeout(timeout);
                resolve();
            };

            this.ws!.onerror = () => {
                clearTimeout(timeout);
                reject(new Error('Connection failed'));
            };
        });
    }

    private handleClose(event: CloseEvent): void {
        if (event.code !== WEBSOCKET_CLOSE_NORMAL) {
            this.handleReconnection().catch(console.error);
        }
        this.closeSubject.next();
    }

    private handleError(error: Event): void {
        console.error('WebSocket error:', error);
        this.updateMetrics('error');
        if (this.ws) {
            this.ws.close(WEBSOCKET_CLOSE_ABNORMAL);
        }
    }
}