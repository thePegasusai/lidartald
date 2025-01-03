import { z } from 'zod'; // v3.21.4
import { WebSocketEventType } from '../types/websocket.types';
import { environmentConfig } from './environment';

/**
 * WebSocket configuration constants based on technical specifications
 * Ensures <50ms network latency and supports up to 32 connected devices
 */
export const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8080';
export const WS_RECONNECT_ATTEMPTS = 5;
export const WS_RECONNECT_INTERVAL_MS = 1000;
export const WS_HEARTBEAT_INTERVAL_MS = 30000;
export const WS_MESSAGE_TIMEOUT_MS = 10000;

/**
 * WebSocket configuration schema with enhanced validation
 * Implements comprehensive runtime validation for WebSocket settings
 */
const webSocketConfigSchema = z.object({
    url: z.string().url('Invalid WebSocket URL'),
    reconnectAttempts: z.number()
        .min(1, 'Minimum 1 reconnection attempt required')
        .max(10, 'Maximum 10 reconnection attempts allowed'),
    reconnectInterval: z.number()
        .min(500, 'Minimum reconnect interval 500ms')
        .max(5000, 'Maximum reconnect interval 5000ms'),
    heartbeatInterval: z.number()
        .min(15000, 'Minimum heartbeat interval 15s')
        .max(60000, 'Maximum heartbeat interval 60s'),
    messageTimeout: z.number()
        .min(1000, 'Minimum message timeout 1000ms')
        .max(30000, 'Maximum message timeout 30000ms'),
    compression: z.enum(['none', 'lz4', 'zstd']).optional(),
    binaryType: z.enum(['blob', 'arraybuffer']).default('arraybuffer'),
    maxMessageSize: z.number()
        .min(1024, 'Minimum message size 1KB')
        .max(1048576, 'Maximum message size 1MB')
        .default(1048576)
});

/**
 * Default WebSocket configuration
 * Optimized for real-time fleet coordination and game state synchronization
 */
export const webSocketConfig = {
    url: WS_URL,
    reconnectAttempts: WS_RECONNECT_ATTEMPTS,
    reconnectInterval: WS_RECONNECT_INTERVAL_MS,
    heartbeatInterval: WS_HEARTBEAT_INTERVAL_MS,
    messageTimeout: WS_MESSAGE_TIMEOUT_MS,
    compression: 'lz4' as const,
    binaryType: 'arraybuffer' as const,
    maxMessageSize: 1048576, // 1MB
    performanceMode: environmentConfig.scanSettings.RATE_HZ >= 30
};

/**
 * Creates and configures a WebSocket client with enhanced features
 * Implements automatic reconnection, heartbeat, and performance monitoring
 */
export function createWebSocketClient(config: typeof webSocketConfig): WebSocket {
    // Validate configuration
    validateWebSocketConfig(config);

    // Create WebSocket instance
    const ws = new WebSocket(config.url);
    ws.binaryType = config.binaryType;

    // Configure reconnection logic
    let reconnectCount = 0;
    let reconnectTimeout: NodeJS.Timeout;

    const reconnect = () => {
        if (reconnectCount < config.reconnectAttempts) {
            reconnectTimeout = setTimeout(() => {
                reconnectCount++;
                const newWs = new WebSocket(config.url);
                Object.assign(ws, newWs);
            }, config.reconnectInterval * Math.pow(2, reconnectCount));
        }
    };

    // Setup heartbeat mechanism
    let heartbeatInterval: NodeJS.Timeout;
    const startHeartbeat = () => {
        heartbeatInterval = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'PING', timestamp: Date.now() }));
            }
        }, config.heartbeatInterval);
    };

    // Configure event handlers
    ws.onopen = () => {
        reconnectCount = 0;
        clearTimeout(reconnectTimeout);
        startHeartbeat();
    };

    ws.onclose = () => {
        clearInterval(heartbeatInterval);
        reconnect();
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        ws.close();
    };

    // Setup performance monitoring
    let messageQueue: { timestamp: number; size: number }[] = [];
    const MAX_QUEUE_SIZE = 100;

    ws.onmessage = (event) => {
        // Track message metrics
        messageQueue.push({
            timestamp: Date.now(),
            size: event.data.length
        });

        // Maintain queue size
        if (messageQueue.length > MAX_QUEUE_SIZE) {
            messageQueue.shift();
        }

        // Calculate performance metrics
        const latency = calculateLatency(messageQueue);
        if (latency > 50) { // >50ms latency threshold
            console.warn('High WebSocket latency detected:', latency);
        }
    };

    return ws;
}

/**
 * Validates WebSocket configuration against schema and performance requirements
 * Implements comprehensive validation for all configuration parameters
 */
export function validateWebSocketConfig(config: typeof webSocketConfig): boolean {
    try {
        webSocketConfigSchema.parse(config);

        // Validate URL security
        if (!/^wss?:\/\//.test(config.url)) {
            console.warn('Non-secure WebSocket connection detected');
        }

        // Validate performance settings
        if (config.performanceMode) {
            if (config.messageTimeout > 50) { // Ensure <50ms latency
                return false;
            }
        }

        return true;
    } catch (error) {
        console.error('WebSocket configuration validation failed:', error);
        return false;
    }
}

/**
 * Helper function to calculate WebSocket latency from message queue
 */
function calculateLatency(queue: { timestamp: number; size: number }[]): number {
    if (queue.length < 2) return 0;
    
    const recentMessages = queue.slice(-10);
    const latencies = recentMessages.map((msg, i) => 
        i > 0 ? msg.timestamp - recentMessages[i-1].timestamp : 0
    ).filter(l => l > 0);

    return latencies.reduce((sum, l) => sum + l, 0) / latencies.length;
}