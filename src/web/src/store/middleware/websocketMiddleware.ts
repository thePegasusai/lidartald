import { Middleware } from '@reduxjs/toolkit';
import { filter, map, bufferTime, retryWhen, delay } from 'rxjs/operators';
import { compress, decompress } from 'pako'; // v2.1.0
import { WebSocketClient } from '../../api/websocketApi';
import { webSocketConfig } from '../../config/websocket';
import { WebSocketEventType } from '../../types/websocket.types';

// Action Types
export const WS_CONNECT = '@websocket/connect';
export const WS_DISCONNECT = '@websocket/disconnect';
export const WS_SEND_MESSAGE = '@websocket/send';
export const WS_HEALTH_CHECK = '@websocket/health';
export const WS_RECONNECT = '@websocket/reconnect';

// Performance monitoring thresholds
const LATENCY_THRESHOLD_MS = 50;
const COMPRESSION_THRESHOLD_BYTES = 1024;
const MESSAGE_BATCH_INTERVAL_MS = 16; // ~60Hz batching
const PERFORMANCE_SAMPLE_SIZE = 100;

// Message queue for performance tracking
interface MessageMetrics {
    timestamp: number;
    size: number;
    latency: number;
}

/**
 * Creates Redux middleware for handling WebSocket connections with enhanced features
 * Implements compression, automatic reconnection, and performance monitoring
 */
export const createWebSocketMiddleware = (
    wsClient: WebSocketClient,
    config = webSocketConfig
): Middleware => {
    // Performance monitoring queue
    let messageMetrics: MessageMetrics[] = [];
    let connectionAttempts = 0;

    return store => next => action => {
        switch (action.type) {
            case WS_CONNECT:
                try {
                    wsClient.connect();
                    setupMessageHandling(wsClient, store.dispatch);
                    setupPerformanceMonitoring(wsClient);
                    connectionAttempts = 0;
                } catch (error) {
                    console.error('WebSocket connection failed:', error);
                    handleReconnection(wsClient, connectionAttempts);
                }
                break;

            case WS_DISCONNECT:
                wsClient.disconnect();
                messageMetrics = [];
                break;

            case WS_SEND_MESSAGE:
                if (action.payload) {
                    sendMessage(wsClient, action.payload);
                }
                break;

            case WS_HEALTH_CHECK:
                performHealthCheck(wsClient, messageMetrics);
                break;

            case WS_RECONNECT:
                handleReconnection(wsClient, connectionAttempts);
                break;
        }

        return next(action);
    };
};

/**
 * Sets up message handling with compression and batching
 * Implements <50ms latency requirement for real-time updates
 */
function setupMessageHandling(wsClient: WebSocketClient, dispatch: any): void {
    wsClient.subscribeToEvents<any>(WebSocketEventType.FLEET_UPDATE)
        .pipe(
            bufferTime(MESSAGE_BATCH_INTERVAL_MS),
            filter(messages => messages.length > 0),
            map(messages => processMessageBatch(messages))
        )
        .subscribe(
            batch => handleMessageBatch(batch, dispatch),
            error => handleMessageError(error, dispatch)
        );

    // Enable compression for optimal network performance
    wsClient.enableCompression();
    wsClient.setHeartbeat(config.heartbeatInterval);
}

/**
 * Processes message batch with compression optimization
 * Implements size-based compression decision
 */
function processMessageBatch(messages: any[]): any[] {
    return messages.map(message => {
        const messageSize = JSON.stringify(message).length;
        if (messageSize > COMPRESSION_THRESHOLD_BYTES) {
            return {
                ...message,
                payload: compress(JSON.stringify(message.payload)),
                compressed: true
            };
        }
        return message;
    });
}

/**
 * Handles batched messages with performance tracking
 * Maintains message metrics for latency monitoring
 */
function handleMessageBatch(batch: any[], dispatch: any): void {
    const startTime = performance.now();

    batch.forEach(message => {
        if (message.compressed) {
            message.payload = JSON.parse(decompress(message.payload));
        }

        // Track message metrics
        updateMessageMetrics({
            timestamp: Date.now(),
            size: JSON.stringify(message).length,
            latency: performance.now() - startTime
        });

        dispatch({
            type: `${message.type}_SUCCESS`,
            payload: message.payload
        });
    });
}

/**
 * Updates performance metrics for monitoring
 * Maintains rolling window of recent message statistics
 */
function updateMessageMetrics(metrics: MessageMetrics): void {
    messageMetrics.push(metrics);
    if (messageMetrics.length > PERFORMANCE_SAMPLE_SIZE) {
        messageMetrics.shift();
    }

    // Check for performance degradation
    const averageLatency = calculateAverageLatency(messageMetrics);
    if (averageLatency > LATENCY_THRESHOLD_MS) {
        console.warn(`High latency detected: ${averageLatency.toFixed(2)}ms`);
    }
}

/**
 * Sends message with automatic compression
 * Optimizes message size based on content
 */
function sendMessage(wsClient: WebSocketClient, payload: any): void {
    const message = {
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        payload
    };

    const messageSize = JSON.stringify(message).length;
    if (messageSize > COMPRESSION_THRESHOLD_BYTES) {
        message.payload = compress(JSON.stringify(payload));
        wsClient.sendMessage({ ...message, compressed: true });
    } else {
        wsClient.sendMessage(message);
    }
}

/**
 * Performs health check on WebSocket connection
 * Monitors latency and connection stability
 */
function performHealthCheck(
    wsClient: WebSocketClient,
    metrics: MessageMetrics[]
): void {
    const averageLatency = calculateAverageLatency(metrics);
    const healthStatus = {
        connected: wsClient !== null,
        latency: averageLatency,
        messageCount: metrics.length,
        timestamp: Date.now()
    };

    if (averageLatency > LATENCY_THRESHOLD_MS) {
        console.warn('WebSocket health check: High latency detected');
    }

    return healthStatus;
}

/**
 * Handles WebSocket reconnection with exponential backoff
 * Implements automatic recovery from connection failures
 */
function handleReconnection(wsClient: WebSocketClient, attempts: number): void {
    if (attempts < config.reconnectAttempts) {
        const backoffDelay = Math.min(
            1000 * Math.pow(2, attempts),
            30000
        );

        setTimeout(() => {
            connectionAttempts++;
            wsClient.connect();
        }, backoffDelay);
    } else {
        console.error('Max reconnection attempts reached');
    }
}

/**
 * Calculates average latency from message metrics
 * Used for performance monitoring and health checks
 */
function calculateAverageLatency(metrics: MessageMetrics[]): number {
    if (metrics.length === 0) return 0;
    
    const latencies = metrics.map(m => m.latency);
    return latencies.reduce((sum, latency) => sum + latency, 0) / latencies.length;
}

/**
 * Handles message processing errors
 * Implements error recovery and logging
 */
function handleMessageError(error: any, dispatch: any): void {
    console.error('WebSocket message error:', error);
    dispatch({
        type: 'WS_ERROR',
        payload: {
            error: error.message,
            timestamp: Date.now()
        }
    });
}

export default createWebSocketMiddleware;