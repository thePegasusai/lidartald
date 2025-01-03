import { useEffect, useCallback, useRef, useState } from 'react'; // v18.2
import { useDispatch, useSelector } from 'react-redux'; // v8.1
import { webSocketConfig } from '../../config/websocket';
import { WebSocketEventType, WebSocketMessage, isValidWebSocketMessage } from '../../types/websocket.types';
import { FleetUpdateEvent, isFleetUpdateEvent } from '../../types/fleet.types';
import LZ4 from 'lz4-js'; // v0.4.1

/**
 * Connection quality levels based on latency and packet loss metrics
 */
enum ConnectionQuality {
    EXCELLENT = 'EXCELLENT', // <20ms latency
    GOOD = 'GOOD',          // 20-50ms latency
    FAIR = 'FAIR',          // 50-100ms latency
    POOR = 'POOR'           // >100ms latency
}

/**
 * Interface for WebSocket hook configuration
 */
interface WebSocketHookConfig {
    url: string;
    reconnectAttempts: number;
    heartbeatInterval: number;
    compressionEnabled: boolean;
    latencyThreshold: number;
}

/**
 * Enhanced WebSocket hook for real-time fleet communication
 * Implements comprehensive connection management, performance monitoring,
 * and data synchronization features
 */
export function useWebSocket(config: WebSocketHookConfig = webSocketConfig) {
    const dispatch = useDispatch();
    const [connected, setConnected] = useState(false);
    const [error, setError] = useState<Error | null>(null);
    const [latency, setLatency] = useState(0);
    const [connectionQuality, setConnectionQuality] = useState<ConnectionQuality>(ConnectionQuality.GOOD);
    const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

    // WebSocket instance reference
    const ws = useRef<WebSocket | null>(null);
    
    // Message queue for handling connection drops
    const messageQueue = useRef<WebSocketMessage[]>([]);
    const MAX_QUEUE_SIZE = 1000;

    // Performance monitoring
    const latencyHistory = useRef<number[]>([]);
    const MAX_LATENCY_HISTORY = 50;

    // Reconnection state
    const reconnectCount = useRef(0);
    const reconnectTimeout = useRef<NodeJS.Timeout>();

    /**
     * Initializes WebSocket connection with compression and monitoring
     */
    const initializeWebSocket = useCallback(() => {
        if (ws.current?.readyState === WebSocket.OPEN) return;

        ws.current = new WebSocket(config.url);
        ws.current.binaryType = 'arraybuffer';

        ws.current.onopen = () => {
            setConnected(true);
            setError(null);
            reconnectCount.current = 0;
            flushMessageQueue();
            startHeartbeat();
        };

        ws.current.onclose = () => {
            setConnected(false);
            handleReconnect();
        };

        ws.current.onerror = (event) => {
            setError(new Error('WebSocket error occurred'));
            console.error('WebSocket error:', event);
        };

        ws.current.onmessage = handleMessage;
    }, [config.url]);

    /**
     * Handles incoming WebSocket messages with compression support
     */
    const handleMessage = useCallback((event: MessageEvent) => {
        try {
            const startTime = performance.now();
            
            // Handle binary messages with compression
            let data: WebSocketMessage;
            if (event.data instanceof ArrayBuffer) {
                const decompressed = config.compressionEnabled ? 
                    LZ4.decode(new Uint8Array(event.data)) :
                    new TextDecoder().decode(event.data);
                data = JSON.parse(decompressed);
            } else {
                data = JSON.parse(event.data);
            }

            // Validate message structure
            if (!isValidWebSocketMessage(data)) {
                throw new Error('Invalid message format');
            }

            // Update latency metrics
            const messageLatency = performance.now() - startTime;
            updateLatencyMetrics(messageLatency);

            // Process message based on type
            switch (data.type) {
                case WebSocketEventType.FLEET_UPDATE:
                    if (isFleetUpdateEvent(data.payload)) {
                        dispatch({ type: 'fleet/update', payload: data.payload });
                    }
                    break;
                case WebSocketEventType.GAME_STATE:
                    dispatch({ type: 'game/updateState', payload: data.payload });
                    break;
                case WebSocketEventType.PROXIMITY_UPDATE:
                    dispatch({ type: 'proximity/update', payload: data.payload });
                    break;
                case WebSocketEventType.LATENCY_CHECK:
                    handleLatencyCheck(data);
                    break;
            }

            setLastMessage(data);
        } catch (error) {
            console.error('Message processing error:', error);
            setError(error as Error);
        }
    }, [dispatch, config.compressionEnabled]);

    /**
     * Sends WebSocket message with compression support
     */
    const sendMessage = useCallback((message: WebSocketMessage) => {
        if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
            if (messageQueue.current.length < MAX_QUEUE_SIZE) {
                messageQueue.current.push(message);
            }
            return;
        }

        try {
            const data = JSON.stringify(message);
            if (config.compressionEnabled && data.length > 1024) {
                const compressed = LZ4.encode(data);
                ws.current.send(compressed);
            } else {
                ws.current.send(data);
            }
        } catch (error) {
            console.error('Message send error:', error);
            setError(error as Error);
        }
    }, [config.compressionEnabled]);

    /**
     * Handles reconnection with exponential backoff
     */
    const handleReconnect = useCallback(() => {
        if (reconnectCount.current >= config.reconnectAttempts) {
            setError(new Error('Maximum reconnection attempts reached'));
            return;
        }

        const backoffDelay = Math.min(1000 * Math.pow(2, reconnectCount.current), 10000);
        reconnectTimeout.current = setTimeout(() => {
            reconnectCount.current++;
            initializeWebSocket();
        }, backoffDelay);
    }, [config.reconnectAttempts, initializeWebSocket]);

    /**
     * Implements heartbeat mechanism for connection monitoring
     */
    const startHeartbeat = useCallback(() => {
        const heartbeat = setInterval(() => {
            if (ws.current?.readyState === WebSocket.OPEN) {
                sendMessage({
                    type: WebSocketEventType.LATENCY_CHECK,
                    payload: { timestamp: Date.now() },
                    timestamp: Date.now(),
                    deviceId: 'device-id',
                    messageId: crypto.randomUUID(),
                    error: null
                });
            }
        }, config.heartbeatInterval);

        return () => clearInterval(heartbeat);
    }, [config.heartbeatInterval, sendMessage]);

    /**
     * Updates latency metrics and connection quality
     */
    const updateLatencyMetrics = useCallback((newLatency: number) => {
        latencyHistory.current.push(newLatency);
        if (latencyHistory.current.length > MAX_LATENCY_HISTORY) {
            latencyHistory.current.shift();
        }

        const avgLatency = latencyHistory.current.reduce((a, b) => a + b, 0) / latencyHistory.current.length;
        setLatency(avgLatency);

        // Update connection quality based on latency
        if (avgLatency < 20) {
            setConnectionQuality(ConnectionQuality.EXCELLENT);
        } else if (avgLatency < 50) {
            setConnectionQuality(ConnectionQuality.GOOD);
        } else if (avgLatency < 100) {
            setConnectionQuality(ConnectionQuality.FAIR);
        } else {
            setConnectionQuality(ConnectionQuality.POOR);
        }
    }, []);

    /**
     * Processes queued messages after reconnection
     */
    const flushMessageQueue = useCallback(() => {
        while (messageQueue.current.length > 0) {
            const message = messageQueue.current.shift();
            if (message) sendMessage(message);
        }
    }, [sendMessage]);

    /**
     * Handles latency check responses
     */
    const handleLatencyCheck = useCallback((message: WebSocketMessage) => {
        const roundTripTime = Date.now() - message.payload.timestamp;
        updateLatencyMetrics(roundTripTime / 2); // One-way latency
    }, [updateLatencyMetrics]);

    // Initialize WebSocket connection
    useEffect(() => {
        initializeWebSocket();
        return () => {
            ws.current?.close();
            clearTimeout(reconnectTimeout.current);
        };
    }, [initializeWebSocket]);

    return {
        connected,
        sendMessage,
        lastMessage,
        error,
        connectionQuality,
        latency
    };
}