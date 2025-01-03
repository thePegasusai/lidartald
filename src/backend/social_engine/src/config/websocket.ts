import WebSocket from 'ws'; // v8.x
import { z } from 'zod'; // v3.x
import compression from 'ws-compression'; // v1.x
import { Meter, MeterProvider } from '@opentelemetry/metrics'; // v1.x
import { getDatabaseService } from './database';

// Environment configuration with defaults and validation
const WS_PORT = Number(process.env.WS_PORT) || 8080;
const WS_HOST = process.env.WS_HOST || '0.0.0.0';
const WS_MAX_CONNECTIONS = 32; // As per spec: up to 32 connected devices
const WS_HEARTBEAT_INTERVAL = 30000; // 30 seconds
const WS_RECONNECT_INTERVAL = 5000; // 5 seconds
const WS_MESSAGE_TIMEOUT = 10000; // 10 seconds
const WS_COMPRESSION_LEVEL = 6;
const WS_RATE_LIMIT = 100; // messages per minute
const WS_RATE_WINDOW = 60000; // 1 minute

// Configuration validation schema
const WebSocketConfigSchema = z.object({
  port: z.number().min(1024).max(65535),
  host: z.string().ip(),
  maxConnections: z.number().max(32),
  heartbeatInterval: z.number().min(1000).max(60000),
  reconnectInterval: z.number().min(1000).max(30000),
  messageTimeout: z.number().min(1000).max(30000),
  compression: z.object({
    level: z.number().min(1).max(9),
    threshold: z.number().min(100),
  }),
  security: z.object({
    rateLimitPerMinute: z.number().min(1),
    maxMessageSize: z.number().max(1048576), // 1MB
    enableOriginCheck: z.boolean(),
  }),
});

type WebSocketConfig = z.infer<typeof WebSocketConfigSchema>;

// Connection pool for managing active connections
class ConnectionPool {
  private connections: Map<string, WebSocket> = new Map();
  private metrics: Meter;

  constructor(metrics: Meter) {
    this.metrics = metrics;
  }

  add(id: string, socket: WebSocket): boolean {
    if (this.connections.size >= WS_MAX_CONNECTIONS) {
      return false;
    }
    this.connections.set(id, socket);
    this.metrics.createCounter('ws.connections.active').add(1);
    return true;
  }

  remove(id: string): void {
    if (this.connections.delete(id)) {
      this.metrics.createCounter('ws.connections.active').add(-1);
    }
  }

  getSize(): number {
    return this.connections.size;
  }
}

// WebSocket Manager implementation
export class WebSocketManager {
  private static instance: WebSocketManager;
  private server?: WebSocket.Server;
  private pool: ConnectionPool;
  private metrics: Meter;
  private heartbeatInterval?: NodeJS.Timeout;

  private constructor() {
    this.metrics = new MeterProvider().getMeter('websocket-server');
    this.pool = new ConnectionPool(this.metrics);
  }

  public static getInstance(): WebSocketManager {
    if (!WebSocketManager.instance) {
      WebSocketManager.instance = new WebSocketManager();
    }
    return WebSocketManager.instance;
  }

  public async start(config: WebSocketConfig): Promise<void> {
    try {
      // Validate configuration
      WebSocketConfigSchema.parse(config);

      // Initialize WebSocket server with security options
      this.server = new WebSocket.Server({
        port: config.port,
        host: config.host,
        perMessageDeflate: {
          zlibDeflateOptions: {
            level: config.compression.level,
          },
          threshold: config.compression.threshold,
        },
        maxPayload: config.security.maxMessageSize,
        verifyClient: this.verifyClient.bind(this),
      });

      // Setup server event handlers
      this.setupServerHandlers();

      // Start heartbeat monitoring
      this.startHeartbeat(config.heartbeatInterval);

      // Initialize metrics collection
      this.initializeMetrics();

      console.log(`WebSocket server started on ${config.host}:${config.port}`);
    } catch (error) {
      console.error('Failed to start WebSocket server:', error);
      throw error;
    }
  }

  private setupServerHandlers(): void {
    if (!this.server) return;

    this.server.on('connection', (socket: WebSocket, request: any) => {
      const clientId = request.headers['x-client-id'] || crypto.randomUUID();

      if (!this.pool.add(clientId, socket)) {
        socket.close(1013, 'Maximum connections reached');
        return;
      }

      // Setup socket event handlers
      this.setupSocketHandlers(socket, clientId);
    });

    this.server.on('error', (error: Error) => {
      console.error('WebSocket server error:', error);
      this.metrics.createCounter('ws.server.errors').add(1);
    });
  }

  private setupSocketHandlers(socket: WebSocket, clientId: string): void {
    let messageCount = 0;
    let lastMessageTime = Date.now();

    socket.on('message', async (data: WebSocket.Data) => {
      try {
        // Rate limiting check
        if (this.isRateLimited(messageCount, lastMessageTime)) {
          socket.close(1008, 'Rate limit exceeded');
          return;
        }

        messageCount++;
        lastMessageTime = Date.now();

        // Process message
        await this.handleMessage(socket, data);

        this.metrics.createCounter('ws.messages.processed').add(1);
      } catch (error) {
        console.error('Message handling error:', error);
        this.metrics.createCounter('ws.messages.errors').add(1);
      }
    });

    socket.on('close', () => {
      this.pool.remove(clientId);
      this.metrics.createCounter('ws.connections.closed').add(1);
    });

    socket.on('error', (error: Error) => {
      console.error(`Socket error for client ${clientId}:`, error);
      this.metrics.createCounter('ws.connection.errors').add(1);
    });
  }

  private async handleMessage(socket: WebSocket, data: WebSocket.Data): Promise<void> {
    const db = await getDatabaseService().getPrisma();
    // Message handling implementation
    // Additional logic would go here based on message type
  }

  private isRateLimited(messageCount: number, lastMessageTime: number): boolean {
    const timeWindow = Date.now() - lastMessageTime;
    return messageCount >= WS_RATE_LIMIT && timeWindow <= WS_RATE_WINDOW;
  }

  private verifyClient(info: { origin: string; secure: boolean; req: any }): boolean {
    // Implement client verification logic
    return true; // Simplified for example
  }

  private startHeartbeat(interval: number): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.server) {
        this.server.clients.forEach((socket) => {
          if (socket.readyState === WebSocket.OPEN) {
            socket.ping();
            this.metrics.createCounter('ws.heartbeats.sent').add(1);
          }
        });
      }
    }, interval);
  }

  private initializeMetrics(): void {
    // Initialize metrics collectors
    this.metrics.createCounter('ws.server.starts');
    this.metrics.createCounter('ws.connections.total');
    this.metrics.createCounter('ws.messages.total');
    this.metrics.createHistogram('ws.message.size');
    this.metrics.createHistogram('ws.message.latency');
  }

  public async stop(): Promise<void> {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    if (this.server) {
      this.server.close();
      this.server = undefined;
    }
  }
}

// Export default configuration
export const webSocketConfig: WebSocketConfig = {
  port: WS_PORT,
  host: WS_HOST,
  maxConnections: WS_MAX_CONNECTIONS,
  heartbeatInterval: WS_HEARTBEAT_INTERVAL,
  reconnectInterval: WS_RECONNECT_INTERVAL,
  messageTimeout: WS_MESSAGE_TIMEOUT,
  compression: {
    level: WS_COMPRESSION_LEVEL,
    threshold: 1024, // 1KB minimum for compression
  },
  security: {
    rateLimitPerMinute: WS_RATE_LIMIT,
    maxMessageSize: 1048576, // 1MB
    enableOriginCheck: true,
  },
};

// Export singleton instance getter
export const getWebSocketManager = WebSocketManager.getInstance;