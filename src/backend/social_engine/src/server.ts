import express, { Express } from 'express'; // v4.18.x
import http from 'http';
import cors from 'cors'; // v2.8.x
import helmet from 'helmet'; // v7.0.x
import compression from 'compression'; // v1.7.x
import debug from 'debug'; // v4.x
import { WebSocketManager } from './config/websocket';
import { errorHandler } from './middleware/error-handler';
import { rateLimiter } from './middleware/rate-limiter';
import { getDatabaseService } from './config/database';

// Initialize debug logger
const log = debug('tald:social-engine:server');

// Environment variables with defaults
const PORT = process.env.PORT || 3000;
const WS_PORT = process.env.WS_PORT || 8080;
const NODE_ENV = process.env.NODE_ENV || 'development';
const MAX_CONNECTIONS = process.env.MAX_CONNECTIONS || 32;
const SHUTDOWN_TIMEOUT = process.env.SHUTDOWN_TIMEOUT || 10000;

// Track active connections for graceful shutdown
const connections = new Set<http.ServerResponse>();

/**
 * Creates and configures the Express application instance
 */
function createExpressApp(): Express {
  const app = express();

  // Security middleware configuration
  app.use(helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'"],
        styleSrc: ["'self'"],
        imgSrc: ["'self'", 'data:', 'blob:'],
        connectSrc: ["'self'", 'wss:', 'ws:'],
      },
    },
    hsts: {
      maxAge: 31536000,
      includeSubDomains: true,
      preload: true
    }
  }));

  // CORS configuration for fleet communication
  app.use(cors({
    origin: (origin, callback) => {
      const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'];
      if (!origin || allowedOrigins.includes(origin)) {
        callback(null, true);
      } else {
        callback(new Error('CORS not allowed'));
      }
    },
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Request-ID'],
    credentials: true,
    maxAge: 86400
  }));

  // Request parsing and compression
  app.use(express.json({ limit: '1mb' }));
  app.use(express.urlencoded({ extended: true, limit: '1mb' }));
  app.use(compression({
    level: 6,
    threshold: 1024,
    filter: (req, res) => {
      if (req.headers['x-no-compression']) {
        return false;
      }
      return compression.filter(req, res);
    }
  }));

  // Rate limiting configuration
  app.use(rateLimiter({
    windowMs: 60000,
    maxRequests: 100,
    keyPrefix: 'social-engine',
    endpointPattern: '^(fleet|environment|session|user)/',
    enableFallback: true,
    redisConfig: {
      host: process.env.REDIS_HOST || 'localhost',
      port: Number(process.env.REDIS_PORT) || 6379,
      tls: process.env.REDIS_TLS === 'true'
    },
    metrics: {
      enabled: true,
      prefix: 'social_engine'
    }
  }));

  // Health check endpoint
  app.get('/health', (req, res) => {
    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      connections: connections.size,
      environment: NODE_ENV
    });
  });

  // Error handling
  app.use(errorHandler);

  return app;
}

/**
 * Starts both HTTP and WebSocket servers
 */
async function start(): Promise<void> {
  try {
    const app = createExpressApp();
    const server = http.createServer(app);

    // Track connections for graceful shutdown
    server.on('connection', (socket) => {
      const res = socket as unknown as http.ServerResponse;
      connections.add(res);
      socket.on('close', () => {
        connections.delete(res);
      });
    });

    // Initialize WebSocket server
    const wsManager = WebSocketManager.getInstance();
    await wsManager.start({
      port: Number(WS_PORT),
      host: '0.0.0.0',
      maxConnections: Number(MAX_CONNECTIONS),
      heartbeatInterval: 30000,
      reconnectInterval: 5000,
      messageTimeout: 10000,
      compression: {
        level: 6,
        threshold: 1024
      },
      security: {
        rateLimitPerMinute: 100,
        maxMessageSize: 1048576,
        enableOriginCheck: true
      }
    });

    // Start HTTP server
    server.listen(PORT, () => {
      log(`Social Engine server started on port ${PORT} (${NODE_ENV})`);
    });

    // Setup graceful shutdown
    process.on('SIGTERM', () => handleShutdown('SIGTERM'));
    process.on('SIGINT', () => handleShutdown('SIGINT'));

  } catch (error) {
    log('Failed to start server:', error);
    process.exit(1);
  }
}

/**
 * Gracefully shuts down servers with connection draining
 */
async function stop(): Promise<void> {
  log('Initiating graceful shutdown...');

  // Close WebSocket connections
  const wsManager = WebSocketManager.getInstance();
  await wsManager.stop();

  // Drain HTTP connections
  for (const conn of connections) {
    conn.end();
  }

  // Disconnect from databases
  await getDatabaseService().disconnect();

  log('Graceful shutdown completed');
}

/**
 * Handles shutdown signals with state persistence
 */
async function handleShutdown(signal: string): Promise<void> {
  log(`Received ${signal} signal`);

  try {
    const timeoutId = setTimeout(() => {
      log('Shutdown timeout exceeded, forcing exit');
      process.exit(1);
    }, Number(SHUTDOWN_TIMEOUT));

    await stop();
    clearTimeout(timeoutId);
    process.exit(0);
  } catch (error) {
    log('Error during shutdown:', error);
    process.exit(1);
  }
}

// Export server lifecycle management
export const server = {
  start,
  stop,
  health: () => ({
    connections: connections.size,
    environment: NODE_ENV
  })
};