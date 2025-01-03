import { Request, Response, NextFunction, RequestHandler } from 'express'; // v4.18.2
import { RateLimiterRedis } from 'rate-limiter-flexible'; // v2.4.1
import winston from 'winston'; // v3.8.2
import { AuthService } from '../services/auth.service';
import { User } from '../types/user.types';

// Constants for authentication configuration
const TOKEN_HEADER = 'Authorization';
const TOKEN_PREFIX = 'Bearer ';
const MAX_REQUESTS_PER_IP = 100;
const REQUEST_WINDOW_MS = 60000; // 1 minute
const SUSPICIOUS_ATTEMPTS_THRESHOLD = 5;

// Initialize logger for security auditing
const securityLogger = winston.createLogger({
    level: 'info',
    format: winston.format.json(),
    defaultMeta: { service: 'auth-middleware' },
    transports: [
        new winston.transports.File({ filename: 'security-audit.log' }),
        new winston.transports.Console({
            format: winston.format.combine(
                winston.format.colorize(),
                winston.format.simple()
            )
        })
    ]
});

// Initialize rate limiter
const rateLimiter = new RateLimiterRedis({
    points: MAX_REQUESTS_PER_IP,
    duration: REQUEST_WINDOW_MS,
    blockDuration: REQUEST_WINDOW_MS,
    storeClient: AuthService.getInstance().getRedisClient()
});

/**
 * Extracts and validates JWT token from request header
 * @param req Express request object
 * @returns Validated token or null
 */
const extractToken = (req: Request): string | null => {
    const authHeader = req.header(TOKEN_HEADER);
    
    if (!authHeader || !authHeader.startsWith(TOKEN_PREFIX)) {
        return null;
    }

    const token = authHeader.slice(TOKEN_PREFIX.length);
    
    // Validate token format
    const tokenRegex = /^[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*$/;
    if (!tokenRegex.test(token)) {
        return null;
    }

    return token;
};

/**
 * Authentication middleware with hardware-backed security and comprehensive monitoring
 */
export const authenticate = async (
    req: Request,
    res: Response,
    next: NextFunction
): Promise<void> => {
    const ip = req.ip;
    
    try {
        // Apply rate limiting
        await rateLimiter.consume(ip);
        
        // Extract token
        const token = extractToken(req);
        if (!token) {
            securityLogger.warn('Missing or invalid token format', { ip });
            res.status(401).json({ error: 'Authentication required' });
            return;
        }

        // Validate token with hardware security module
        const authService = AuthService.getInstance();
        const validationResult = await authService.validateWithHSM(token);

        if (!validationResult.isValid) {
            securityLogger.warn('Invalid token', {
                ip,
                reason: validationResult.reason
            });
            res.status(401).json({ error: 'Invalid authentication token' });
            return;
        }

        // Validate session
        const sessionValidation = await authService.validateSession(
            validationResult.sessionId,
            validationResult.deviceId
        );

        if (!sessionValidation.isValid) {
            securityLogger.warn('Invalid session', {
                ip,
                sessionId: validationResult.sessionId
            });
            res.status(401).json({ error: 'Invalid session' });
            return;
        }

        // Attach validated user to request
        req.user = validationResult.user;

        // Log successful authentication
        securityLogger.info('Authentication successful', {
            userId: req.user.id,
            ip,
            sessionId: validationResult.sessionId
        });

        next();
    } catch (error) {
        if (error.name === 'RateLimiterError') {
            securityLogger.warn('Rate limit exceeded', { ip });
            res.status(429).json({ error: 'Too many requests' });
            return;
        }

        securityLogger.error('Authentication error', {
            ip,
            error: error.message
        });
        res.status(500).json({ error: 'Authentication failed' });
    }
};

/**
 * Role-based authorization middleware with audit logging
 * @param allowedRoles Array of roles allowed to access the resource
 */
export const authorize = (allowedRoles: Array<User['role']>): RequestHandler => {
    return async (req: Request, res: Response, next: NextFunction) => {
        try {
            const user = req.user as User;
            
            if (!user) {
                securityLogger.warn('Authorization attempted without user context', {
                    ip: req.ip
                });
                res.status(401).json({ error: 'Authentication required' });
                return;
            }

            const authService = AuthService.getInstance();
            const hasPermission = await authService.validateSession(
                req.headers['x-session-id'] as string,
                req.headers['x-device-id'] as string,
                allowedRoles
            );

            if (!hasPermission.isValid) {
                securityLogger.warn('Insufficient permissions', {
                    userId: user.id,
                    role: user.role,
                    requiredRoles: allowedRoles,
                    ip: req.ip
                });
                res.status(403).json({ error: 'Insufficient permissions' });
                return;
            }

            // Log successful authorization
            securityLogger.info('Authorization successful', {
                userId: user.id,
                role: user.role,
                ip: req.ip
            });

            next();
        } catch (error) {
            securityLogger.error('Authorization error', {
                error: error.message,
                ip: req.ip
            });
            res.status(500).json({ error: 'Authorization failed' });
        }
    };
};