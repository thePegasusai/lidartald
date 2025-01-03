import { Request, Response, NextFunction } from 'express'; // express@4.18.x
import { Logger } from 'winston'; // winston@3.8.x
import { FleetError } from '../../fleet_manager/src/error';
import { v4 as uuidv4 } from 'uuid';

// Error code ranges based on system components
const ERROR_CODES = {
  LIDAR_HARDWARE: { MIN: 1000, MAX: 1999 },
  POINT_CLOUD_PROCESSING: { MIN: 2000, MAX: 2999 },
  NETWORK_COMMUNICATION: { MIN: 3000, MAX: 3999 },
  FLEET_COORDINATION: { MIN: 4000, MAX: 4999 },
  GAME_INTEGRATION: { MIN: 5000, MAX: 5999 },
  SYSTEM_CRITICAL: { MIN: 9000, MAX: 9999 }
} as const;

// Logging levels for different error severities
const ERROR_LEVELS = {
  WARN: 'warn',
  ERROR: 'error',
  CRITICAL: 'critical'
} as const;

/**
 * Custom API error class with enhanced error details and tracking
 */
export class ApiError extends Error {
  public readonly statusCode: number;
  public readonly details: any;
  public readonly errorCode: string;
  public readonly requestId: string;

  constructor(statusCode: number, message: string, details?: any) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.details = this.sanitizeErrorDetails(details);
    this.requestId = uuidv4();
    this.errorCode = this.determineErrorCode();
    Error.captureStackTrace(this, ApiError);
  }

  private sanitizeErrorDetails(details: any): any {
    if (!details) return undefined;
    
    // Remove sensitive information
    const sanitized = { ...details };
    const sensitiveFields = ['password', 'token', 'key', 'secret'];
    sensitiveFields.forEach(field => {
      if (field in sanitized) delete sanitized[field];
    });
    
    return sanitized;
  }

  private determineErrorCode(): string {
    // Map status code ranges to error code ranges
    if (this.statusCode >= 500) return `${ERROR_CODES.SYSTEM_CRITICAL.MIN}`;
    if (this.message.toLowerCase().includes('fleet')) return `${ERROR_CODES.FLEET_COORDINATION.MIN}`;
    if (this.message.toLowerCase().includes('lidar')) return `${ERROR_CODES.LIDAR_HARDWARE.MIN}`;
    return `${ERROR_CODES.NETWORK_COMMUNICATION.MIN}`;
  }
}

/**
 * Maps different error types to appropriate HTTP status codes
 */
function mapErrorToStatusCode(error: Error): number {
  if (error instanceof ApiError) {
    return error.statusCode;
  }

  if (error instanceof FleetError) {
    switch(true) {
      case error instanceof FleetError.DiscoveryError:
        return 404;
      case error instanceof FleetError.ConnectionError:
        return 503;
      default:
        return 500;
    }
  }

  // Map common error types
  switch(error.name) {
    case 'ValidationError':
      return 400;
    case 'UnauthorizedError':
      return 401;
    case 'ForbiddenError':
      return 403;
    case 'RateLimitError':
      return 429;
    default:
      return 500;
  }
}

/**
 * Global error handling middleware
 */
export function errorHandler(
  error: Error,
  req: Request,
  res: Response,
  next: NextFunction
): void {
  // If headers already sent, delegate to default Express error handler
  if (res.headersSent) {
    return next(error);
  }

  const requestId = error instanceof ApiError ? error.requestId : uuidv4();
  const statusCode = mapErrorToStatusCode(error);
  
  // Determine logging level based on status code
  const logLevel = statusCode >= 500 ? ERROR_LEVELS.CRITICAL :
                  statusCode >= 400 ? ERROR_LEVELS.ERROR :
                  ERROR_LEVELS.WARN;

  // Log error with appropriate level
  const logger: Logger = req.app.get('logger');
  logger[logLevel]({
    requestId,
    message: error.message,
    stack: error.stack,
    statusCode,
    path: req.path,
    method: req.method,
    timestamp: new Date().toISOString()
  });

  // Format error response
  const errorResponse = {
    status: 'error',
    statusCode,
    message: error.message,
    requestId,
    errorCode: error instanceof ApiError ? error.errorCode : 
               error instanceof FleetError ? error.error_code().toString() :
               `${ERROR_CODES.SYSTEM_CRITICAL.MIN}`,
    details: error instanceof ApiError ? error.details : undefined,
    timestamp: new Date().toISOString()
  };

  // Add security headers
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-Request-ID', requestId);

  // Send error response
  res.status(statusCode).json(errorResponse);

  // Track error metrics if monitoring is enabled
  if (req.app.get('metrics')) {
    req.app.get('metrics').incrementCounter('api_errors_total', {
      status_code: statusCode,
      error_code: errorResponse.errorCode,
      path: req.path
    });
  }
}