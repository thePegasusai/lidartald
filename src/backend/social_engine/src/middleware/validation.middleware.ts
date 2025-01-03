import { Request, Response, NextFunction } from 'express'; // v4.18.x
import { z } from 'zod'; // v3.x
import { ApiError } from '../../api/src/middleware/error-handler';
import { userSchema, createUserSchema, updateUserSchema } from '../types/user.types';
import { fleetSchema, deviceCapabilitiesSchema, fleetDeviceSchema } from '../types/fleet.types';

/**
 * Configuration options for validation middleware behavior
 */
export interface ValidationOptions {
    stripUnknown?: boolean;  // Remove unknown fields from payload
    strict?: boolean;        // Enforce strict type checking
    cache?: boolean;         // Enable validation result caching
    errorMap?: z.ZodErrorMap; // Custom error messages
}

// Cache for validation results to improve performance
const validationCache = new Map<string, any>();

/**
 * Higher-order function that creates validation middleware for specific schemas
 * @param schema Zod schema to validate against
 * @param options Validation configuration options
 */
function validateRequest(schema: z.Schema, options: ValidationOptions = {}) {
    return async (req: Request, res: Response, next: NextFunction) => {
        try {
            const data = req.body || req.query || req.params;
            const cacheKey = options.cache ? 
                `${schema.toString()}-${JSON.stringify(data)}` : null;

            // Check cache if enabled
            if (cacheKey && validationCache.has(cacheKey)) {
                req.validatedData = validationCache.get(cacheKey);
                return next();
            }

            // Parse and validate data
            const validatedData = await schema.parseAsync(data, {
                strict: options.strict ?? true,
                stripUnknown: options.stripUnknown ?? true,
                errorMap: options.errorMap
            });

            // Cache result if enabled
            if (cacheKey) {
                validationCache.set(cacheKey, validatedData);
                // Implement cache expiry after 5 minutes
                setTimeout(() => validationCache.delete(cacheKey), 300000);
            }

            // Attach validated data to request
            req.validatedData = validatedData;
            next();
        } catch (error) {
            if (error instanceof z.ZodError) {
                next(new ApiError(400, 'Validation failed', {
                    errors: error.errors.map(err => ({
                        path: err.path.join('.'),
                        message: err.message,
                        code: 'VALIDATION_ERROR'
                    }))
                }));
            } else {
                next(error);
            }
        }
    };
}

/**
 * Middleware for validating request body data with strict checking
 * @param schema Zod schema to validate against
 */
export function validateBody(schema: z.Schema) {
    return validateRequest(schema, {
        strict: true,
        stripUnknown: true,
        cache: false,
        errorMap: (error, ctx) => {
            switch (error.code) {
                case z.ZodIssueCode.invalid_type:
                    return { message: `Expected ${error.expected}, received ${error.received}` };
                case z.ZodIssueCode.unrecognized_keys:
                    return { message: 'Unrecognized fields detected' };
                default:
                    return { message: ctx.defaultError };
            }
        }
    });
}

/**
 * Middleware for validating query parameters with type coercion
 * @param schema Zod schema to validate against
 */
export function validateQuery(schema: z.Schema) {
    return validateRequest(schema, {
        strict: false,
        stripUnknown: true,
        cache: true,
        errorMap: (error, ctx) => {
            if (error.code === z.ZodIssueCode.invalid_type) {
                return { message: `Invalid query parameter type: ${error.path.join('.')}` };
            }
            return { message: ctx.defaultError };
        }
    });
}

/**
 * Middleware for validating route parameters with caching
 * @param schema Zod schema to validate against
 */
export function validateParams(schema: z.Schema) {
    return validateRequest(schema, {
        strict: true,
        stripUnknown: false,
        cache: true,
        errorMap: (error, ctx) => {
            if (error.code === z.ZodIssueCode.invalid_type) {
                return { message: `Invalid route parameter: ${error.path.join('.')}` };
            }
            return { message: ctx.defaultError };
        }
    });
}

// Extend Express Request interface to include validated data
declare global {
    namespace Express {
        interface Request {
            validatedData: any;
        }
    }
}

// Export validation schemas for convenience
export const schemas = {
    user: userSchema,
    createUser: createUserSchema,
    updateUser: updateUserSchema,
    fleet: fleetSchema,
    fleetDevice: fleetDeviceSchema,
    deviceCapabilities: deviceCapabilitiesSchema
};