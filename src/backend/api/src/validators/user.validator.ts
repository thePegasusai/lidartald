import { z } from 'zod'; // v3.x
import { Request, Response, NextFunction } from 'express'; // v4.18.x
import { RateLimit } from 'express-rate-limit'; // v6.x
import { User } from '../../../social_engine/src/types/user.types';
import { ROLES, PERMISSIONS } from '../../../security/src/rbac';

// Constants for validation rules
const PASSWORD_MIN_LENGTH = 8;
const USERNAME_MIN_LENGTH = 3;
const USERNAME_MAX_LENGTH = 30;
const ALLOWED_EMAIL_DOMAINS = ['gmail.com', 'yahoo.com', 'hotmail.com'];
const PASSWORD_COMPLEXITY_REGEX = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
const USERNAME_PATTERN_REGEX = /^[a-zA-Z0-9_-]+$/;
const EMAIL_PATTERN_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

/**
 * Enhanced schema for user creation with strict validation rules
 */
export const createUserSchema = z.object({
    email: z.string()
        .email('Invalid email format')
        .regex(EMAIL_PATTERN_REGEX, 'Invalid email format')
        .refine(
            (email) => ALLOWED_EMAIL_DOMAINS.some(domain => email.endsWith(`@${domain}`)),
            'Email domain not allowed'
        ),
    username: z.string()
        .min(USERNAME_MIN_LENGTH, `Username must be at least ${USERNAME_MIN_LENGTH} characters`)
        .max(USERNAME_MAX_LENGTH, `Username cannot exceed ${USERNAME_MAX_LENGTH} characters`)
        .regex(USERNAME_PATTERN_REGEX, 'Username can only contain letters, numbers, underscores, and hyphens')
        .refine(
            (username) => !username.toLowerCase().includes('admin'),
            'Username cannot contain restricted words'
        ),
    password: z.string()
        .min(PASSWORD_MIN_LENGTH, `Password must be at least ${PASSWORD_MIN_LENGTH} characters`)
        .regex(
            PASSWORD_COMPLEXITY_REGEX,
            'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character'
        ),
    role: z.enum([ROLES.GUEST, ROLES.BASIC_USER, ROLES.PREMIUM_USER, ROLES.DEVELOPER])
        .default(ROLES.BASIC_USER)
});

/**
 * Enhanced schema for user updates with role-based restrictions
 */
export const updateUserSchema = z.object({
    email: z.string()
        .email('Invalid email format')
        .regex(EMAIL_PATTERN_REGEX, 'Invalid email format')
        .refine(
            (email) => ALLOWED_EMAIL_DOMAINS.some(domain => email.endsWith(`@${domain}`)),
            'Email domain not allowed'
        )
        .optional(),
    username: z.string()
        .min(USERNAME_MIN_LENGTH, `Username must be at least ${USERNAME_MIN_LENGTH} characters`)
        .max(USERNAME_MAX_LENGTH, `Username cannot exceed ${USERNAME_MAX_LENGTH} characters`)
        .regex(USERNAME_PATTERN_REGEX, 'Username can only contain letters, numbers, underscores, and hyphens')
        .refine(
            (username) => !username.toLowerCase().includes('admin'),
            'Username cannot contain restricted words'
        )
        .optional(),
    role: z.enum([ROLES.BASIC_USER, ROLES.PREMIUM_USER, ROLES.DEVELOPER])
        .optional()
});

/**
 * Middleware for validating user creation requests with enhanced security measures
 */
@RateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 5 // limit each IP to 5 requests per minute
})
export const validateCreateUser = async (
    req: Request,
    res: Response,
    next: NextFunction
): Promise<void> => {
    try {
        // Sanitize input data
        const sanitizedData = {
            email: req.body.email?.toLowerCase().trim(),
            username: req.body.username?.trim(),
            password: req.body.password,
            role: req.body.role
        };

        // Validate against schema
        const validatedData = await createUserSchema.parseAsync(sanitizedData);

        // Additional security checks
        if (validatedData.password.toLowerCase().includes(validatedData.username.toLowerCase())) {
            throw new Error('Password cannot contain username');
        }

        // Attach validated data to request
        req.body = validatedData;
        next();
    } catch (error) {
        // Sanitize error message for security
        const safeErrorMessage = error instanceof z.ZodError
            ? error.errors.map(e => e.message).join(', ')
            : 'Validation failed';

        res.status(400).json({
            status: 'error',
            message: safeErrorMessage
        });
    }
};

/**
 * Middleware for validating user update requests with role-based permissions
 */
@RateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 10 // limit each IP to 10 requests per minute
})
export const validateUpdateUser = async (
    req: Request,
    res: Response,
    next: NextFunction
): Promise<void> => {
    try {
        // Verify user has required permissions
        const userRole = req.user?.role;
        if (!userRole || !(await hasPermission(userRole, PERMISSIONS.PROFILE_WRITE))) {
            throw new Error('Insufficient permissions');
        }

        // Sanitize input data
        const sanitizedData = {
            email: req.body.email?.toLowerCase().trim(),
            username: req.body.username?.trim(),
            role: req.body.role
        };

        // Validate against schema
        const validatedData = await updateUserSchema.parseAsync(sanitizedData);

        // Role-based validation
        if (validatedData.role && userRole !== ROLES.ADMIN) {
            throw new Error('Role modification not allowed');
        }

        // Attach validated data to request
        req.body = validatedData;
        next();
    } catch (error) {
        // Sanitize error message for security
        const safeErrorMessage = error instanceof z.ZodError
            ? error.errors.map(e => e.message).join(', ')
            : 'Validation failed';

        res.status(400).json({
            status: 'error',
            message: safeErrorMessage
        });
    }
};

/**
 * Helper function to check user permissions
 */
async function hasPermission(role: string, permission: string): Promise<boolean> {
    // Implementation would use the RBAC service
    return Object.values(ROLES).includes(role) && 
           Object.values(PERMISSIONS).includes(permission);
}