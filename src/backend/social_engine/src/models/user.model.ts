import { z } from 'zod'; // v3.21.4
import bcrypt from 'bcryptjs'; // v2.4.3
import { DatabaseService, getPrisma } from '../config/database';
import { 
    User, 
    UserProfile, 
    UserPreferences,
    createUserSchema,
    updateUserSchema,
    userPreferencesSchema
} from '../types/user.types';
import { ROLES, PERMISSIONS } from '../../security/src/rbac';

/**
 * UserModel class handles all user-related operations with comprehensive validation,
 * security measures, and fleet coordination capabilities.
 */
export class UserModel {
    private db: DatabaseService;
    private readonly SALT_ROUNDS = 12;
    private readonly DEFAULT_SCAN_RESOLUTION = 0.01;
    private readonly DEFAULT_SCAN_RANGE = 5;

    constructor() {
        this.db = DatabaseService.getInstance();
    }

    /**
     * Creates a new user with validated data, secure password hashing,
     * and initial profile/preferences setup
     */
    public async createUser(userData: z.infer<typeof createUserSchema>): Promise<User> {
        try {
            // Validate user data
            const validatedData = createUserSchema.parse(userData);

            const prisma = await getPrisma();

            // Check email uniqueness
            const existingUser = await prisma.user.findUnique({
                where: { email: validatedData.email }
            });

            if (existingUser) {
                throw new Error('Email already registered');
            }

            // Hash password
            const hashedPassword = await bcrypt.hash(
                validatedData.password,
                this.SALT_ROUNDS
            );

            // Setup default preferences
            const defaultPreferences: UserPreferences = {
                scanResolution: validatedData.initialPreferences?.scanResolution || this.DEFAULT_SCAN_RESOLUTION,
                scanRange: validatedData.initialPreferences?.scanRange || this.DEFAULT_SCAN_RANGE,
                autoJoinFleet: validatedData.initialPreferences?.autoJoinFleet || false,
                defaultGameMode: validatedData.initialPreferences?.defaultGameMode || 'standard',
                notificationsEnabled: true,
                privacySettings: {
                    profileVisibility: 'public',
                    locationSharing: true,
                    fleetDiscoverable: true
                }
            };

            // Create user with profile and preferences
            const user = await prisma.user.create({
                data: {
                    email: validatedData.email,
                    username: validatedData.username,
                    password: hashedPassword,
                    role: ROLES.BASIC_USER,
                    permissions: [PERMISSIONS.SCAN, PERMISSIONS.FLEET_JOIN, PERMISSIONS.PROFILE_READ],
                    profile: {
                        create: {
                            displayName: validatedData.username,
                            level: 1,
                            experience: 0,
                            lastActive: new Date(),
                            preferences: defaultPreferences,
                            fleetHistory: [],
                            achievements: [],
                            totalGamesPlayed: 0,
                            winRate: 0
                        }
                    }
                },
                include: {
                    profile: true
                }
            });

            // Remove sensitive data before returning
            const { password: _, ...safeUser } = user;
            return safeUser as User;

        } catch (error) {
            if (error instanceof z.ZodError) {
                throw new Error(`Validation error: ${error.errors[0].message}`);
            }
            throw error;
        }
    }

    /**
     * Updates user profile with validation and fleet history tracking
     */
    public async updateUserProfile(
        userId: string,
        profileData: Partial<UserProfile>
    ): Promise<UserProfile> {
        try {
            const prisma = await getPrisma();

            // Validate profile data
            if (profileData.preferences) {
                userPreferencesSchema.parse(profileData.preferences);
            }

            // Check user exists
            const existingUser = await prisma.user.findUnique({
                where: { id: userId },
                include: { profile: true }
            });

            if (!existingUser) {
                throw new Error('User not found');
            }

            // Update profile
            const updatedProfile = await prisma.userProfile.update({
                where: { userId },
                data: {
                    displayName: profileData.displayName,
                    preferences: profileData.preferences ? {
                        ...existingUser.profile.preferences,
                        ...profileData.preferences
                    } : undefined,
                    fleetHistory: profileData.fleetHistory ? {
                        push: profileData.fleetHistory
                    } : undefined,
                    updatedAt: new Date()
                }
            });

            return updatedProfile;

        } catch (error) {
            if (error instanceof z.ZodError) {
                throw new Error(`Validation error: ${error.errors[0].message}`);
            }
            throw error;
        }
    }

    /**
     * Updates user's last active timestamp and fleet status
     */
    public async updateLastActive(userId: string): Promise<void> {
        try {
            const prisma = await getPrisma();

            await prisma.userProfile.update({
                where: { userId },
                data: {
                    lastActive: new Date()
                }
            });

        } catch (error) {
            throw new Error(`Failed to update last active status: ${error.message}`);
        }
    }

    /**
     * Updates user's fleet participation status
     */
    public async updateFleetStatus(
        userId: string,
        fleetId: string,
        status: 'joined' | 'left'
    ): Promise<void> {
        try {
            const prisma = await getPrisma();

            const user = await prisma.user.findUnique({
                where: { id: userId },
                include: { profile: true }
            });

            if (!user) {
                throw new Error('User not found');
            }

            const fleetHistory = user.profile.fleetHistory || [];
            
            if (status === 'joined' && !fleetHistory.includes(fleetId)) {
                await prisma.userProfile.update({
                    where: { userId },
                    data: {
                        fleetHistory: {
                            push: fleetId
                        }
                    }
                });
            }

        } catch (error) {
            throw new Error(`Failed to update fleet status: ${error.message}`);
        }
    }

    /**
     * Validates user credentials and returns user data
     */
    public async validateCredentials(
        email: string,
        password: string
    ): Promise<User | null> {
        try {
            const prisma = await getPrisma();

            const user = await prisma.user.findUnique({
                where: { email },
                include: { profile: true }
            });

            if (!user) {
                return null;
            }

            const isValid = await bcrypt.compare(password, user.password);
            if (!isValid) {
                return null;
            }

            const { password: _, ...safeUser } = user;
            return safeUser as User;

        } catch (error) {
            throw new Error(`Authentication failed: ${error.message}`);
        }
    }
}

export default UserModel;