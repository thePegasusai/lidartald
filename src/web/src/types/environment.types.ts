import { Box3 } from 'three'; // v0.150.0
import { Point3D } from './lidar.types';

/**
 * Constants for environment processing and validation
 */
export const MIN_FEATURE_CONFIDENCE = 0.85;
export const MAX_ENVIRONMENT_SIZE = 25; // square meters
export const UPDATE_INTERVAL_MS = 33; // ~30Hz update rate

/**
 * Enumeration of possible surface classifications in the environment
 */
export enum SurfaceClassification {
    Floor = 'FLOOR',
    Wall = 'WALL',
    Obstacle = 'OBSTACLE',
    Unknown = 'UNKNOWN'
}

/**
 * Interface representing a detected feature in the environment
 * Includes spatial data, classification, and metadata for tracking
 */
export interface EnvironmentFeature {
    /** Unique identifier for the feature */
    id: string;

    /** Type of the detected feature */
    type: string;

    /** Array of 3D points comprising the feature */
    points: Point3D[];

    /** Surface classification of the feature */
    classification: SurfaceClassification;

    /** Detection confidence score (0.0 - 1.0) */
    confidence: number;

    /** Additional feature metadata */
    metadata: Record<string, any>;

    /** Timestamp of last feature update */
    lastUpdated: number;

    /** 3D bounding box containing the feature */
    boundingBox: Box3;
}

/**
 * Interface representing a complete environment map
 * Contains all detected features, boundaries, and processing metadata
 */
export interface EnvironmentMap {
    /** Unique identifier for the environment map */
    id: string;

    /** Creation timestamp */
    timestamp: number;

    /** Array of all scanned points in the environment */
    points: Point3D[];

    /** Array of detected features */
    features: EnvironmentFeature[];

    /** 3D bounding box of entire environment */
    boundaries: Box3;

    /** Scanning resolution in centimeters */
    resolution: number;

    /** Environment map version number */
    version: number;

    /** Timestamp of last processing update */
    lastProcessed: number;
}

/**
 * Interface representing an incremental update to an environment map
 * Used for real-time updates at 30Hz frequency
 */
export interface EnvironmentUpdate {
    /** ID of the environment map being updated */
    mapId: string;

    /** Update timestamp */
    timestamp: number;

    /** New points added in this update */
    newPoints: Point3D[];

    /** Features that were modified or added */
    updatedFeatures: EnvironmentFeature[];

    /** IDs of features that were removed */
    removedFeatureIds: string[];

    /** Monotonically increasing sequence number */
    sequenceNumber: number;

    /** Changes to environment boundaries (null if unchanged) */
    boundaryChanges: Box3 | null;

    /** Indicates if this is a partial update */
    partialUpdate: boolean;
}