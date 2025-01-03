import { Matrix4 } from 'three'; // v0.150.0 - For 3D transformation matrix types

/**
 * Default scanning parameters for LiDAR configuration
 * Resolution: 0.01cm (ultra-high precision)
 * Range: 5.0m (maximum scanning distance)
 * ScanRate: 30Hz (real-time capture rate)
 */
export const DEFAULT_SCAN_PARAMETERS = {
    resolution: 0.01, // cm
    range: 5.0, // meters
    scanRate: 30 // Hz
} as const;

/**
 * Interface representing a 3D point with intensity data
 * Used for individual points in point cloud data structure
 */
export interface Point3D {
    x: number;        // X coordinate in 3D space
    y: number;        // Y coordinate in 3D space
    z: number;        // Z coordinate in 3D space
    intensity: number; // Point intensity value (0.0 - 1.0)
}

/**
 * Interface representing a complete point cloud dataset
 * Contains array of points and transformation data
 */
export interface PointCloud {
    points: Point3D[];      // Array of 3D points with intensity
    timestamp: number;      // Unix timestamp of scan
    transformMatrix: Matrix4; // 4x4 transformation matrix for point cloud
}

/**
 * Enum defining different types of detectable features
 * in the point cloud data
 */
export enum FeatureType {
    SURFACE = 'SURFACE',     // Flat surfaces like floors, walls
    OBSTACLE = 'OBSTACLE',   // Objects that obstruct movement
    BOUNDARY = 'BOUNDARY'    // Environment boundaries/edges
}

/**
 * Interface representing a detected feature in the point cloud
 * Contains feature identification and classification data
 */
export interface Feature {
    id: string;           // Unique feature identifier
    type: FeatureType;    // Type of detected feature
    coordinates: Point3D[]; // Points comprising the feature
    confidence: number;    // Detection confidence (0.0 - 1.0)
}

/**
 * Interface defining LiDAR scanning parameters
 * Controls the precision and performance of scanning
 */
export interface ScanParameters {
    resolution: number;  // Scanning resolution in cm
    range: number;      // Maximum scanning range in meters
    scanRate: number;   // Scanning frequency in Hz
}

/**
 * Interface representing a complete scan result
 * Includes point cloud, detected features, and quality metrics
 */
export interface ScanResult {
    pointCloud: PointCloud;        // Raw point cloud data
    features: Feature[];           // Detected features
    scanParameters: ScanParameters; // Used scan parameters
    quality: number;               // Overall scan quality (0.0 - 1.0)
}