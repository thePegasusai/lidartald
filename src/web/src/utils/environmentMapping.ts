import * as THREE from 'three'; // v0.150.0
import { GPU } from 'gpu.js'; // v2.16.0
import { 
    EnvironmentMap, 
    EnvironmentFeature, 
    EnvironmentUpdate, 
    SurfaceClassification,
    MIN_FEATURE_CONFIDENCE,
    MAX_ENVIRONMENT_SIZE
} from '../types/environment.types';
import { 
    processPointCloud, 
    detectFeatures, 
    mergePointClouds, 
    transformPointCloud 
} from './pointCloud';
import { Point3D, PointCloud } from '../types/lidar.types';

// Constants for environment mapping configuration
const UPDATE_INTERVAL_MS = 33; // 30Hz update rate
const POINT_CLOUD_BUFFER_SIZE = 1048576; // Maximum points in memory buffer
const SAFETY_MARGIN = 0.1; // 10cm safety margin for boundaries

// Initialize GPU context for environment processing
const gpu = new GPU({
    mode: 'gpu',
    tactic: 'precision'
});

/**
 * Creates a new environment map from initial point cloud data with GPU acceleration
 * @param points Initial point cloud data
 * @param resolution Scanning resolution in centimeters (minimum 0.01cm)
 * @param gpuContext GPU context for accelerated processing
 * @returns Promise resolving to new environment map
 */
export async function createEnvironmentMap(
    points: Point3D[],
    resolution: number,
    gpuContext: GPU
): Promise<EnvironmentMap> {
    // Validate input parameters
    if (resolution < 0.01) {
        throw new Error('Resolution must be at least 0.01cm');
    }

    // Initialize point cloud structure
    const initialCloud: PointCloud = {
        points,
        timestamp: Date.now(),
        transformMatrix: new THREE.Matrix4()
    };

    // Process initial point cloud with GPU acceleration
    const processedCloud = await processPointCloud(initialCloud, {
        resolution,
        range: 5.0,
        scanRate: 30
    });

    // Detect initial features
    const features = await detectFeatures(processedCloud);

    // Calculate environment boundaries
    const boundaries = calculateBoundaries(processedCloud.points, SAFETY_MARGIN);

    // Create and return new environment map
    return {
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        points: processedCloud.points,
        features: features.map(f => ({
            id: f.id,
            type: f.type,
            points: f.coordinates,
            classification: SurfaceClassification.Unknown,
            confidence: f.confidence,
            metadata: {},
            lastUpdated: Date.now(),
            boundingBox: new THREE.Box3().setFromPoints(
                f.coordinates.map(p => new THREE.Vector3(p.x, p.y, p.z))
            )
        })),
        boundaries,
        resolution,
        version: 1,
        lastProcessed: Date.now()
    };
}

/**
 * Updates existing environment map with new point cloud data using incremental processing
 * @param currentMap Current environment map state
 * @param update Incremental update data
 * @param gpuContext GPU context for accelerated processing
 * @returns Promise resolving to updated environment map
 */
export async function updateEnvironment(
    currentMap: EnvironmentMap,
    update: EnvironmentUpdate,
    gpuContext: GPU
): Promise<EnvironmentMap> {
    // Validate update version sequence
    if (update.sequenceNumber <= currentMap.version) {
        throw new Error('Invalid update sequence');
    }

    // Merge new points with existing map
    const mergedCloud = mergePointClouds([
        { 
            points: currentMap.points,
            timestamp: currentMap.timestamp,
            transformMatrix: new THREE.Matrix4()
        },
        {
            points: update.newPoints,
            timestamp: update.timestamp,
            transformMatrix: new THREE.Matrix4()
        }
    ]);

    // Enforce point cloud buffer size limit
    if (mergedCloud.points.length > POINT_CLOUD_BUFFER_SIZE) {
        mergedCloud.points = mergedCloud.points.slice(-POINT_CLOUD_BUFFER_SIZE);
    }

    // Update features incrementally
    const updatedFeatures = [...currentMap.features];
    
    // Remove obsolete features
    update.removedFeatureIds.forEach(id => {
        const index = updatedFeatures.findIndex(f => f.id === id);
        if (index !== -1) {
            updatedFeatures.splice(index, 1);
        }
    });

    // Add/update new features
    update.updatedFeatures.forEach(feature => {
        const index = updatedFeatures.findIndex(f => f.id === feature.id);
        if (index !== -1) {
            updatedFeatures[index] = feature;
        } else {
            updatedFeatures.push(feature);
        }
    });

    // Update boundaries if needed
    const boundaries = update.boundaryChanges || currentMap.boundaries;

    return {
        ...currentMap,
        points: mergedCloud.points,
        features: updatedFeatures,
        boundaries,
        version: update.sequenceNumber,
        lastProcessed: Date.now()
    };
}

/**
 * Classifies surfaces in the environment map using GPU-accelerated algorithms
 * @param map Environment map to classify
 * @param gpuContext GPU context for accelerated processing
 * @returns Promise resolving to classified surface features
 */
export async function classifySurfaces(
    map: EnvironmentMap,
    gpuContext: GPU
): Promise<EnvironmentFeature[]> {
    // Initialize GPU kernel for surface classification
    const classifyKernel = gpuContext.createKernel(function(
        points: number[][],
        normalThreshold: number
    ) {
        const idx = this.thread.x;
        const normal = calculateNormal(points, idx);
        const verticalAlignment = Math.abs(normal[1]); // Y-axis alignment
        
        // Classify based on normal vector alignment
        if (verticalAlignment > normalThreshold) {
            return SurfaceClassification.Floor;
        } else if (verticalAlignment < 0.1) {
            return SurfaceClassification.Wall;
        }
        return SurfaceClassification.Unknown;
    })
    .setOutput([map.points.length])
    .setTactic('precision');

    // Prepare point data for GPU processing
    const pointsArray = map.points.map(p => [p.x, p.y, p.z]);
    
    // Execute GPU-accelerated classification
    const classifications = await classifyKernel(pointsArray, 0.9) as SurfaceClassification[];

    // Group points by classification
    const classifiedFeatures: EnvironmentFeature[] = [];
    const pointGroups = new Map<SurfaceClassification, Point3D[]>();

    map.points.forEach((point, index) => {
        const classification = classifications[index];
        if (!pointGroups.has(classification)) {
            pointGroups.set(classification, []);
        }
        pointGroups.get(classification)!.push(point);
    });

    // Create features for each classification group
    pointGroups.forEach((points, classification) => {
        if (points.length > 0) {
            classifiedFeatures.push({
                id: crypto.randomUUID(),
                type: 'SURFACE',
                points,
                classification,
                confidence: calculateConfidence(points),
                metadata: {
                    pointCount: points.length,
                    averageIntensity: points.reduce((sum, p) => sum + p.intensity, 0) / points.length
                },
                lastUpdated: Date.now(),
                boundingBox: new THREE.Box3().setFromPoints(
                    points.map(p => new THREE.Vector3(p.x, p.y, p.z))
                )
            });
        }
    });

    return classifiedFeatures.filter(f => f.confidence >= MIN_FEATURE_CONFIDENCE);
}

/**
 * Calculates the 3D boundaries of the environment with safety margins
 * @param points Array of 3D points
 * @param safetyMargin Safety margin in meters
 * @returns 3D bounding box with safety margin
 */
export function calculateBoundaries(
    points: Point3D[],
    safetyMargin: number
): THREE.Box3 {
    if (!points.length) {
        throw new Error('No points provided for boundary calculation');
    }

    // Convert points to Vector3 for THREE.js processing
    const vectors = points.map(p => new THREE.Vector3(p.x, p.y, p.z));
    
    // Calculate initial bounding box
    const bounds = new THREE.Box3().setFromPoints(vectors);
    
    // Apply safety margin
    bounds.min.subScalar(safetyMargin);
    bounds.max.addScalar(safetyMargin);
    
    // Validate environment size
    const size = bounds.getSize(new THREE.Vector3());
    const area = size.x * size.z;
    
    if (area > MAX_ENVIRONMENT_SIZE) {
        throw new Error(`Environment size ${area}m² exceeds maximum of ${MAX_ENVIRONMENT_SIZE}m²`);
    }
    
    return bounds;
}

// Helper function to calculate feature confidence based on point density and distribution
function calculateConfidence(points: Point3D[]): number {
    if (points.length < 10) return 0;
    
    // Calculate point density
    const boundingBox = new THREE.Box3().setFromPoints(
        points.map(p => new THREE.Vector3(p.x, p.y, p.z))
    );
    const volume = boundingBox.getSize(new THREE.Vector3()).length();
    const density = points.length / volume;
    
    // Calculate point distribution uniformity
    const centroid = points.reduce(
        (acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y, z: acc.z + p.z }),
        { x: 0, y: 0, z: 0 }
    );
    centroid.x /= points.length;
    centroid.y /= points.length;
    centroid.z /= points.length;
    
    const avgDistance = points.reduce((sum, p) => 
        sum + Math.sqrt(
            Math.pow(p.x - centroid.x, 2) +
            Math.pow(p.y - centroid.y, 2) +
            Math.pow(p.z - centroid.z, 2)
        ), 0) / points.length;
    
    // Combine metrics for final confidence score
    return Math.min(1.0, (density * 0.6 + (1 - avgDistance) * 0.4));
}