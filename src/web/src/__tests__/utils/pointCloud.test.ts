import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals'; // v29.5.0
import { Matrix4, Vector3, Box3 } from 'three'; // v0.150.0
import { 
    processPointCloud, 
    detectFeatures, 
    mergePointClouds, 
    transformPointCloud 
} from '../../utils/pointCloud';
import { 
    Point3D, 
    PointCloud, 
    Feature, 
    FeatureType, 
    ScanParameters 
} from '../../types/lidar.types';

// Test constants
const PERFORMANCE_THRESHOLD = 33; // 33ms max processing time (30Hz)
const RESOLUTION_PRECISION = 0.01; // 0.01cm resolution
const SCAN_RANGE = 5.0; // 5-meter range

// Test fixtures
const createTestPoint = (x: number, y: number, z: number): Point3D => ({
    x, y, z, intensity: 1.0
});

const createTestPointCloud = (numPoints: number = 1000): PointCloud => ({
    points: Array.from({ length: numPoints }, (_, i) => 
        createTestPoint(
            Math.random() * SCAN_RANGE,
            Math.random() * SCAN_RANGE,
            Math.random() * SCAN_RANGE
        )
    ),
    timestamp: Date.now(),
    transformMatrix: new Matrix4()
});

const TEST_SCAN_PARAMETERS: ScanParameters = {
    resolution: RESOLUTION_PRECISION,
    range: SCAN_RANGE,
    scanRate: 30
};

describe('processPointCloud', () => {
    let performanceNow: jest.SpyInstance;
    let testCloud: PointCloud;

    beforeEach(() => {
        performanceNow = jest.spyOn(performance, 'now');
        testCloud = createTestPointCloud();
    });

    afterEach(() => {
        performanceNow.mockRestore();
    });

    it('should process point cloud within 33ms performance threshold', async () => {
        const startTime = performance.now();
        await processPointCloud(testCloud, TEST_SCAN_PARAMETERS);
        const processingTime = performance.now() - startTime;

        expect(processingTime).toBeLessThanOrEqual(PERFORMANCE_THRESHOLD);
    });

    it('should maintain 0.01cm resolution precision', async () => {
        const processed = await processPointCloud(testCloud, TEST_SCAN_PARAMETERS);

        processed.points.forEach(point => {
            const xPrecision = point.x.toString().split('.')[1]?.length || 0;
            const yPrecision = point.y.toString().split('.')[1]?.length || 0;
            const zPrecision = point.z.toString().split('.')[1]?.length || 0;

            expect(xPrecision).toBeGreaterThanOrEqual(2);
            expect(yPrecision).toBeGreaterThanOrEqual(2);
            expect(zPrecision).toBeGreaterThanOrEqual(2);
        });
    });

    it('should filter out low intensity noise', async () => {
        const noisyCloud: PointCloud = {
            ...testCloud,
            points: [...testCloud.points, createTestPoint(1, 1, 1)].map(p => ({ ...p, intensity: 0.05 }))
        };

        const processed = await processPointCloud(noisyCloud, TEST_SCAN_PARAMETERS);
        expect(processed.points.length).toBeLessThan(noisyCloud.points.length);
    });
});

describe('detectFeatures', () => {
    let testCloud: PointCloud;

    beforeEach(() => {
        testCloud = createTestPointCloud();
    });

    it('should detect features within performance threshold', async () => {
        const startTime = performance.now();
        await detectFeatures(testCloud);
        const processingTime = performance.now() - startTime;

        expect(processingTime).toBeLessThanOrEqual(PERFORMANCE_THRESHOLD);
    });

    it('should classify features with high confidence', async () => {
        const features = await detectFeatures(testCloud);
        
        features.forEach(feature => {
            expect(feature.confidence).toBeGreaterThanOrEqual(0.8);
            expect(Object.values(FeatureType)).toContain(feature.type);
        });
    });

    it('should detect surface features accurately', async () => {
        const surfaceCloud: PointCloud = {
            ...testCloud,
            points: Array.from({ length: 100 }, (_, i) => 
                createTestPoint(i * 0.01, 0, 0)
            )
        };

        const features = await detectFeatures(surfaceCloud);
        const surfaceFeatures = features.filter(f => f.type === FeatureType.SURFACE);

        expect(surfaceFeatures.length).toBeGreaterThan(0);
        expect(surfaceFeatures[0].confidence).toBeGreaterThanOrEqual(0.9);
    });
});

describe('mergePointClouds', () => {
    let testClouds: PointCloud[];

    beforeEach(() => {
        testClouds = [
            createTestPointCloud(500),
            createTestPointCloud(500)
        ];
    });

    it('should merge clouds within performance threshold', () => {
        const startTime = performance.now();
        mergePointClouds(testClouds);
        const processingTime = performance.now() - startTime;

        expect(processingTime).toBeLessThanOrEqual(PERFORMANCE_THRESHOLD);
    });

    it('should remove duplicate points within resolution threshold', () => {
        const duplicatePoint = createTestPoint(1, 1, 1);
        const cloudWithDuplicates: PointCloud[] = [
            { ...testClouds[0], points: [...testClouds[0].points, duplicatePoint] },
            { ...testClouds[1], points: [...testClouds[1].points, duplicatePoint] }
        ];

        const merged = mergePointClouds(cloudWithDuplicates);
        const duplicateCount = merged.points.filter(p => 
            Math.abs(p.x - duplicatePoint.x) < RESOLUTION_PRECISION &&
            Math.abs(p.y - duplicatePoint.y) < RESOLUTION_PRECISION &&
            Math.abs(p.z - duplicatePoint.z) < RESOLUTION_PRECISION
        ).length;

        expect(duplicateCount).toBe(1);
    });

    it('should preserve transformation matrices', () => {
        const transform = new Matrix4().makeRotationX(Math.PI / 2);
        testClouds[0].transformMatrix = transform;

        const merged = mergePointClouds(testClouds);
        expect(merged.transformMatrix.elements).toEqual(transform.elements);
    });
});

describe('transformPointCloud', () => {
    let testCloud: PointCloud;
    let transformMatrix: Matrix4;

    beforeEach(() => {
        testCloud = createTestPointCloud();
        transformMatrix = new Matrix4().makeRotationY(Math.PI / 2);
    });

    it('should transform cloud within performance threshold', () => {
        const startTime = performance.now();
        transformPointCloud(testCloud, transformMatrix);
        const processingTime = performance.now() - startTime;

        expect(processingTime).toBeLessThanOrEqual(PERFORMANCE_THRESHOLD);
    });

    it('should maintain point cloud resolution after transformation', () => {
        const transformed = transformPointCloud(testCloud, transformMatrix);

        transformed.points.forEach(point => {
            const xPrecision = point.x.toString().split('.')[1]?.length || 0;
            const yPrecision = point.y.toString().split('.')[1]?.length || 0;
            const zPrecision = point.z.toString().split('.')[1]?.length || 0;

            expect(xPrecision).toBeGreaterThanOrEqual(2);
            expect(yPrecision).toBeGreaterThanOrEqual(2);
            expect(zPrecision).toBeGreaterThanOrEqual(2);
        });
    });

    it('should correctly compose transformation matrices', () => {
        const rotation = new Matrix4().makeRotationX(Math.PI / 4);
        const translation = new Matrix4().makeTranslation(1, 2, 3);
        
        const transformed1 = transformPointCloud(testCloud, rotation);
        const transformed2 = transformPointCloud(transformed1, translation);

        const expectedMatrix = new Matrix4()
            .multiply(translation)
            .multiply(rotation)
            .multiply(testCloud.transformMatrix);

        expect(transformed2.transformMatrix.elements)
            .toEqual(expectedMatrix.elements);
    });

    it('should preserve point cloud boundaries', () => {
        const bounds = new Box3().setFromPoints(
            testCloud.points.map(p => new Vector3(p.x, p.y, p.z))
        );
        
        const transformed = transformPointCloud(testCloud, transformMatrix);
        const transformedBounds = new Box3().setFromPoints(
            transformed.points.map(p => new Vector3(p.x, p.y, p.z))
        );

        expect(transformedBounds.getSize(new Vector3()).length())
            .toBeCloseTo(bounds.getSize(new Vector3()).length(), 5);
    });
});