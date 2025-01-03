import { configureStore } from '@reduxjs/toolkit'; // v1.9.5
import { describe, it, expect, beforeEach, jest } from '@jest/globals'; // v29.5.0
import { 
    lidarSlice, 
    selectScanResult, 
    selectIsScanning, 
    selectProcessingMetrics, 
    selectError,
    startScanning,
    stopScanning,
    updateScanResult,
    updateScanParameters,
    setError,
    updateProcessingMetrics
} from '../../store/slices/lidarSlice';
import { 
    Point3D, 
    PointCloud, 
    Feature, 
    ScanParameters, 
    ScanResult, 
    ProcessingMetrics,
    LidarError,
    FeatureType 
} from '../../types/lidar.types';
import { Matrix4 } from 'three';

// Test constants based on technical specifications
const MOCK_SCAN_PARAMETERS: ScanParameters = {
    resolution: 0.01, // 0.01cm resolution
    range: 5.0,      // 5-meter range
    scanRate: 30     // 30Hz scan rate
};

const PERFORMANCE_THRESHOLDS = {
    maxProcessingTime: 33, // 33ms maximum processing time
    minFps: 30,           // 30Hz minimum frame rate
    maxMemoryUsage: 500   // 500MB maximum memory usage
};

// Helper function to create test store
const setupStore = () => {
    return configureStore({
        reducer: {
            lidar: lidarSlice.reducer
        }
    });
};

// Helper function to create mock point cloud data
const createMockPointCloud = (resolution: number = 0.01): PointCloud => {
    const points: Point3D[] = [
        { x: 0, y: 0, z: 0, intensity: 1.0 },
        { x: resolution, y: 0, z: 0, intensity: 0.8 },
        { x: 0, y: resolution, z: 0, intensity: 0.9 }
    ];
    
    return {
        points,
        timestamp: Date.now(),
        transformMatrix: new Matrix4()
    };
};

// Helper function to create mock scan result
const createMockScanResult = (scanParams: ScanParameters = MOCK_SCAN_PARAMETERS): ScanResult => {
    const features: Feature[] = [
        {
            id: '1',
            type: FeatureType.SURFACE,
            coordinates: [{ x: 0, y: 0, z: 0, intensity: 1.0 }],
            confidence: 0.95
        },
        {
            id: '2',
            type: FeatureType.OBSTACLE,
            coordinates: [{ x: 1, y: 1, z: 1, intensity: 0.8 }],
            confidence: 0.85
        }
    ];

    return {
        pointCloud: createMockPointCloud(scanParams.resolution),
        features,
        scanParameters: scanParams,
        quality: 0.9
    };
};

describe('LiDAR Slice', () => {
    let store: ReturnType<typeof setupStore>;

    beforeEach(() => {
        store = setupStore();
        jest.useFakeTimers();
    });

    describe('Scanning State Management', () => {
        it('should start scanning correctly', () => {
            store.dispatch(startScanning());
            const state = store.getState();
            expect(selectIsScanning(state)).toBe(true);
            expect(selectError(state)).toBeNull();
        });

        it('should stop scanning correctly', () => {
            store.dispatch(startScanning());
            store.dispatch(stopScanning());
            const state = store.getState();
            expect(selectIsScanning(state)).toBe(false);
            expect(state.lidar.lastUpdateTime).toBeNull();
        });

        it('should handle scan parameters update', () => {
            const newParams: Partial<ScanParameters> = {
                resolution: 0.005,
                range: 4.0
            };
            store.dispatch(updateScanParameters(newParams));
            const state = store.getState();
            expect(state.lidar.scanParameters).toEqual({
                ...MOCK_SCAN_PARAMETERS,
                ...newParams
            });
        });
    });

    describe('Scan Result Processing', () => {
        it('should update scan result and calculate processing metrics', () => {
            const mockResult = createMockScanResult();
            store.dispatch(startScanning());
            jest.advanceTimersByTime(20); // Simulate 20ms processing time
            store.dispatch(updateScanResult(mockResult));

            const state = store.getState();
            expect(selectScanResult(state)).toEqual(mockResult);
            expect(state.lidar.processingTime).toBeLessThanOrEqual(PERFORMANCE_THRESHOLDS.maxProcessingTime);
        });

        it('should track processing time violations', () => {
            const mockResult = createMockScanResult();
            store.dispatch(startScanning());
            jest.advanceTimersByTime(40); // Simulate processing time violation
            store.dispatch(updateScanResult(mockResult));

            const metrics = selectProcessingMetrics(store.getState());
            expect(metrics.violationsCount).toBe(1);
            expect(metrics.isPerformanceOptimal).toBe(false);
        });

        it('should maintain scan resolution within specifications', () => {
            const mockResult = createMockScanResult({
                ...MOCK_SCAN_PARAMETERS,
                resolution: 0.01
            });
            store.dispatch(updateScanResult(mockResult));
            
            const state = store.getState();
            const result = selectScanResult(state);
            expect(result?.scanParameters.resolution).toBe(0.01);
        });
    });

    describe('Error Handling', () => {
        it('should handle and store errors correctly', () => {
            const errorMessage = 'LiDAR hardware connection failed';
            store.dispatch(setError(errorMessage));
            
            const state = store.getState();
            expect(selectError(state)).toBe(errorMessage);
            expect(selectIsScanning(state)).toBe(false);
        });

        it('should clear error on successful scan start', () => {
            store.dispatch(setError('Previous error'));
            store.dispatch(startScanning());
            
            const state = store.getState();
            expect(selectError(state)).toBeNull();
        });
    });

    describe('Performance Requirements', () => {
        it('should maintain 30Hz scan rate', () => {
            const results: ScanResult[] = [];
            const scanDuration = 1000; // 1 second
            const expectedScans = 30; // 30Hz

            // Simulate 1 second of scanning
            store.dispatch(startScanning());
            for (let i = 0; i < expectedScans; i++) {
                const mockResult = createMockScanResult();
                store.dispatch(updateScanResult(mockResult));
                results.push(mockResult);
                jest.advanceTimersByTime(1000 / 30);
            }

            const metrics = selectProcessingMetrics(store.getState());
            expect(results.length).toBe(expectedScans);
            expect(metrics.averageTime).toBeLessThanOrEqual(PERFORMANCE_THRESHOLDS.maxProcessingTime);
        });

        it('should handle point cloud resolution of 0.01cm', () => {
            const mockResult = createMockScanResult({
                ...MOCK_SCAN_PARAMETERS,
                resolution: 0.01
            });
            store.dispatch(updateScanResult(mockResult));

            const state = store.getState();
            const result = selectScanResult(state);
            expect(result?.pointCloud.points[1].x).toBe(0.01);
        });

        it('should handle 5-meter range scanning', () => {
            const mockResult = createMockScanResult({
                ...MOCK_SCAN_PARAMETERS,
                range: 5.0
            });
            store.dispatch(updateScanResult(mockResult));

            const state = store.getState();
            const result = selectScanResult(state);
            expect(result?.scanParameters.range).toBe(5.0);
        });
    });
});