import { createSlice, PayloadAction } from '@reduxjs/toolkit'; // v1.9.5
import { Point3D, PointCloud, Feature, ScanParameters, ScanResult } from '../../types/lidar.types';

// Performance thresholds based on technical specifications
const PROCESSING_TIME_THRESHOLD = 33; // 33ms maximum processing time
const DEFAULT_SCAN_PARAMETERS = {
    resolution: 0.01, // 0.01cm resolution
    range: 5.0,      // 5-meter range
    scanRate: 30     // 30Hz scan rate
};

interface ProcessingMetrics {
    averageTime: number;
    maxTime: number;
    violationsCount: number;
    lastViolationTime: number | null;
}

interface LidarState {
    isScanning: boolean;
    scanResult: ScanResult | null;
    scanParameters: ScanParameters;
    lastUpdateTime: number | null;
    processingTime: number | null;
    error: string | null;
    processingMetrics: ProcessingMetrics;
}

const initialState: LidarState = {
    isScanning: false,
    scanResult: null,
    scanParameters: DEFAULT_SCAN_PARAMETERS,
    lastUpdateTime: null,
    processingTime: null,
    error: null,
    processingMetrics: {
        averageTime: 0,
        maxTime: 0,
        violationsCount: 0,
        lastViolationTime: null
    }
};

export const lidarSlice = createSlice({
    name: 'lidar',
    initialState,
    reducers: {
        startScanning: (state) => {
            state.isScanning = true;
            state.error = null;
            state.lastUpdateTime = Date.now();
        },
        
        stopScanning: (state) => {
            state.isScanning = false;
            state.lastUpdateTime = null;
            state.processingTime = null;
        },
        
        updateScanResult: (state, action: PayloadAction<ScanResult>) => {
            const currentTime = Date.now();
            const processingTime = state.lastUpdateTime ? 
                currentTime - state.lastUpdateTime : 0;
                
            // Update scan result
            state.scanResult = action.payload;
            state.lastUpdateTime = currentTime;
            state.processingTime = processingTime;
            
            // Update processing metrics
            if (processingTime > 0) {
                const metrics = state.processingMetrics;
                metrics.averageTime = (metrics.averageTime + processingTime) / 2;
                metrics.maxTime = Math.max(metrics.maxTime, processingTime);
                
                if (processingTime > PROCESSING_TIME_THRESHOLD) {
                    metrics.violationsCount++;
                    metrics.lastViolationTime = currentTime;
                }
            }
        },
        
        updateScanParameters: (state, action: PayloadAction<Partial<ScanParameters>>) => {
            state.scanParameters = {
                ...state.scanParameters,
                ...action.payload
            };
        },
        
        setError: (state, action: PayloadAction<string>) => {
            state.error = action.payload;
            state.isScanning = false;
        },
        
        resetProcessingMetrics: (state) => {
            state.processingMetrics = {
                averageTime: 0,
                maxTime: 0,
                violationsCount: 0,
                lastViolationTime: null
            };
        }
    }
});

// Selectors
export const selectScanResult = (state: { lidar: LidarState }): ScanResult | null => 
    state.lidar.scanResult;

export const selectIsScanning = (state: { lidar: LidarState }): boolean => 
    state.lidar.isScanning;

export const selectProcessingMetrics = (state: { lidar: LidarState }): {
    averageTime: number;
    maxTime: number;
    violationsCount: number;
    lastViolationTime: number | null;
    isPerformanceOptimal: boolean;
} => {
    const metrics = state.lidar.processingMetrics;
    return {
        ...metrics,
        isPerformanceOptimal: metrics.averageTime <= PROCESSING_TIME_THRESHOLD
    };
};

export const selectScanParameters = (state: { lidar: LidarState }): ScanParameters =>
    state.lidar.scanParameters;

export const selectError = (state: { lidar: LidarState }): string | null =>
    state.lidar.error;

// Export actions
export const {
    startScanning,
    stopScanning,
    updateScanResult,
    updateScanParameters,
    setError,
    resetProcessingMetrics
} = lidarSlice.actions;

// Export reducer
export default lidarSlice.reducer;