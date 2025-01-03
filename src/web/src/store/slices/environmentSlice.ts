import { createSlice, PayloadAction } from '@reduxjs/toolkit'; // v1.9.5
import { EnvironmentMap, EnvironmentFeature } from '../types/environment.types';
import { Point3D } from '../types/lidar.types';
import { environmentApi } from '../../api/environmentApi';

// Constants for environment processing
const UPDATE_INTERVAL_MS = 33; // 30Hz update rate
const MAX_POINTS_PER_UPDATE = 1000;
const MIN_FEATURE_CONFIDENCE = 0.85;
const RETRY_ATTEMPTS = 3;
const BATCH_TIMEOUT_MS = 16;
const MEMORY_LIMIT_MB = 512;

// Interface for environment state
interface EnvironmentState {
    activeMap: EnvironmentMap | null;
    isProcessing: boolean;
    error: string | null;
    lastUpdateTime: number;
    processingMetrics: {
        pointCount: number;
        featureCount: number;
        memoryUsage: number;
        updateLatency: number;
    };
    fleetSyncStatus: {
        isSyncing: boolean;
        lastSyncTime: number;
        pendingUpdates: number;
    };
}

// Initial state
const initialState: EnvironmentState = {
    activeMap: null,
    isProcessing: false,
    error: null,
    lastUpdateTime: 0,
    processingMetrics: {
        pointCount: 0,
        featureCount: 0,
        memoryUsage: 0,
        updateLatency: 0
    },
    fleetSyncStatus: {
        isSyncing: false,
        lastSyncTime: 0,
        pendingUpdates: 0
    }
};

// Environment slice
const environmentSlice = createSlice({
    name: 'environment',
    initialState,
    reducers: {
        setActiveMap: (state, action: PayloadAction<EnvironmentMap>) => {
            state.activeMap = action.payload;
            state.lastUpdateTime = Date.now();
        },
        clearEnvironment: (state) => {
            state.activeMap = null;
            state.isProcessing = false;
            state.error = null;
            state.processingMetrics = initialState.processingMetrics;
        },
        updateProcessingMetrics: (state, action: PayloadAction<Partial<typeof initialState.processingMetrics>>) => {
            state.processingMetrics = {
                ...state.processingMetrics,
                ...action.payload
            };
        },
        setFleetSyncStatus: (state, action: PayloadAction<Partial<typeof initialState.fleetSyncStatus>>) => {
            state.fleetSyncStatus = {
                ...state.fleetSyncStatus,
                ...action.payload
            };
        },
        setError: (state, action: PayloadAction<string>) => {
            state.error = action.payload;
            state.isProcessing = false;
        }
    },
    extraReducers: (builder) => {
        builder
            .addCase(initializeEnvironment.pending, (state) => {
                state.isProcessing = true;
                state.error = null;
            })
            .addCase(initializeEnvironment.fulfilled, (state, action) => {
                state.activeMap = action.payload;
                state.isProcessing = false;
                state.lastUpdateTime = Date.now();
            })
            .addCase(initializeEnvironment.rejected, (state, action) => {
                state.isProcessing = false;
                state.error = action.error.message || 'Failed to initialize environment';
            })
            .addCase(updateEnvironment.pending, (state) => {
                state.isProcessing = true;
            })
            .addCase(updateEnvironment.fulfilled, (state, action) => {
                if (state.activeMap) {
                    state.activeMap = {
                        ...state.activeMap,
                        ...action.payload
                    };
                    state.lastUpdateTime = Date.now();
                }
                state.isProcessing = false;
            })
            .addCase(updateEnvironment.rejected, (state, action) => {
                state.isProcessing = false;
                state.error = action.error.message || 'Failed to update environment';
            });
    }
});

// Async thunks
export const initializeEnvironment = createAsyncThunk(
    'environment/initialize',
    async ({ points, resolution }: { points: Point3D[], resolution: number }) => {
        try {
            const map = await environmentApi.createEnvironmentMap(points, resolution);
            return map;
        } catch (error) {
            console.error('[Environment Initialization Error]', error);
            throw error;
        }
    }
);

export const updateEnvironment = createAsyncThunk(
    'environment/update',
    async ({ points, mapId }: { points: Point3D[], mapId: string }, { getState, dispatch }) => {
        const startTime = performance.now();
        const batchedPoints: Point3D[][] = [];

        // Batch points for optimal processing
        for (let i = 0; i < points.length; i += MAX_POINTS_PER_UPDATE) {
            batchedPoints.push(points.slice(i, i + MAX_POINTS_PER_UPDATE));
        }

        try {
            for (const batch of batchedPoints) {
                await environmentApi.updateEnvironmentMap({
                    mapId,
                    newPoints: batch,
                    timestamp: Date.now(),
                    sequenceNumber: Date.now(),
                    partialUpdate: true
                });

                // Update processing metrics
                dispatch(updateProcessingMetrics({
                    pointCount: batch.length,
                    updateLatency: performance.now() - startTime,
                    memoryUsage: calculateMemoryUsage()
                }));

                // Throttle updates to maintain 30Hz
                await new Promise(resolve => setTimeout(resolve, UPDATE_INTERVAL_MS));
            }

            // Validate and filter features
            const features = await environmentApi.getEnvironmentFeatures(mapId);
            const validatedFeatures = filterFeatures(features);

            return {
                points,
                features: validatedFeatures,
                lastUpdateTime: Date.now()
            };
        } catch (error) {
            console.error('[Environment Update Error]', error);
            throw error;
        }
    }
);

// Helper functions
const filterFeatures = (features: EnvironmentFeature[]): EnvironmentFeature[] => {
    return features
        .filter(feature => feature.confidence >= MIN_FEATURE_CONFIDENCE)
        .sort((a, b) => b.confidence - a.confidence);
};

const calculateMemoryUsage = (): number => {
    if (performance.memory) {
        return Math.round(performance.memory.usedJSHeapSize / (1024 * 1024));
    }
    return 0;
};

// Export actions and reducer
export const {
    setActiveMap,
    clearEnvironment,
    updateProcessingMetrics,
    setFleetSyncStatus,
    setError
} = environmentSlice.actions;

export default environmentSlice.reducer;