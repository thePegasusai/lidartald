import { configureStore, Middleware } from '@reduxjs/toolkit'; // v1.9.5
import { thunk } from 'redux-thunk'; // v2.4.2
import environmentReducer from './slices/environmentSlice';
import fleetReducer from './slices/fleetSlice';
import createWebSocketMiddleware from './middleware/websocketMiddleware';
import { websocketClient } from '../api/websocketApi';
import { FLEET_CONSTANTS, UI_CONSTANTS } from '../config/constants';

// Performance monitoring constants
const PERFORMANCE_SAMPLE_SIZE = 100;
const PERFORMANCE_WARNING_THRESHOLD_MS = 16; // ~60 FPS target
const NETWORK_LATENCY_THRESHOLD_MS = 50;

// Batch processing configuration for optimal performance
const BATCH_CONFIG = {
    maxBatchSize: MAX_BATCH_SIZE,
    batchTimeoutMs: SYNC_INTERVAL,
    flushThresholdMs: NETWORK_LATENCY_THRESHOLD_MS
};

// Performance monitoring metrics
interface PerformanceMetrics {
    timestamp: number;
    duration: number;
    actionType: string;
}

let performanceMetrics: PerformanceMetrics[] = [];

// Performance monitoring middleware
const performanceMiddleware: Middleware = store => next => action => {
    const start = performance.now();
    const result = next(action);
    const duration = performance.now() - start;

    // Track performance metrics
    performanceMetrics.push({
        timestamp: Date.now(),
        duration,
        actionType: action.type
    });

    // Maintain rolling window of metrics
    if (performanceMetrics.length > PERFORMANCE_SAMPLE_SIZE) {
        performanceMetrics.shift();
    }

    // Check for performance degradation
    if (duration > PERFORMANCE_WARNING_THRESHOLD_MS) {
        console.warn(`Performance warning: Action ${action.type} took ${duration.toFixed(2)}ms`);
    }

    return result;
};

// State serialization middleware for optimized network transfer
const serializationMiddleware: Middleware = () => next => action => {
    if (action.payload && typeof action.payload === 'object') {
        // Optimize payload size for network transmission
        action.payload = JSON.parse(JSON.stringify(action.payload));
    }
    return next(action);
};

// Configure WebSocket middleware with optimized settings
const wsMiddleware = createWebSocketMiddleware(websocketClient, {
    reconnectInterval: FLEET_CONSTANTS.SYNC_INTERVAL_MS,
    maxReconnectAttempts: 5,
    batchConfig: BATCH_CONFIG
});

// Configure Redux store with performance optimizations
export const store = configureStore({
    reducer: {
        environment: environmentReducer,
        fleet: fleetReducer
    },
    middleware: (getDefaultMiddleware) => getDefaultMiddleware({
        serializableCheck: {
            // Ignore WebSocket and Three.js non-serializable values
            ignoredActions: ['ws/message', 'environment/updatePointCloud'],
            ignoredPaths: ['environment.pointCloud', 'fleet.meshNetwork']
        },
        thunk: {
            extraArgument: {
                wsClient: websocketClient
            }
        }
    }).concat([
        thunk,
        wsMiddleware,
        performanceMiddleware,
        serializationMiddleware
    ]),
    devTools: REDUX_DEVTOOLS && {
        maxAge: 50,
        latency: 500,
        trace: true
    }
});

// Export type-safe hooks
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Performance monitoring utilities
export const getPerformanceMetrics = () => {
    const averageDuration = performanceMetrics.reduce(
        (sum, metric) => sum + metric.duration, 
        0
    ) / performanceMetrics.length;

    return {
        averageActionDuration: averageDuration,
        maxActionDuration: Math.max(...performanceMetrics.map(m => m.duration)),
        totalActions: performanceMetrics.length,
        isPerformant: averageDuration < PERFORMANCE_WARNING_THRESHOLD_MS
    };
};

// State selectors with memoization for optimal performance
export const selectEnvironmentState = (state: RootState) => state.environment;
export const selectFleetState = (state: RootState) => state.fleet;

// Export configured store instance
export default store;