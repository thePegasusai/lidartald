import { useEffect, useCallback, useState, useRef } from 'react'; // v18.2.0
import { useSelector, useDispatch } from 'react-redux'; // v8.1.0
import * as THREE from 'three'; // v0.150.0
import { 
    EnvironmentMap, 
    EnvironmentFeature, 
    EnvironmentUpdate, 
    SurfaceClassification,
    Point3D 
} from '../types/environment.types';
import { 
    createEnvironmentMap, 
    updateEnvironment, 
    classifySurfaces, 
    calculateBoundaries 
} from '../utils/environmentMapping';
import { 
    selectCurrentMap, 
    selectScanProgress, 
    selectMemoryUsage 
} from '../store/slices/environmentSlice';
import { LIDAR_CONSTANTS, ENVIRONMENT_CONSTANTS } from '../config/constants';

// Constants for environment mapping configuration
const UPDATE_INTERVAL_MS = LIDAR_CONSTANTS.POINT_CLOUD_UPDATE_INTERVAL_MS;
const MIN_FEATURE_CONFIDENCE = LIDAR_CONSTANTS.MIN_FEATURE_CONFIDENCE;
const MAX_POINTS_PER_BATCH = 10000;
const MEMORY_CLEANUP_INTERVAL = 5000;

/**
 * Interface for memory usage statistics
 */
interface MemoryStats {
    totalPoints: number;
    activeFeatures: number;
    memoryUsageMB: number;
    lastCleanup: number;
}

/**
 * Custom hook for managing real-time environment mapping with LiDAR integration
 * Provides 30Hz scanning, GPU-accelerated processing, and memory optimization
 */
export function useEnvironmentMap(
    resolution: number = LIDAR_CONSTANTS.MIN_RESOLUTION_CM,
    autoStart: boolean = false,
    batchSize: number = MAX_POINTS_PER_BATCH
) {
    const dispatch = useDispatch();
    const currentMap = useSelector(selectCurrentMap);
    const scanProgress = useSelector(selectScanProgress);
    const memoryUsage = useSelector(selectMemoryUsage);

    // State management
    const [isScanning, setIsScanning] = useState<boolean>(false);
    const [processingStats, setProcessingStats] = useState<MemoryStats>({
        totalPoints: 0,
        activeFeatures: 0,
        memoryUsageMB: 0,
        lastCleanup: Date.now()
    });

    // Refs for cleanup and GPU context
    const scanIntervalRef = useRef<NodeJS.Timer>();
    const gpuContextRef = useRef<THREE.WebGLRenderer>();
    const pointBufferRef = useRef<Point3D[]>([]);

    /**
     * Initializes GPU context for accelerated processing
     */
    const initializeGPUContext = useCallback(() => {
        if (!gpuContextRef.current) {
            gpuContextRef.current = new THREE.WebGLRenderer({
                powerPreference: 'high-performance',
                precision: 'highp'
            });
        }
    }, []);

    /**
     * Handles real-time environment updates with batching and memory optimization
     */
    const handleEnvironmentUpdate = useCallback(async (
        update: EnvironmentUpdate,
        batchSize: number
    ): Promise<void> => {
        try {
            // Validate update data
            if (!update.newPoints.length || !update.mapId) {
                throw new Error('Invalid update data');
            }

            // Process points in batches
            for (let i = 0; i < update.newPoints.length; i += batchSize) {
                const batch = update.newPoints.slice(i, i + batchSize);
                
                // Update environment map
                await updateEnvironment(
                    currentMap!,
                    {
                        ...update,
                        newPoints: batch,
                        partialUpdate: i + batchSize < update.newPoints.length
                    },
                    gpuContextRef.current!
                );

                // Update processing stats
                setProcessingStats(prev => ({
                    ...prev,
                    totalPoints: prev.totalPoints + batch.length,
                    memoryUsageMB: Math.round(performance.memory?.usedJSHeapSize / 1024 / 1024 || 0)
                }));
            }

            // Classify surfaces after batch processing
            const features = await classifySurfaces(currentMap!, gpuContextRef.current!);
            
            // Update feature stats
            setProcessingStats(prev => ({
                ...prev,
                activeFeatures: features.length
            }));

        } catch (error) {
            console.error('[Environment Update Error]', error);
            throw error;
        }
    }, [currentMap]);

    /**
     * Starts continuous environment scanning
     */
    const startScan = useCallback(() => {
        if (isScanning) return;

        initializeGPUContext();
        setIsScanning(true);

        scanIntervalRef.current = setInterval(() => {
            if (pointBufferRef.current.length >= batchSize) {
                const update: EnvironmentUpdate = {
                    mapId: currentMap?.id || crypto.randomUUID(),
                    timestamp: Date.now(),
                    newPoints: pointBufferRef.current.slice(0, batchSize),
                    updatedFeatures: [],
                    removedFeatureIds: [],
                    sequenceNumber: Date.now(),
                    boundaryChanges: null,
                    partialUpdate: true
                };

                handleEnvironmentUpdate(update, batchSize);
                pointBufferRef.current = pointBufferRef.current.slice(batchSize);
            }
        }, UPDATE_INTERVAL_MS);
    }, [isScanning, batchSize, currentMap, handleEnvironmentUpdate, initializeGPUContext]);

    /**
     * Stops environment scanning and performs cleanup
     */
    const stopScan = useCallback(() => {
        if (!isScanning) return;

        if (scanIntervalRef.current) {
            clearInterval(scanIntervalRef.current);
        }
        setIsScanning(false);
    }, [isScanning]);

    /**
     * Resets environment map and clears buffers
     */
    const resetMap = useCallback(() => {
        pointBufferRef.current = [];
        setProcessingStats({
            totalPoints: 0,
            activeFeatures: 0,
            memoryUsageMB: 0,
            lastCleanup: Date.now()
        });
    }, []);

    /**
     * Optimizes memory usage by cleaning up unused resources
     */
    const optimizeMemory = useCallback(() => {
        const now = Date.now();
        if (now - processingStats.lastCleanup >= MEMORY_CLEANUP_INTERVAL) {
            // Clean up point buffer
            if (pointBufferRef.current.length > MAX_POINTS_PER_BATCH * 2) {
                pointBufferRef.current = pointBufferRef.current.slice(-MAX_POINTS_PER_BATCH);
            }

            // Update cleanup timestamp
            setProcessingStats(prev => ({
                ...prev,
                lastCleanup: now,
                memoryUsageMB: Math.round(performance.memory?.usedJSHeapSize / 1024 / 1024 || 0)
            }));
        }
    }, [processingStats.lastCleanup]);

    // Auto-start scanning if enabled
    useEffect(() => {
        if (autoStart) {
            startScan();
        }
        return () => stopScan();
    }, [autoStart, startScan, stopScan]);

    // Periodic memory optimization
    useEffect(() => {
        const cleanupInterval = setInterval(optimizeMemory, MEMORY_CLEANUP_INTERVAL);
        return () => clearInterval(cleanupInterval);
    }, [optimizeMemory]);

    // Cleanup GPU context on unmount
    useEffect(() => {
        return () => {
            if (gpuContextRef.current) {
                gpuContextRef.current.dispose();
            }
        };
    }, []);

    return {
        currentMap,
        scanProgress,
        isScanning,
        memoryUsage: processingStats,
        startScan,
        stopScan,
        resetMap,
        optimizeMemory
    };
}