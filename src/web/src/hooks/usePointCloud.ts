import { useState, useEffect, useCallback, useRef } from 'react'; // v18.2.0
import { useDispatch, useSelector } from 'react-redux'; // v8.0.5
import * as THREE from 'three'; // v0.150.0

import { Point3D, PointCloud, ScanParameters, ScanQuality } from '../../types/lidar.types';
import { startScan, stopScan, getPointCloud } from '../../api/lidarApi';
import {
    startScanning,
    stopScanning,
    selectScanResult,
    selectIsScanning,
    selectScanQuality,
} from '../../store/slices/lidarSlice';
import { LIDAR_CONSTANTS } from '../../config/constants';

// Constants based on technical specifications
const SCAN_INTERVAL = 1000 / LIDAR_CONSTANTS.SCAN_RATE_HZ; // 33.33ms for 30Hz
const DEFAULT_SCAN_PARAMETERS: ScanParameters = {
    resolution: LIDAR_CONSTANTS.MIN_RESOLUTION_CM, // 0.01cm
    range: LIDAR_CONSTANTS.MAX_RANGE_M, // 5.0m
    scanRate: LIDAR_CONSTANTS.SCAN_RATE_HZ // 30Hz
};

// Performance monitoring thresholds
const MAX_RETRY_ATTEMPTS = 3;
const MEMORY_THRESHOLD_MB = 512;
const PROCESSING_TIME_THRESHOLD_MS = 16;

interface PerformanceMetrics {
    fps: number;
    memoryUsage: number;
    processingTime: number;
    droppedFrames: number;
}

export function usePointCloud(parameters: ScanParameters = DEFAULT_SCAN_PARAMETERS) {
    const dispatch = useDispatch();
    const isScanning = useSelector(selectIsScanning);
    const scanResult = useSelector(selectScanResult);
    
    const [pointCloud, setPointCloud] = useState<PointCloud | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [metrics, setMetrics] = useState<PerformanceMetrics>({
        fps: 0,
        memoryUsage: 0,
        processingTime: 0,
        droppedFrames: 0
    });

    // Refs for performance tracking
    const frameCountRef = useRef(0);
    const lastFrameTimeRef = useRef(Date.now());
    const scanIntervalRef = useRef<NodeJS.Timeout>();
    const retryAttemptsRef = useRef(0);

    // WebGL context and buffers
    const glContextRef = useRef<WebGLRenderingContext | null>(null);
    const pointBufferRef = useRef<THREE.BufferGeometry | null>(null);

    // Initialize WebGL context and verify hardware support
    useEffect(() => {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl');
        
        if (!gl) {
            setError('WebGL not supported');
            return;
        }

        const extensions = [
            'OES_texture_float',
            'WEBGL_depth_texture'
        ];

        const missingExtensions = extensions.filter(ext => !gl.getExtension(ext));
        if (missingExtensions.length > 0) {
            setError(`Required WebGL extensions not supported: ${missingExtensions.join(', ')}`);
            return;
        }

        glContextRef.current = gl;
        pointBufferRef.current = new THREE.BufferGeometry();

        return () => {
            if (pointBufferRef.current) {
                pointBufferRef.current.dispose();
            }
        };
    }, []);

    // Performance monitoring
    const updatePerformanceMetrics = useCallback(() => {
        const now = Date.now();
        const elapsed = now - lastFrameTimeRef.current;
        
        if (elapsed >= 1000) { // Update metrics every second
            const fps = (frameCountRef.current * 1000) / elapsed;
            const memoryUsage = (performance as any).memory?.usedJSHeapSize / (1024 * 1024) || 0;
            
            setMetrics(prev => ({
                ...prev,
                fps,
                memoryUsage,
                droppedFrames: Math.max(0, (LIDAR_CONSTANTS.SCAN_RATE_HZ - fps))
            }));

            frameCountRef.current = 0;
            lastFrameTimeRef.current = now;
        }
    }, []);

    // Point cloud processing
    const processPointCloud = useCallback(async () => {
        try {
            const startTime = performance.now();
            
            if (!scanResult?.pointCloud) {
                return;
            }

            // Optimize point cloud data
            const optimizedPoints = scanResult.pointCloud.points.filter(point => 
                point.intensity > LIDAR_CONSTANTS.MIN_FEATURE_CONFIDENCE
            );

            // Update buffer geometry
            if (pointBufferRef.current) {
                const positions = new Float32Array(optimizedPoints.length * 3);
                const intensities = new Float32Array(optimizedPoints.length);

                optimizedPoints.forEach((point, index) => {
                    positions[index * 3] = point.x;
                    positions[index * 3 + 1] = point.y;
                    positions[index * 3 + 2] = point.z;
                    intensities[index] = point.intensity;
                });

                pointBufferRef.current.setAttribute(
                    'position',
                    new THREE.BufferAttribute(positions, 3)
                );
                pointBufferRef.current.setAttribute(
                    'intensity',
                    new THREE.BufferAttribute(intensities, 1)
                );
            }

            const processingTime = performance.now() - startTime;
            setMetrics(prev => ({ ...prev, processingTime }));

            frameCountRef.current++;
            updatePerformanceMetrics();

            // Check performance thresholds
            if (processingTime > PROCESSING_TIME_THRESHOLD_MS) {
                console.warn(`Processing time exceeded threshold: ${processingTime.toFixed(2)}ms`);
            }

            const memoryUsage = (performance as any).memory?.usedJSHeapSize / (1024 * 1024) || 0;
            if (memoryUsage > MEMORY_THRESHOLD_MB) {
                console.warn(`Memory usage exceeded threshold: ${memoryUsage.toFixed(2)}MB`);
            }

        } catch (err) {
            console.error('Point cloud processing error:', err);
            handleError(err);
        }
    }, [scanResult, updatePerformanceMetrics]);

    // Error handling
    const handleError = useCallback((error: any) => {
        const errorMessage = error?.message || 'Unknown error occurred';
        setError(errorMessage);

        if (retryAttemptsRef.current < MAX_RETRY_ATTEMPTS) {
            retryAttemptsRef.current++;
            console.warn(`Retrying scan (${retryAttemptsRef.current}/${MAX_RETRY_ATTEMPTS})`);
            startScanning();
        } else {
            stopScanning();
            retryAttemptsRef.current = 0;
        }
    }, []);

    // Start scanning
    const startScanning = useCallback(async () => {
        try {
            const { scanId } = await startScan(parameters);
            dispatch(startScanning());
            retryAttemptsRef.current = 0;

            scanIntervalRef.current = setInterval(async () => {
                const pointCloudData = await getPointCloud(scanId);
                if (pointCloudData) {
                    setPointCloud(pointCloudData);
                    await processPointCloud();
                }
            }, SCAN_INTERVAL);

        } catch (err) {
            handleError(err);
        }
    }, [dispatch, parameters, processPointCloud, handleError]);

    // Stop scanning
    const stopScanning = useCallback(() => {
        if (scanIntervalRef.current) {
            clearInterval(scanIntervalRef.current);
        }
        dispatch(stopScanning());
    }, [dispatch]);

    // Cleanup
    useEffect(() => {
        return () => {
            if (scanIntervalRef.current) {
                clearInterval(scanIntervalRef.current);
            }
        };
    }, []);

    return {
        pointCloud,
        isScanning,
        startScanning,
        stopScanning,
        error,
        scanQuality: scanResult?.quality || null,
        processingTime: metrics.processingTime,
        memoryUsage: metrics.memoryUsage
    };
}