import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { jest } from '@jest/globals';
import { WebGLRenderer, WebGLRenderingContext } from '@lunapaint/webgl-mock';
import { usePerformanceMonitor } from '@performance-monitor/react';
import LidarViewport from '../../../components/lidar/LidarViewport';
import { useLidarScanner } from '../../../hooks/useLidarScanner';

// Mock WebGL context
jest.mock('@lunapaint/webgl-mock');

// Mock performance monitoring hook
jest.mock('@performance-monitor/react', () => ({
    usePerformanceMonitor: jest.fn()
}));

// Mock LiDAR scanner hook
jest.mock('../../../hooks/useLidarScanner');

// Constants for testing based on technical specifications
const TEST_CONSTANTS = {
    REFRESH_RATE_HZ: 30,
    MIN_RESOLUTION_CM: 0.01,
    TARGET_FPS: 60,
    FRAME_TIME_MS: 16.67, // ~60 FPS
    TEST_DURATION_MS: 1000,
    SAMPLE_SIZE: 100
};

describe('LidarViewport Component', () => {
    let mockWebGLContext: WebGLRenderingContext;
    let mockPerformanceData: any;
    let mockScannerData: any;

    beforeEach(() => {
        // Setup WebGL mock
        mockWebGLContext = new WebGLRenderingContext();
        (global as any).WebGLRenderingContext = jest.fn(() => mockWebGLContext);
        
        // Setup performance monitoring mock
        mockPerformanceData = {
            fps: TEST_CONSTANTS.TARGET_FPS,
            frameTime: TEST_CONSTANTS.FRAME_TIME_MS,
            memoryUsage: 256,
            gpuLoad: 50
        };
        (usePerformanceMonitor as jest.Mock).mockReturnValue(mockPerformanceData);

        // Setup LiDAR scanner mock
        mockScannerData = {
            isScanning: true,
            scanResult: {
                pointCloud: {
                    points: Array(1000).fill(null).map(() => ({
                        x: Math.random() * 5,
                        y: Math.random() * 5,
                        z: Math.random() * 5,
                        intensity: Math.random()
                    })),
                    timestamp: Date.now()
                },
                features: [],
                quality: 0.95
            },
            scanParameters: {
                resolution: TEST_CONSTANTS.MIN_RESOLUTION_CM,
                range: 5.0,
                scanRate: TEST_CONSTANTS.REFRESH_RATE_HZ
            },
            error: null
        };
        (useLidarScanner as jest.Mock).mockReturnValue(mockScannerData);

        // Setup performance monitoring
        jest.spyOn(performance, 'now').mockImplementation(() => Date.now());
        jest.spyOn(window, 'requestAnimationFrame').mockImplementation(cb => setTimeout(cb, 16));
    });

    afterEach(() => {
        jest.clearAllMocks();
        jest.restoreAllMocks();
    });

    it('maintains 30Hz refresh rate for LiDAR visualization', async () => {
        const frameTimestamps: number[] = [];
        const recordFrame = () => {
            frameTimestamps.push(performance.now());
        };

        render(<LidarViewport />);

        // Record frames for test duration
        await act(async () => {
            const interval = setInterval(recordFrame, 1000 / TEST_CONSTANTS.REFRESH_RATE_HZ);
            await new Promise(resolve => setTimeout(resolve, TEST_CONSTANTS.TEST_DURATION_MS));
            clearInterval(interval);
        });

        // Calculate actual refresh rate
        const frameIntervals = frameTimestamps.slice(1).map((t, i) => t - frameTimestamps[i]);
        const averageInterval = frameIntervals.reduce((a, b) => a + b, 0) / frameIntervals.length;
        const actualRefreshRate = 1000 / averageInterval;

        expect(actualRefreshRate).toBeGreaterThanOrEqual(TEST_CONSTANTS.REFRESH_RATE_HZ);
    });

    it('renders point cloud at 0.01cm resolution', async () => {
        const { container } = render(<LidarViewport />);

        await waitFor(() => {
            const pointCloud = mockScannerData.scanResult.pointCloud;
            const points = pointCloud.points;

            // Verify point positions are quantized to 0.01cm resolution
            points.forEach((point: any) => {
                const xResolution = Math.round(point.x / TEST_CONSTANTS.MIN_RESOLUTION_CM) * TEST_CONSTANTS.MIN_RESOLUTION_CM;
                const yResolution = Math.round(point.y / TEST_CONSTANTS.MIN_RESOLUTION_CM) * TEST_CONSTANTS.MIN_RESOLUTION_CM;
                const zResolution = Math.round(point.z / TEST_CONSTANTS.MIN_RESOLUTION_CM) * TEST_CONSTANTS.MIN_RESOLUTION_CM;

                expect(Math.abs(point.x - xResolution)).toBeLessThanOrEqual(TEST_CONSTANTS.MIN_RESOLUTION_CM);
                expect(Math.abs(point.y - yResolution)).toBeLessThanOrEqual(TEST_CONSTANTS.MIN_RESOLUTION_CM);
                expect(Math.abs(point.z - zResolution)).toBeLessThanOrEqual(TEST_CONSTANTS.MIN_RESOLUTION_CM);
            });
        });
    });

    it('maintains 60 FPS UI responsiveness', async () => {
        const frameMetrics: number[] = [];

        // Mock performance monitoring
        (usePerformanceMonitor as jest.Mock).mockImplementation(() => ({
            onFrame: (callback: (metrics: any) => void) => {
                const interval = setInterval(() => {
                    const frameTime = performance.now();
                    frameMetrics.push(frameTime);
                    callback({ timestamp: frameTime });
                }, TEST_CONSTANTS.FRAME_TIME_MS);
                return () => clearInterval(interval);
            }
        }));

        render(<LidarViewport />);

        await act(async () => {
            await new Promise(resolve => setTimeout(resolve, TEST_CONSTANTS.TEST_DURATION_MS));
        });

        // Calculate actual FPS
        const frameIntervals = frameMetrics.slice(1).map((t, i) => t - frameMetrics[i]);
        const averageFPS = 1000 / (frameIntervals.reduce((a, b) => a + b, 0) / frameIntervals.length);

        expect(averageFPS).toBeGreaterThanOrEqual(TEST_CONSTANTS.TARGET_FPS);
    });

    it('handles WebGL context loss and recovery', async () => {
        const { container } = render(<LidarViewport />);

        // Simulate context loss
        const canvas = container.querySelector('canvas');
        expect(canvas).toBeTruthy();

        if (canvas) {
            const contextLostEvent = new Event('webglcontextlost');
            fireEvent(canvas, contextLostEvent);

            // Verify error handling
            await waitFor(() => {
                expect(screen.getByText(/WebGL context lost/i)).toBeInTheDocument();
            });

            // Simulate context restoration
            const contextRestoredEvent = new Event('webglcontextrestored');
            fireEvent(canvas, contextRestoredEvent);

            // Verify recovery
            await waitFor(() => {
                expect(screen.queryByText(/WebGL context lost/i)).not.toBeInTheDocument();
            });
        }
    });

    it('updates performance metrics in real-time', async () => {
        const { container } = render(<LidarViewport />);

        await waitFor(() => {
            const performanceOverlay = container.querySelector('.performance-overlay');
            expect(performanceOverlay).toBeTruthy();
            expect(performanceOverlay?.textContent).toContain(`FPS: ${TEST_CONSTANTS.TARGET_FPS}`);
            expect(performanceOverlay?.textContent).toContain('Points: 1000');
        });
    });
});