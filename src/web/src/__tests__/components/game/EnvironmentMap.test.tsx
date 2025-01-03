import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'; // v14.0.0
import { Provider } from 'react-redux'; // v8.1.0
import * as THREE from 'three'; // v0.150.0
import { PerformanceObserver } from 'perf_hooks'; // v1.0.0

import { EnvironmentMap } from '../../../components/game/EnvironmentMap';
import { EnvironmentMap as IEnvironmentMap, EnvironmentFeature, Point3D, MemoryOptimizationConfig } from '../../../types/environment.types';
import { useEnvironmentMap } from '../../../hooks/useEnvironmentMap';

// Test constants
const TEST_RESOLUTION = 0.01;
const TEST_SCAN_RATE = 30;
const MOCK_CANVAS_SIZE = 800;
const TARGET_FPS = 60;
const GPU_MEMORY_LIMIT = 512;

// Mock WebGL context
const mockWebGLContext = {
    getContext: jest.fn(),
    drawArrays: jest.fn(),
    createBuffer: jest.fn(),
    bindBuffer: jest.fn(),
    bufferData: jest.fn()
};

// Mock Three.js
jest.mock('three', () => ({
    ...jest.requireActual('three'),
    WebGLRenderer: jest.fn().mockImplementation(() => ({
        setSize: jest.fn(),
        render: jest.fn(),
        dispose: jest.fn(),
        domElement: document.createElement('canvas'),
        info: {
            render: {
                calls: 0,
                triangles: 0,
                points: 0
            }
        }
    })),
    Scene: jest.fn().mockImplementation(() => ({
        add: jest.fn(),
        remove: jest.fn()
    })),
    PerspectiveCamera: jest.fn().mockImplementation(() => ({
        position: { set: jest.fn() },
        aspect: 1,
        updateProjectionMatrix: jest.fn()
    }))
}));

// Mock Redux hooks
jest.mock('react-redux', () => ({
    ...jest.requireActual('react-redux'),
    useSelector: jest.fn(),
    useDispatch: jest.fn()
}));

// Mock environment hook
jest.mock('../../../hooks/useEnvironmentMap');

describe('EnvironmentMap Component', () => {
    let mockStore: any;
    let mockPerformanceObserver: any;

    // Setup test environment
    const setupTest = (mockState = {}, gpuConfig = {}) => {
        // Initialize mock store
        mockStore = {
            getState: () => mockState,
            dispatch: jest.fn(),
            subscribe: jest.fn()
        };

        // Setup WebGL context
        const canvas = document.createElement('canvas');
        canvas.width = MOCK_CANVAS_SIZE;
        canvas.height = MOCK_CANVAS_SIZE;
        Object.defineProperty(canvas, 'getContext', {
            value: () => mockWebGLContext
        });

        // Setup performance monitoring
        mockPerformanceObserver = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            entries.forEach(entry => {
                if (entry.entryType === 'measure' && entry.name === 'frame') {
                    const fps = 1000 / entry.duration;
                    expect(fps).toBeGreaterThanOrEqual(TARGET_FPS);
                }
            });
        });
        mockPerformanceObserver.observe({ entryTypes: ['measure'] });

        return render(
            <Provider store={mockStore}>
                <EnvironmentMap
                    resolution={TEST_RESOLUTION}
                    autoStart={true}
                    gpuAcceleration={true}
                    memoryOptimization={{
                        maxPoints: 100000,
                        cleanupInterval: 5000,
                        maxMemoryUsage: GPU_MEMORY_LIMIT
                    }}
                />
            </Provider>
        );
    };

    // Generate mock point cloud data
    const mockPointCloud = (resolution: number, confidence: number): Point3D[] => {
        const points: Point3D[] = [];
        for (let i = 0; i < 1000; i++) {
            points.push({
                x: Math.random() * 5,
                y: Math.random() * 5,
                z: Math.random() * 5,
                intensity: Math.random()
            });
        }
        return points;
    };

    beforeEach(() => {
        jest.clearAllMocks();
        (useEnvironmentMap as jest.Mock).mockReturnValue({
            currentMap: null,
            scanProgress: 0,
            isScanning: false,
            memoryUsage: { memoryUsageMB: 0 },
            startScan: jest.fn(),
            stopScan: jest.fn()
        });
    });

    afterEach(() => {
        mockPerformanceObserver.disconnect();
    });

    it('renders with GPU acceleration enabled', async () => {
        const { container } = setupTest();
        const canvas = container.querySelector('canvas');
        expect(canvas).toBeInTheDocument();
        expect(mockWebGLContext.getContext).toHaveBeenCalledWith('webgl2');
    });

    it('maintains target performance metrics', async () => {
        const { container } = setupTest();
        
        await act(async () => {
            // Simulate 30Hz updates
            for (let i = 0; i < 30; i++) {
                const points = mockPointCloud(TEST_RESOLUTION, 0.9);
                (useEnvironmentMap as jest.Mock).mockReturnValue({
                    currentMap: {
                        points,
                        features: [],
                        timestamp: Date.now()
                    },
                    scanProgress: (i / 30) * 100,
                    isScanning: true,
                    memoryUsage: { memoryUsageMB: 256 }
                });

                // Wait for frame
                await new Promise(resolve => setTimeout(resolve, 1000 / TEST_SCAN_RATE));
            }
        });

        // Verify performance metrics
        expect(mockWebGLContext.drawArrays).toHaveBeenCalled();
        const memoryUsage = screen.getByText(/Memory:/);
        expect(memoryUsage).toHaveTextContent(/Memory: \d+MB/);
        expect(parseInt(memoryUsage.textContent!.match(/\d+/)![0])).toBeLessThan(GPU_MEMORY_LIMIT);
    });

    it('handles feature detection correctly', async () => {
        const mockFeatures: EnvironmentFeature[] = [
            {
                id: '1',
                type: 'SURFACE',
                points: mockPointCloud(TEST_RESOLUTION, 0.95),
                classification: 'FLOOR',
                confidence: 0.95,
                metadata: {},
                lastUpdated: Date.now(),
                boundingBox: new THREE.Box3()
            }
        ];

        setupTest({
            environment: {
                currentMap: {
                    points: mockPointCloud(TEST_RESOLUTION, 0.9),
                    features: mockFeatures
                }
            }
        });

        await waitFor(() => {
            expect(mockWebGLContext.bufferData).toHaveBeenCalled();
        });
    });

    it('manages resources efficiently', async () => {
        const { unmount } = setupTest();

        // Simulate memory pressure
        await act(async () => {
            for (let i = 0; i < 10; i++) {
                (useEnvironmentMap as jest.Mock).mockReturnValue({
                    currentMap: {
                        points: mockPointCloud(TEST_RESOLUTION, 0.9),
                        features: [],
                        timestamp: Date.now()
                    },
                    scanProgress: 100,
                    isScanning: true,
                    memoryUsage: { memoryUsageMB: GPU_MEMORY_LIMIT - 100 }
                });

                await new Promise(resolve => setTimeout(resolve, 100));
            }
        });

        // Verify cleanup
        unmount();
        expect(mockWebGLContext.dispose).toHaveBeenCalled();
    });
});