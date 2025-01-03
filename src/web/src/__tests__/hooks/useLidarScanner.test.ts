import { renderHook, act } from '@testing-library/react-hooks'; // v8.0.1
import { Provider } from 'react-redux'; // v8.1.1
import { configureStore } from '@reduxjs/toolkit'; // v1.9.5
import { jest } from '@jest/globals'; // v29.5.0
import { useLidarScanner } from '../../hooks/useLidarScanner';
import { startScan, stopScan, getPointCloud, getScanStatus, DEFAULT_SCAN_PARAMETERS } from '../../api/lidarApi';
import lidarReducer from '../../store/slices/lidarSlice';

// Mock API functions
jest.mock('../../api/lidarApi');
const mockStartScan = startScan as jest.MockedFunction<typeof startScan>;
const mockStopScan = stopScan as jest.MockedFunction<typeof stopScan>;
const mockGetPointCloud = getPointCloud as jest.MockedFunction<typeof getPointCloud>;
const mockGetScanStatus = getScanStatus as jest.MockedFunction<typeof getScanStatus>;

// Mock performance monitoring
const mockPerformanceNow = jest.spyOn(performance, 'now');
const mockMemory = { usedJSHeapSize: 100 * 1024 * 1024 }; // 100MB
Object.defineProperty(performance, 'memory', { get: () => mockMemory });

// Test store configuration
const createTestStore = () => configureStore({
    reducer: {
        lidar: lidarReducer
    }
});

describe('useLidarScanner', () => {
    let store: ReturnType<typeof createTestStore>;

    beforeEach(() => {
        jest.clearAllMocks();
        store = createTestStore();

        // Reset performance mocks
        mockPerformanceNow.mockReturnValue(0);
        
        // Mock successful scan start
        mockStartScan.mockResolvedValue({ scanId: '123', quality: 1.0 });
        
        // Mock point cloud data
        mockGetPointCloud.mockResolvedValue({
            points: Array(1000).fill(null).map((_, i) => ({
                x: i * 0.01,
                y: i * 0.01,
                z: i * 0.01,
                intensity: 0.8
            })),
            quality: 0.95,
            density: 100
        });

        // Mock scan status
        mockGetScanStatus.mockResolvedValue({
            isActive: true,
            temperature: 45,
            memoryUsage: 100,
            batteryLevel: 80
        });

        // Setup fake timers
        jest.useFakeTimers();
    });

    afterEach(() => {
        jest.clearAllTimers();
        jest.useRealTimers();
    });

    it('should initialize with default state', () => {
        const wrapper = ({ children }: { children: React.ReactNode }) => (
            <Provider store={store}>{children}</Provider>
        );

        const { result } = renderHook(() => useLidarScanner(), { wrapper });

        expect(result.current.isScanning).toBe(false);
        expect(result.current.scanResult).toBeNull();
        expect(result.current.error).toBeNull();
        expect(result.current.performance).toEqual({
            processingTime: 0,
            memoryUsage: 0,
            pointCount: 0,
            updateLatency: 0,
            thermalStatus: 0
        });
    });

    it('should maintain 30Hz scan rate', async () => {
        const wrapper = ({ children }: { children: React.ReactNode }) => (
            <Provider store={store}>{children}</Provider>
        );

        const { result } = renderHook(() => useLidarScanner(), { wrapper });

        // Start scanning
        await act(async () => {
            await result.current.startScan();
        });

        // Verify initial state
        expect(result.current.isScanning).toBe(true);

        // Track scan updates
        const scanUpdates: number[] = [];
        let lastUpdate = Date.now();

        // Monitor 10 scan cycles
        for (let i = 0; i < 10; i++) {
            await act(async () => {
                // Advance timer by 33ms (30Hz)
                jest.advanceTimersByTime(33);
                
                // Track time between updates
                const now = Date.now();
                scanUpdates.push(now - lastUpdate);
                lastUpdate = now;
            });
        }

        // Verify scan rate
        const averageInterval = scanUpdates.reduce((a, b) => a + b) / scanUpdates.length;
        expect(averageInterval).toBeCloseTo(33, 1); // 33ms ±1ms tolerance

        // Verify point cloud processing
        expect(mockGetPointCloud).toHaveBeenCalledTimes(10);
        expect(result.current.performance.processingTime).toBeLessThan(33); // Under 33ms
        expect(result.current.scanResult?.quality).toBeGreaterThan(0.8);
    });

    it('should handle scan errors correctly', async () => {
        const wrapper = ({ children }: { children: React.ReactNode }) => (
            <Provider store={store}>{children}</Provider>
        );

        const { result } = renderHook(() => useLidarScanner(), { wrapper });

        // Mock scan error
        mockStartScan.mockRejectedValueOnce(new Error('Hardware initialization failed'));

        // Attempt to start scanning
        await act(async () => {
            await result.current.startScan();
        });

        // Verify error handling
        expect(result.current.isScanning).toBe(false);
        expect(result.current.error).toBe('Hardware initialization failed');
        expect(result.current.scanResult).toBeNull();

        // Verify cleanup
        expect(mockStopScan).toHaveBeenCalled();
    });

    it('should clean up resources on unmount', async () => {
        const wrapper = ({ children }: { children: React.ReactNode }) => (
            <Provider store={store}>{children}</Provider>
        );

        const { result, unmount } = renderHook(() => useLidarScanner(), { wrapper });

        // Start scanning
        await act(async () => {
            await result.current.startScan();
        });

        // Verify scanning started
        expect(result.current.isScanning).toBe(true);

        // Unmount hook
        unmount();

        // Verify cleanup
        expect(mockStopScan).toHaveBeenCalled();
        expect(store.getState().lidar.isScanning).toBe(false);
    });

    it('should validate scan parameters', async () => {
        const wrapper = ({ children }: { children: React.ReactNode }) => (
            <Provider store={store}>{children}</Provider>
        );

        const { result } = renderHook(() => useLidarScanner({
            resolution: 0.005, // 0.005cm resolution
            range: 6.0 // 6m range (exceeds limit)
        }), { wrapper });

        await act(async () => {
            await result.current.startScan();
        });

        // Verify parameters were validated
        expect(mockStartScan).toHaveBeenCalledWith({
            ...DEFAULT_SCAN_PARAMETERS,
            resolution: 0.01, // Should be clamped to minimum
            range: 5.0 // Should be clamped to maximum
        });
    });
});