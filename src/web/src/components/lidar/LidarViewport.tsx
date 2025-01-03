import React, { useEffect, useCallback, useState, useRef, useMemo } from 'react';
import styled from '@emotion/styled'; // v11.10.6
import { ErrorBoundary } from '@sentry/react'; // v7.0.0
import { usePerformanceMonitor } from '@microsoft/performance-monitor'; // v1.0.0
import PointCloudRenderer from './PointCloudRenderer';
import FeatureDetection from './FeatureDetection';
import { useLidarScanner } from '../../hooks/useLidarScanner';

// Constants based on technical specifications
const VIEWPORT_UPDATE_INTERVAL = 33; // 30Hz refresh rate
const SCAN_RANGE_METERS = 5;
const MIN_RESOLUTION_CM = 0.01;
const PERFORMANCE_THRESHOLDS = {
    fps: 60,
    latency: 16.67, // ~60 FPS target
    memoryLimit: 512 // MB
};

// Styled components for enterprise-grade UI
const ViewportContainer = styled.div`
    position: relative;
    width: 100%;
    height: 100%;
    background: #000;
    overflow: hidden;
`;

const PerformanceOverlay = styled.div`
    position: absolute;
    top: 8px;
    right: 8px;
    padding: 4px 8px;
    background: rgba(0, 0, 0, 0.7);
    color: #fff;
    font-family: monospace;
    font-size: 12px;
    z-index: 100;
`;

const ErrorOverlay = styled.div`
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    padding: 16px;
    background: rgba(255, 0, 0, 0.8);
    color: #fff;
    border-radius: 4px;
    z-index: 1000;
`;

// Types for component props
interface LidarViewportProps {
    className?: string;
    style?: React.CSSProperties;
    onError?: (error: Error) => void;
    onPerformanceAlert?: (metrics: PerformanceMetrics) => void;
}

interface PerformanceMetrics {
    fps: number;
    latency: number;
    memoryUsage: number;
    pointCount: number;
}

// Custom hook for viewport dimensions
const useViewportDimensions = () => {
    const [dimensions, setDimensions] = useState({
        width: 0,
        height: 0,
        aspectRatio: 1
    });

    const updateDimensions = useCallback((element: HTMLDivElement) => {
        const { clientWidth, clientHeight } = element;
        setDimensions({
            width: clientWidth,
            height: clientHeight,
            aspectRatio: clientWidth / clientHeight
        });
    }, []);

    return { dimensions, updateDimensions };
};

// Performance monitoring decorator
function PerformanceMonitoring() {
    return function (target: any) {
        return class extends target {
            private performanceMonitor = usePerformanceMonitor({
                thresholds: PERFORMANCE_THRESHOLDS
            });

            componentDidMount() {
                super.componentDidMount?.();
                this.startPerformanceMonitoring();
            }

            componentWillUnmount() {
                super.componentWillUnmount?.();
                this.stopPerformanceMonitoring();
            }

            private startPerformanceMonitoring() {
                this.performanceMonitor.start();
            }

            private stopPerformanceMonitoring() {
                this.performanceMonitor.stop();
            }
        };
    };
}

// Main component implementation
@ErrorBoundary()
@PerformanceMonitoring()
const LidarViewport: React.FC<LidarViewportProps> = ({
    className,
    style,
    onError,
    onPerformanceAlert
}) => {
    const viewportRef = useRef<HTMLDivElement>(null);
    const { dimensions, updateDimensions } = useViewportDimensions();
    const { isScanning, scanResult, error, performance, startScan, stopScan } = useLidarScanner({
        resolution: MIN_RESOLUTION_CM,
        range: SCAN_RANGE_METERS,
        scanRate: 30
    });

    // Performance monitoring
    useEffect(() => {
        if (onPerformanceAlert && performance) {
            onPerformanceAlert({
                fps: performance.processingTime ? 1000 / performance.processingTime : 0,
                latency: performance.updateLatency,
                memoryUsage: performance.memoryUsage,
                pointCount: scanResult?.pointCloud.points.length || 0
            });
        }
    }, [performance, scanResult, onPerformanceAlert]);

    // Dimension updates
    useEffect(() => {
        if (!viewportRef.current) return;

        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                updateDimensions(entry.target as HTMLDivElement);
            }
        });

        resizeObserver.observe(viewportRef.current);
        return () => resizeObserver.disconnect();
    }, [updateDimensions]);

    // Error handling
    useEffect(() => {
        if (error && onError) {
            onError(new Error(error));
        }
    }, [error, onError]);

    // Auto-start scanning
    useEffect(() => {
        startScan();
        return () => stopScan();
    }, [startScan, stopScan]);

    return (
        <ViewportContainer
            ref={viewportRef}
            className={className}
            style={style}
        >
            <PointCloudRenderer
                width={dimensions.width}
                height={dimensions.height}
                performanceMode="balanced"
                onPerformanceMetrics={onPerformanceAlert}
            />

            <FeatureDetection
                className="feature-overlay"
                style={{ position: 'absolute', top: 0, left: 0 }}
                confidenceThreshold={0.85}
                frameRateLimit={30}
            />

            {performance && (
                <PerformanceOverlay>
                    FPS: {Math.round(performance.processingTime ? 1000 / performance.processingTime : 0)}
                    <br />
                    Points: {scanResult?.pointCloud.points.length || 0}
                    <br />
                    Memory: {Math.round(performance.memoryUsage)}MB
                    <br />
                    Latency: {Math.round(performance.updateLatency)}ms
                </PerformanceOverlay>
            )}

            {error && (
                <ErrorOverlay>
                    Error: {error}
                </ErrorOverlay>
            )}
        </ViewportContainer>
    );
};

export default LidarViewport;