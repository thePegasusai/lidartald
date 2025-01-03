import React, { useEffect, useCallback, useState, useRef, useMemo } from 'react';
import { Box, Button, Typography, Alert, Paper, CircularProgress } from '@mui/material';
import styled from '@emotion/styled';
import LidarViewport from '../components/lidar/LidarViewport';
import ScanProgress from '../components/lidar/ScanProgress';
import { useLidarScanner } from '../hooks/useLidarScanner';

// Constants based on technical specifications
const SCAN_PARAMETERS = {
    resolution: 0.01, // 0.01cm resolution
    range: 5.0,      // 5-meter range
    scanRate: 30     // 30Hz scan rate
};

const PERFORMANCE_THRESHOLDS = {
    targetFps: 60,
    minFps: 30,
    maxLatency: 50,
    memoryThreshold: 500 // MB
};

// Styled components with GPU acceleration and Material Design elevation
const ScannerContainer = styled(Paper)`
    display: flex;
    flex-direction: column;
    height: 100vh;
    padding: 24px;
    background-color: ${({ theme }) => theme.palette.background.default};
    transition: all 200ms ease-in-out;
    position: relative;
    isolation: isolate;
    transform: translateZ(0);
    will-change: transform;
`;

const ViewportContainer = styled(Box)`
    flex: 1;
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: ${({ theme }) => theme.shadows[8]};
    background-color: ${({ theme }) => theme.palette.background.paper};
    transform: translateZ(0);
    will-change: transform;
`;

const ControlsContainer = styled(Box)`
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 16px;
    gap: 16px;
`;

const PerformanceIndicator = styled(Box)`
    position: absolute;
    top: 16px;
    right: 16px;
    padding: 8px;
    background: rgba(0, 0, 0, 0.7);
    border-radius: 4px;
    color: ${({ theme }) => theme.palette.common.white};
    font-family: monospace;
    z-index: 10;
`;

const EnvironmentScanner: React.FC = () => {
    const {
        isScanning,
        scanResult,
        startScan,
        stopScan,
        error,
        performance,
        deviceStatus
    } = useLidarScanner(SCAN_PARAMETERS);

    const [performanceMode, setPerformanceMode] = useState<'quality' | 'balanced' | 'performance'>('balanced');
    const frameCountRef = useRef(0);
    const lastFrameTimeRef = useRef(Date.now());

    // Monitor performance and adjust settings
    useEffect(() => {
        const monitorPerformance = () => {
            const now = Date.now();
            const elapsed = now - lastFrameTimeRef.current;
            
            if (elapsed >= 1000) {
                const fps = (frameCountRef.current * 1000) / elapsed;
                
                // Adjust performance mode based on FPS and battery
                if (deviceStatus.batteryLevel < 20) {
                    setPerformanceMode('performance');
                } else if (fps < PERFORMANCE_THRESHOLDS.minFps) {
                    setPerformanceMode('performance');
                } else if (fps >= PERFORMANCE_THRESHOLDS.targetFps) {
                    setPerformanceMode('quality');
                }

                frameCountRef.current = 0;
                lastFrameTimeRef.current = now;
            }

            frameCountRef.current++;
            requestAnimationFrame(monitorPerformance);
        };

        const rafId = requestAnimationFrame(monitorPerformance);
        return () => cancelAnimationFrame(rafId);
    }, [deviceStatus.batteryLevel]);

    // Handle scan completion
    const handleScanComplete = useCallback(() => {
        if (scanResult) {
            console.log('Scan completed:', {
                points: scanResult.pointCloud.points.length,
                quality: scanResult.quality,
                processingTime: performance?.processingTime
            });
        }
    }, [scanResult, performance]);

    // Render performance metrics
    const renderPerformanceMetrics = useMemo(() => (
        <PerformanceIndicator>
            <Typography variant="caption" component="div">
                FPS: {Math.round(performance?.processingTime ? 1000 / performance.processingTime : 0)}
            </Typography>
            <Typography variant="caption" component="div">
                Points: {scanResult?.pointCloud.points.length || 0}
            </Typography>
            <Typography variant="caption" component="div">
                Memory: {Math.round(performance?.memoryUsage || 0)}MB
            </Typography>
            <Typography variant="caption" component="div">
                Battery: {deviceStatus.batteryLevel}%
            </Typography>
        </PerformanceIndicator>
    ), [performance, scanResult, deviceStatus.batteryLevel]);

    return (
        <ScannerContainer elevation={4}>
            <ViewportContainer>
                <LidarViewport
                    className="lidar-viewport"
                    style={{ width: '100%', height: '100%' }}
                    performanceMode={performanceMode}
                />
                {renderPerformanceMetrics}
            </ViewportContainer>

            <ScanProgress
                showDetails
                onComplete={handleScanComplete}
            />

            <ControlsContainer>
                <Button
                    variant="contained"
                    color={isScanning ? "error" : "primary"}
                    onClick={isScanning ? stopScan : startScan}
                    disabled={deviceStatus.batteryLevel < 5}
                >
                    {isScanning ? "Stop Scan" : "Start Scan"}
                </Button>

                <Typography variant="body2" color="textSecondary">
                    Resolution: {SCAN_PARAMETERS.resolution}cm | 
                    Range: {SCAN_PARAMETERS.range}m | 
                    Rate: {SCAN_PARAMETERS.scanRate}Hz
                </Typography>
            </ControlsContainer>

            {error && (
                <Alert 
                    severity="error" 
                    sx={{ mt: 2 }}
                    onClose={() => {}}
                >
                    {error}
                </Alert>
            )}

            {deviceStatus.batteryLevel < 20 && (
                <Alert 
                    severity="warning" 
                    sx={{ mt: 2 }}
                >
                    Low battery: Performance may be reduced
                </Alert>
            )}
        </ScannerContainer>
    );
};

export default EnvironmentScanner;