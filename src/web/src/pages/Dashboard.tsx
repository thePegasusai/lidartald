import React, { useEffect, useCallback, useState, useMemo } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Grid, Paper, Box, Typography, useTheme, useMediaQuery } from '@mui/material'; // v5.13.0
import { ErrorBoundary } from 'react-error-boundary'; // v4.0.11
import { usePerformanceMonitor } from '@performance-monitor/react'; // v1.0.0

import MainLayout from '../components/layout/MainLayout';
import LidarViewport from '../components/lidar/LidarViewport';
import FleetManager from '../components/fleet/FleetManager';

// Constants based on technical specifications
const VIEWPORT_MIN_HEIGHT = 400;
const FLEET_SECTION_HEIGHT = 300;
const UPDATE_INTERVAL = 33; // ~30Hz refresh rate
const PERFORMANCE_THRESHOLD = 16.67; // Target 60 FPS
const MAX_BATCH_SIZE = 32;
const RECONNECT_INTERVAL = 5000;

// Styled components
const DashboardContainer = styled(Box)(({ theme }) => ({
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    padding: theme.spacing(3),
    overflow: 'hidden',
    transform: 'translateZ(0)', // Hardware acceleration
    backfaceVisibility: 'hidden',
    perspective: 1000
}));

const ViewportSection = styled(Paper)(({ theme }) => ({
    flex: 1,
    minHeight: VIEWPORT_MIN_HEIGHT,
    marginBottom: theme.spacing(3),
    position: 'relative',
    overflow: 'hidden',
    boxShadow: theme.shadows[4],
    '& canvas': {
        width: '100%',
        height: '100%'
    }
}));

const FleetSection = styled(Paper)(({ theme }) => ({
    height: FLEET_SECTION_HEIGHT,
    marginBottom: theme.spacing(3),
    overflow: 'auto',
    boxShadow: theme.shadows[2]
}));

// Error Fallback component
const ErrorFallback = ({ error, resetErrorBoundary }) => (
    <Box p={3} textAlign="center">
        <Typography variant="h6" color="error" gutterBottom>
            Something went wrong:
        </Typography>
        <Typography variant="body1" gutterBottom>
            {error.message}
        </Typography>
        <Button onClick={resetErrorBoundary} variant="contained" color="primary">
            Retry
        </Button>
    </Box>
);

// Custom hook for dashboard state management
const useDashboardState = () => {
    const dispatch = useDispatch();
    const [performanceMetrics, setPerformanceMetrics] = useState({
        fps: 0,
        memoryUsage: 0,
        updateLatency: 0,
        pointCount: 0
    });

    const handlePerformanceUpdate = useCallback((metrics) => {
        setPerformanceMetrics(metrics);
        if (metrics.updateLatency > PERFORMANCE_THRESHOLD) {
            console.warn(`Performance degradation detected: ${metrics.updateLatency}ms latency`);
        }
    }, []);

    const handleError = useCallback((error: Error) => {
        console.error('Dashboard error:', error);
        // Implement error handling logic
    }, []);

    return {
        performanceMetrics,
        handlePerformanceUpdate,
        handleError
    };
};

// Main Dashboard component
const Dashboard: React.FC = React.memo(() => {
    const theme = useTheme();
    const isLandscape = useMediaQuery(theme.breakpoints.up('md'));
    const { performanceMetrics, handlePerformanceUpdate, handleError } = useDashboardState();

    // Performance monitoring
    const { startMonitoring, stopMonitoring } = usePerformanceMonitor({
        sampleSize: 60,
        thresholds: {
            fps: 60,
            latency: PERFORMANCE_THRESHOLD,
            memory: 512 // MB
        }
    });

    useEffect(() => {
        startMonitoring();
        return () => stopMonitoring();
    }, [startMonitoring, stopMonitoring]);

    return (
        <ErrorBoundary FallbackComponent={ErrorFallback} onReset={() => window.location.reload()}>
            <MainLayout>
                <DashboardContainer>
                    <Grid container spacing={3}>
                        <Grid item xs={12} md={isLandscape ? 8 : 12}>
                            <ViewportSection elevation={4}>
                                <LidarViewport
                                    onPerformanceMetrics={handlePerformanceUpdate}
                                    onError={handleError}
                                />
                            </ViewportSection>
                        </Grid>

                        <Grid item xs={12} md={isLandscape ? 4 : 12}>
                            <FleetSection elevation={2}>
                                <FleetManager
                                    deviceId={crypto.randomUUID()}
                                    onError={handleError}
                                />
                            </FleetSection>

                            {performanceMetrics.fps > 0 && (
                                <Paper elevation={1} sx={{ p: 2, mt: 2 }}>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Performance Metrics
                                    </Typography>
                                    <Typography variant="body2">
                                        FPS: {Math.round(performanceMetrics.fps)}
                                    </Typography>
                                    <Typography variant="body2">
                                        Memory: {Math.round(performanceMetrics.memoryUsage)}MB
                                    </Typography>
                                    <Typography variant="body2">
                                        Latency: {Math.round(performanceMetrics.updateLatency)}ms
                                    </Typography>
                                    <Typography variant="body2">
                                        Points: {performanceMetrics.pointCount}
                                    </Typography>
                                </Paper>
                            )}
                        </Grid>
                    </Grid>
                </DashboardContainer>
            </MainLayout>
        </ErrorBoundary>
    );
});

Dashboard.displayName = 'Dashboard';

export default Dashboard;