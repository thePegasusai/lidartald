import React, { useEffect, useCallback, memo, useState } from 'react';
import { Box, Typography, Alert, CircularProgress, useTheme } from '@mui/material'; // v5.13.0
import { useSelector, useDispatch } from 'react-redux'; // v8.0.5
import { ErrorBoundary } from 'react-error-boundary'; // v4.0.4

import MainLayout from '../components/layout/MainLayout';
import FleetManager from '../components/fleet/FleetManager';
import { useFleetConnection } from '../hooks/useFleetConnection';
import { 
    selectFleetStatus, 
    selectFleetSyncStatus, 
    selectActiveMemberCount,
    clearFleetError 
} from '../store/slices/fleetSlice';
import { FleetStatus } from '../types/fleet.types';
import { FLEET_CONSTANTS } from '../config/constants';

// Performance monitoring constants
const PERFORMANCE_THRESHOLD_MS = 50;
const TELEMETRY_INTERVAL = 5000;

// Enhanced error fallback component
const ErrorFallback = ({ error, resetErrorBoundary }: { 
    error: Error; 
    resetErrorBoundary: () => void; 
}) => (
    <Alert 
        severity="error" 
        onClose={resetErrorBoundary}
        sx={{ margin: 2 }}
    >
        <Typography variant="subtitle1">Fleet Connection Error</Typography>
        <Typography variant="body2">{error.message}</Typography>
    </Alert>
);

// Main Fleet Mode component
const FleetMode: React.FC = memo(() => {
    const theme = useTheme();
    const dispatch = useDispatch();
    const [deviceId] = useState(() => crypto.randomUUID());
    const [performanceWarning, setPerformanceWarning] = useState(false);

    // Fleet connection hook with auto-reconnect
    const { 
        fleet,
        status,
        isConnected,
        metrics,
        createFleet,
        joinFleet,
        leaveFleet,
        syncState,
        getNetworkStats
    } = useFleetConnection({
        autoReconnect: true,
        syncInterval: FLEET_CONSTANTS.SYNC_INTERVAL_MS,
        encryption: true,
        maxRetries: FLEET_CONSTANTS.MAX_RETRY_ATTEMPTS
    });

    // Redux selectors
    const fleetStatus = useSelector(selectFleetStatus);
    const syncStatus = useSelector(selectFleetSyncStatus);
    const activeMemberCount = useSelector(selectActiveMemberCount);

    // Enhanced error handler with automatic recovery
    const handleConnectionError = useCallback((error: Error) => {
        console.error('[Fleet Mode Error]', {
            error,
            deviceId,
            status: fleetStatus,
            timestamp: new Date().toISOString()
        });

        // Attempt automatic recovery
        if (fleetStatus === FleetStatus.ACTIVE) {
            syncState().catch(console.error);
        }

        dispatch(clearFleetError());
    }, [fleetStatus, deviceId, syncState, dispatch]);

    // Performance monitoring effect
    useEffect(() => {
        const monitorPerformance = () => {
            const stats = getNetworkStats();
            setPerformanceWarning(stats.averageLatency > PERFORMANCE_THRESHOLD_MS);

            // Log telemetry data
            console.debug('[Fleet Performance]', {
                latency: stats.averageLatency,
                connections: stats.activeConnections,
                topology: stats.topology,
                healthScore: stats.healthScore,
                timestamp: new Date().toISOString()
            });
        };

        const monitorInterval = setInterval(monitorPerformance, TELEMETRY_INTERVAL);
        return () => clearInterval(monitorInterval);
    }, [getNetworkStats]);

    // Fleet status effect for automatic recovery
    useEffect(() => {
        if (fleetStatus === FleetStatus.DISCONNECTED && isConnected) {
            syncState().catch(handleConnectionError);
        }
    }, [fleetStatus, isConnected, syncState, handleConnectionError]);

    return (
        <MainLayout fleetSync={isConnected}>
            <ErrorBoundary
                FallbackComponent={ErrorFallback}
                onReset={() => dispatch(clearFleetError())}
                onError={handleConnectionError}
            >
                <Box
                    sx={{
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 2,
                        p: 2
                    }}
                >
                    {/* Status Header */}
                    <Box sx={{ 
                        display: 'flex', 
                        justifyContent: 'space-between',
                        alignItems: 'center'
                    }}>
                        <Typography variant="h6" component="h1">
                            Fleet Management
                        </Typography>
                        {status === FleetStatus.SYNCING && (
                            <CircularProgress 
                                size={24}
                                sx={{ ml: 2 }}
                                aria-label="Syncing fleet"
                            />
                        )}
                    </Box>

                    {/* Performance Warning */}
                    {performanceWarning && (
                        <Alert 
                            severity="warning"
                            sx={{ mb: 2 }}
                        >
                            High latency detected. Network performance may be degraded.
                        </Alert>
                    )}

                    {/* Fleet Status */}
                    <Alert 
                        severity={isConnected ? "success" : "info"}
                        sx={{ mb: 2 }}
                    >
                        <Typography variant="body2">
                            {isConnected 
                                ? `Connected - ${activeMemberCount}/${FLEET_CONSTANTS.MAX_DEVICES} devices` 
                                : 'Establishing fleet connection...'}
                        </Typography>
                        {syncStatus.syncProgress < 100 && (
                            <Typography variant="caption" display="block">
                                Synchronizing: {syncStatus.syncProgress}%
                            </Typography>
                        )}
                    </Alert>

                    {/* Fleet Manager Component */}
                    <FleetManager
                        deviceId={deviceId}
                        onFleetUpdate={syncState}
                        onError={handleConnectionError}
                    />

                    {/* Network Metrics */}
                    <Box sx={{ 
                        mt: 'auto',
                        p: 2,
                        bgcolor: theme.palette.background.paper,
                        borderRadius: 1
                    }}>
                        <Typography variant="subtitle2" gutterBottom>
                            Network Metrics
                        </Typography>
                        <Typography variant="body2">
                            Latency: {metrics.averageLatency}ms
                        </Typography>
                        <Typography variant="body2">
                            Active Connections: {metrics.activeConnections}
                        </Typography>
                        <Typography variant="body2">
                            Health Score: {metrics.healthScore}%
                        </Typography>
                    </Box>
                </Box>
            </ErrorBoundary>
        </MainLayout>
    );
});

FleetMode.displayName = 'FleetMode';

export default FleetMode;