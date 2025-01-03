import React, { useEffect, useMemo, useCallback, useRef } from 'react';
import { useSelector } from 'react-redux';
import { styled } from '@mui/material/styles';
import { Box, Typography, CircularProgress } from '@mui/material';
import { VirtualList } from 'react-window';
import { usePerformanceMonitor } from 'react-performance-monitor'; // v1.0.0

import PlayerStatus from './PlayerStatus';
import {
    GameSession,
    GameMode,
    GameStatus,
    PlayerState,
    EnvironmentData,
    FleetSync
} from '../../types/game.types';
import {
    selectCurrentSession,
    selectPlayerStates,
    selectEnvironmentData,
    selectFleetSync
} from '../../store/slices/gameSlice';

// Performance optimization constants
const OVERLAY_UPDATE_INTERVAL = 16.67; // 60 FPS target
const MAX_PLAYERS = 32;
const ENVIRONMENT_UPDATE_THRESHOLD = 100;

// Styled components for optimized rendering
const OverlayContainer = styled(Box)(({ theme }) => ({
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    display: 'flex',
    flexDirection: 'column',
    pointerEvents: 'none',
    zIndex: 100,
    padding: theme.spacing(2),
    background: 'linear-gradient(to bottom, rgba(0,0,0,0.2) 0%, transparent 20%)',
    '& > *': {
        pointerEvents: 'auto'
    }
}));

const StatusBar = styled(Box)(({ theme }) => ({
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: theme.spacing(1, 2),
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: theme.shape.borderRadius,
    marginBottom: theme.spacing(2)
}));

const EnvironmentInfo = styled(Box)(({ theme }) => ({
    position: 'absolute',
    bottom: theme.spacing(2),
    left: theme.spacing(2),
    padding: theme.spacing(1, 2),
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: theme.shape.borderRadius,
    maxWidth: '300px'
}));

const FleetStatus = styled(Box)(({ theme }) => ({
    position: 'absolute',
    top: theme.spacing(2),
    right: theme.spacing(2),
    padding: theme.spacing(1, 2),
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: theme.shape.borderRadius
}));

/**
 * High-performance game overlay component with environment and fleet integration
 */
const GameOverlay: React.FC = () => {
    // Performance monitoring setup
    const { fps, startMonitoring, stopMonitoring } = usePerformanceMonitor();
    
    // Refs for performance optimization
    const frameRef = useRef<number>();
    const lastUpdateRef = useRef<number>(0);

    // Memoized selectors for optimal performance
    const currentSession = useSelector(selectCurrentSession);
    const playerStates = useSelector(selectPlayerStates);
    const environmentData = useSelector(selectEnvironmentData);
    const fleetSync = useSelector(selectFleetSync);

    /**
     * Optimized render loop for 60 FPS performance
     */
    const updateOverlay = useCallback(() => {
        const now = performance.now();
        const delta = now - lastUpdateRef.current;

        if (delta >= OVERLAY_UPDATE_INTERVAL) {
            lastUpdateRef.current = now;
            // Additional frame-based updates can be implemented here
        }

        frameRef.current = requestAnimationFrame(updateOverlay);
    }, []);

    /**
     * Initialize performance monitoring and render loop
     */
    useEffect(() => {
        startMonitoring();
        frameRef.current = requestAnimationFrame(updateOverlay);

        return () => {
            stopMonitoring();
            if (frameRef.current) {
                cancelAnimationFrame(frameRef.current);
            }
        };
    }, [startMonitoring, stopMonitoring, updateOverlay]);

    /**
     * Memoized environment status component
     */
    const EnvironmentStatus = useMemo(() => {
        if (!environmentData) return null;

        return (
            <EnvironmentInfo>
                <Typography variant="subtitle2" color="white">
                    Environment Status
                </Typography>
                <Typography variant="body2" color="white">
                    Resolution: {environmentData.resolution.toFixed(2)}cm
                </Typography>
                <Typography variant="body2" color="white">
                    Features: {environmentData.features.length}
                </Typography>
                <Typography variant="body2" color="white">
                    Last Update: {new Date(environmentData.lastProcessed).toLocaleTimeString()}
                </Typography>
            </EnvironmentInfo>
        );
    }, [environmentData]);

    /**
     * Memoized fleet synchronization status
     */
    const FleetSyncStatus = useMemo(() => {
        if (!fleetSync) return null;

        return (
            <FleetStatus>
                <Typography variant="subtitle2" color="white">
                    Fleet Status
                </Typography>
                <Typography variant="body2" color="white">
                    Connected: {fleetSync.activeConnections}/{MAX_PLAYERS}
                </Typography>
                <Typography variant="body2" color="white">
                    Latency: {fleetSync.averageLatency.toFixed(1)}ms
                </Typography>
                <Typography variant="body2" color="white">
                    Health: {fleetSync.healthScore}%
                </Typography>
            </FleetStatus>
        );
    }, [fleetSync]);

    if (!currentSession) {
        return (
            <OverlayContainer>
                <Box display="flex" justifyContent="center" alignItems="center">
                    <CircularProgress size={24} />
                    <Typography variant="body1" color="white" ml={2}>
                        Initializing game session...
                    </Typography>
                </Box>
            </OverlayContainer>
        );
    }

    return (
        <OverlayContainer>
            <StatusBar>
                <Typography variant="h6" color="white">
                    {currentSession.mode} - {currentSession.status}
                </Typography>
                <Typography variant="body1" color="white">
                    FPS: {fps} | Players: {playerStates.length}/{MAX_PLAYERS}
                </Typography>
            </StatusBar>

            <Box display="flex" flex={1}>
                <PlayerStatus />
            </Box>

            {EnvironmentStatus}
            {FleetSyncStatus}
        </OverlayContainer>
    );
};

export default React.memo(GameOverlay);