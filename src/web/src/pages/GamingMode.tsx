import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import styled from '@emotion/styled'; // v11.10.6
import { Box, Typography, CircularProgress } from '@mui/material'; // v5.11.0

import EnvironmentMap from '../components/game/EnvironmentMap';
import GameControls from '../components/game/GameControls';
import { useGameSession } from '../hooks/useGameSession';
import { UI_CONSTANTS, ENVIRONMENT_CONSTANTS } from '../config/constants';
import { GameStatus, GameMode, GameMetrics } from '../types/game.types';
import { SurfaceClassification } from '../types/environment.types';

// Constants for performance optimization
const RENDER_INTERVAL_MS = Math.floor(1000 / UI_CONSTANTS.TARGET_FPS);
const ENVIRONMENT_UPDATE_INTERVAL = Math.floor(1000 / ENVIRONMENT_CONSTANTS.FEATURE_UPDATE_INTERVAL_MS);

// Styled components with high-contrast optimization for outdoor visibility
const GameContainer = styled(Box)`
    position: relative;
    width: 100%;
    height: 100vh;
    display: flex;
    background-color: rgba(0, 0, 0, 0.9);
`;

const EnvironmentContainer = styled(Box)`
    flex: 1;
    position: relative;
    overflow: hidden;
`;

const ControlsOverlay = styled(Box)`
    position: absolute;
    top: 0;
    right: 0;
    z-index: 10;
    max-width: 400px;
    height: 100%;
    background: rgba(0, 0, 0, 0.85);
    backdrop-filter: blur(10px);
`;

const LoadingOverlay = styled(Box)`
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.7);
    z-index: 20;
`;

const PerformanceMetrics = styled(Box)`
    position: absolute;
    bottom: 16px;
    left: 16px;
    padding: 8px;
    background: rgba(0, 0, 0, 0.8);
    border-radius: 4px;
    color: #ffffff;
    font-family: monospace;
    font-size: 12px;
    z-index: 15;
`;

/**
 * GamingMode component - Main page for TALD UNIA gaming interface
 * Implements 60 FPS UI updates and 30Hz environment synchronization
 */
const GamingMode: React.FC = () => {
    const dispatch = useDispatch();
    
    // Performance monitoring refs
    const frameRef = useRef<number>(0);
    const lastFrameTime = useRef<number>(0);
    const performanceMetrics = useRef<GameMetrics>({
        fps: 0,
        updateLatency: 0,
        playerCount: 0,
        environmentLoad: 0,
        networkStatus: {
            connected: false,
            latency: 0,
            packetLoss: 0
        }
    });

    // State management
    const [isInitializing, setIsInitializing] = useState(true);
    const [environmentMapId, setEnvironmentMapId] = useState<string>('');
    const [fleetId, setFleetId] = useState<string>('');

    // Initialize game session with optimized configuration
    const {
        session,
        performance,
        updateGameState,
        gameState
    } = useGameSession({
        fleetId,
        gameMode: GameMode.BATTLE_ARENA,
        environmentConfig: {
            resolution: ENVIRONMENT_CONSTANTS.MIN_SURFACE_POINTS,
            range: LIDAR_CONSTANTS.MAX_RANGE_M,
            updateRate: ENVIRONMENT_CONSTANTS.FEATURE_UPDATE_INTERVAL_MS
        }
    });

    /**
     * Handles environment map updates with batched processing
     */
    const handleEnvironmentUpdate = useCallback((map: EnvironmentMap) => {
        if (!session) return;

        // Update game state with new environment data
        updateGameState({
            sessionId: session.id,
            timestamp: Date.now(),
            playerUpdates: [],
            environmentUpdates: {
                mapId: map.id,
                features: map.features.filter(f => 
                    f.confidence >= ENVIRONMENT_CONSTANTS.MIN_SURFACE_POINTS &&
                    f.classification !== SurfaceClassification.Unknown
                )
            }
        });

        setEnvironmentMapId(map.id);
    }, [session, updateGameState]);

    /**
     * Handles performance metric updates
     */
    const handlePerformanceUpdate = useCallback((metrics: { fps: number; latency: number }) => {
        performanceMetrics.current = {
            ...performanceMetrics.current,
            fps: metrics.fps,
            updateLatency: metrics.latency,
            playerCount: session?.players.length || 0,
            environmentLoad: performance.environmentLoad
        };
    }, [session, performance]);

    /**
     * Initializes RAF-based render loop for smooth UI updates
     */
    useEffect(() => {
        let animationFrameId: number;

        const renderLoop = (timestamp: number) => {
            // Calculate FPS
            if (timestamp - lastFrameTime.current >= 1000) {
                performanceMetrics.current.fps = frameRef.current;
                frameRef.current = 0;
                lastFrameTime.current = timestamp;
            }
            frameRef.current++;

            animationFrameId = requestAnimationFrame(renderLoop);
        };

        animationFrameId = requestAnimationFrame(renderLoop);

        // Cleanup
        return () => {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
            }
        };
    }, []);

    /**
     * Initialize fleet and environment on component mount
     */
    useEffect(() => {
        const initializeGame = async () => {
            try {
                // Create new fleet session
                const fleetResponse = await fetch('/api/fleet/create', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const { fleetId: newFleetId } = await fleetResponse.json();
                setFleetId(newFleetId);

                setIsInitializing(false);
            } catch (error) {
                console.error('[GamingMode] Initialization error:', error);
                setIsInitializing(false);
            }
        };

        initializeGame();
    }, []);

    if (isInitializing) {
        return (
            <LoadingOverlay>
                <CircularProgress />
                <Typography variant="h6" color="white" ml={2}>
                    Initializing Gaming Environment...
                </Typography>
            </LoadingOverlay>
        );
    }

    return (
        <GameContainer>
            <EnvironmentContainer>
                <EnvironmentMap
                    resolution={0.01}
                    autoStart={true}
                    onMapUpdate={handleEnvironmentUpdate}
                />
            </EnvironmentContainer>

            <ControlsOverlay>
                <GameControls
                    fleetId={fleetId}
                    environmentMapId={environmentMapId}
                    onPerformanceUpdate={handlePerformanceUpdate}
                />
            </ControlsOverlay>

            <PerformanceMetrics>
                <Typography variant="caption" component="div">
                    FPS: {performanceMetrics.current.fps}
                </Typography>
                <Typography variant="caption" component="div">
                    Latency: {performanceMetrics.current.updateLatency}ms
                </Typography>
                <Typography variant="caption" component="div">
                    Players: {performanceMetrics.current.playerCount}
                </Typography>
                <Typography variant="caption" component="div">
                    Environment Load: {Math.round(performanceMetrics.current.environmentLoad * 100)}%
                </Typography>
                <Typography variant="caption" component="div">
                    Network: {performanceMetrics.current.networkStatus.connected ? 'Connected' : 'Disconnected'}
                </Typography>
            </PerformanceMetrics>
        </GameContainer>
    );
};

export default GamingMode;