import React, { useCallback, useState, useEffect, useRef, useMemo } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import styled from '@emotion/styled'; // v11.10.6
import { Button, Select, MenuItem, FormControl, InputLabel } from '@mui/material'; // v5.11.0
import { PerformanceMonitor } from '@performance-monitor/react'; // v1.0.0

import {
    GameMode,
    GameStatus,
    GameSession,
    PlayerState,
    GameConfig,
    EnvironmentData
} from '../../types/game.types';
import { useGameSession, useEnvironmentSync, useFleetSync } from '../../hooks/useGameSession';

// Performance optimization constants
const TARGET_FRAME_RATE = 60;
const ENVIRONMENT_UPDATE_INTERVAL = 33; // ~30Hz
const FLEET_SYNC_INTERVAL = 50; // <50ms latency

// Styled components with outdoor visibility optimization
const ControlsContainer = styled.div`
    background: rgba(0, 0, 0, 0.85);
    border-radius: 8px;
    padding: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    color: #ffffff;
    max-width: 400px;
    margin: 20px;

    @media (max-width: 768px) {
        width: 90%;
        margin: 10px auto;
    }
`;

const ControlRow = styled.div`
    display: flex;
    align-items: center;
    margin-bottom: 16px;
    gap: 12px;
`;

const StatusIndicator = styled.div<{ status: GameStatus }>`
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: ${({ status }) => {
        switch (status) {
            case GameStatus.IN_PROGRESS:
                return '#4CAF50';
            case GameStatus.PAUSED:
                return '#FFC107';
            case GameStatus.COMPLETED:
                return '#9E9E9E';
            default:
                return '#F44336';
        }
    }};
    margin-right: 8px;
`;

const MetricsDisplay = styled.div`
    font-family: monospace;
    font-size: 12px;
    color: #A0A0A0;
    margin-top: 16px;
    padding: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
`;

interface GameControlsProps {
    fleetId: string;
    environmentMapId: string;
    onPerformanceUpdate?: (metrics: { fps: number; latency: number }) => void;
}

const GameControls: React.FC<GameControlsProps> = ({
    fleetId,
    environmentMapId,
    onPerformanceUpdate
}) => {
    const dispatch = useDispatch();
    const performanceRef = useRef<{ fps: number; latency: number }>({ fps: 0, latency: 0 });
    
    // Game session management
    const {
        session,
        performance,
        updateGameState,
        gameState
    } = useGameSession({
        fleetId,
        environmentMapId,
        config: {
            resolution: 0.01,
            range: 5.0,
            updateRate: 30
        }
    });

    // Local state management
    const [selectedMode, setSelectedMode] = useState<GameMode>(GameMode.BATTLE_ARENA);
    const [playerCount, setPlayerCount] = useState<number>(2);

    // Memoized environment configuration
    const environmentConfig = useMemo(() => ({
        resolution: 0.01,
        range: 5.0,
        updateRate: ENVIRONMENT_UPDATE_INTERVAL
    }), []);

    // Performance monitoring callback
    const handlePerformanceUpdate = useCallback((metrics: { fps: number; latency: number }) => {
        performanceRef.current = metrics;
        onPerformanceUpdate?.(metrics);
    }, [onPerformanceUpdate]);

    // Game control handlers
    const handleGameStart = useCallback(async () => {
        try {
            if (!session) {
                const config: GameConfig = {
                    maxPlayers: playerCount,
                    duration: 3600,
                    scoreLimit: 1000,
                    environmentSettings: environmentConfig
                };

                await updateGameState({
                    sessionId: session?.id || '',
                    timestamp: Date.now(),
                    playerUpdates: [],
                    environmentUpdates: {}
                });
            }
        } catch (error) {
            console.error('[GameControls] Start error:', error);
        }
    }, [session, playerCount, environmentConfig, updateGameState]);

    const handleGamePause = useCallback(() => {
        if (session?.status === GameStatus.IN_PROGRESS) {
            updateGameState({
                sessionId: session.id,
                timestamp: Date.now(),
                playerUpdates: [],
                environmentUpdates: { status: GameStatus.PAUSED }
            });
        }
    }, [session, updateGameState]);

    const handleGameEnd = useCallback(async () => {
        if (session) {
            try {
                await updateGameState({
                    sessionId: session.id,
                    timestamp: Date.now(),
                    playerUpdates: [],
                    environmentUpdates: { status: GameStatus.COMPLETED }
                });
            } catch (error) {
                console.error('[GameControls] End error:', error);
            }
        }
    }, [session, updateGameState]);

    // Performance monitoring effect
    useEffect(() => {
        const performanceMonitor = new PerformanceMonitor({
            targetFrameRate: TARGET_FRAME_RATE,
            sampleSize: 60,
            onUpdate: handlePerformanceUpdate
        });

        performanceMonitor.start();

        return () => {
            performanceMonitor.stop();
        };
    }, [handlePerformanceUpdate]);

    return (
        <ControlsContainer>
            <ControlRow>
                <StatusIndicator status={session?.status || GameStatus.INITIALIZING} />
                <span>Game Status: {session?.status || 'Not Started'}</span>
            </ControlRow>

            <ControlRow>
                <FormControl fullWidth variant="outlined">
                    <InputLabel>Game Mode</InputLabel>
                    <Select
                        value={selectedMode}
                        onChange={(e) => setSelectedMode(e.target.value as GameMode)}
                        label="Game Mode"
                        disabled={session?.status === GameStatus.IN_PROGRESS}
                    >
                        {Object.values(GameMode).map((mode) => (
                            <MenuItem key={mode} value={mode}>
                                {mode.replace('_', ' ')}
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>
            </ControlRow>

            <ControlRow>
                <FormControl fullWidth variant="outlined">
                    <InputLabel>Players</InputLabel>
                    <Select
                        value={playerCount}
                        onChange={(e) => setPlayerCount(Number(e.target.value))}
                        label="Players"
                        disabled={session?.status === GameStatus.IN_PROGRESS}
                    >
                        {Array.from({ length: 31 }, (_, i) => i + 2).map((num) => (
                            <MenuItem key={num} value={num}>
                                {num} Players
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>
            </ControlRow>

            <ControlRow>
                <Button
                    variant="contained"
                    color="primary"
                    onClick={handleGameStart}
                    disabled={session?.status === GameStatus.IN_PROGRESS}
                    fullWidth
                >
                    Start Game
                </Button>
            </ControlRow>

            <ControlRow>
                <Button
                    variant="outlined"
                    color="warning"
                    onClick={handleGamePause}
                    disabled={!session || session.status !== GameStatus.IN_PROGRESS}
                    fullWidth
                >
                    Pause Game
                </Button>
            </ControlRow>

            <ControlRow>
                <Button
                    variant="outlined"
                    color="error"
                    onClick={handleGameEnd}
                    disabled={!session || session.status === GameStatus.COMPLETED}
                    fullWidth
                >
                    End Game
                </Button>
            </ControlRow>

            <MetricsDisplay>
                <div>FPS: {performanceRef.current.fps}</div>
                <div>Latency: {performanceRef.current.latency}ms</div>
                <div>Players: {session?.players.length || 0}/{playerCount}</div>
                <div>Environment: {environmentMapId}</div>
                <div>Fleet: {fleetId}</div>
            </MetricsDisplay>
        </ControlsContainer>
    );
};

export default GameControls;