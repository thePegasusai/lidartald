import { useEffect, useCallback, useState, useRef, useMemo } from 'react'; // v18.2.0
import { useDispatch, useSelector, createSelector } from 'react-redux'; // v8.0.5
import * as automerge from 'automerge'; // v1.0.1

import {
  GameSession,
  GameMode,
  GameStatus,
  PlayerState,
  GameStateUpdate,
  EnvironmentState,
  MeshTopology
} from '../../types/game.types';
import { gameApi } from '../../api/gameApi';

// Constants for performance optimization
const STATE_UPDATE_INTERVAL_MS = 16; // ~60Hz updates
const MAX_PLAYERS = 32;
const ENVIRONMENT_UPDATE_RATE = 30;
const PREDICTION_THRESHOLD_MS = 50;

/**
 * Interface for hook configuration
 */
interface GameSessionConfig {
  fleetId: string;
  gameMode: GameMode;
  environmentConfig: {
    resolution: number;
    range: number;
    updateRate: number;
  };
}

/**
 * Interface for performance metrics
 */
interface PerformanceMetrics {
  fps: number;
  updateLatency: number;
  predictionAccuracy: number;
  environmentLoad: number;
}

/**
 * Custom hook for managing game session lifecycle with optimized performance
 */
export const useGameSession = ({
  fleetId,
  gameMode,
  environmentConfig
}: GameSessionConfig) => {
  // Redux setup
  const dispatch = useDispatch();
  
  // Memoized selectors for performance
  const selectGameState = useMemo(
    () => createSelector(
      [(state: any) => state.game],
      (gameState) => gameState
    ),
    []
  );

  // Local state management
  const [session, setSession] = useState<GameSession | null>(null);
  const [performance, setPerformance] = useState<PerformanceMetrics>({
    fps: 0,
    updateLatency: 0,
    predictionAccuracy: 0,
    environmentLoad: 0
  });

  // CRDT state for conflict-free updates
  const gameState = useRef(automerge.init<GameSession>());
  const lastUpdate = useRef<number>(0);
  const frameCount = useRef<number>(0);
  const animationFrameId = useRef<number>();

  /**
   * Initialize game session with environment integration
   */
  const initializeSession = useCallback(async () => {
    try {
      const newSession = await gameApi.createGameSession({
        fleetId,
        mode: gameMode,
        config: {
          maxPlayers: MAX_PLAYERS,
          duration: 3600,
          scoreLimit: 1000,
          environmentSettings: {
            resolution: environmentConfig.resolution,
            range: environmentConfig.range,
            updateRate: environmentConfig.updateRate
          }
        }
      });

      // Initialize CRDT state
      gameState.current = automerge.from(newSession);
      setSession(newSession);

      // Start performance monitoring
      startPerformanceMonitoring();
    } catch (error) {
      console.error('[GameSession] Initialization error:', error);
      throw error;
    }
  }, [fleetId, gameMode, environmentConfig]);

  /**
   * Handle optimized state updates with prediction
   */
  const handleStateUpdate = useCallback(async (update: GameStateUpdate) => {
    const now = performance.now();
    const timeSinceLastUpdate = now - lastUpdate.current;

    // Apply state prediction if update latency is high
    if (timeSinceLastUpdate > PREDICTION_THRESHOLD_MS) {
      const predictedState = predictGameState(gameState.current, update);
      gameState.current = automerge.change(gameState.current, 'predict', doc => {
        Object.assign(doc, predictedState);
      });
    }

    // Batch updates using requestAnimationFrame
    if (!animationFrameId.current) {
      animationFrameId.current = requestAnimationFrame(async () => {
        try {
          // Apply CRDT update
          gameState.current = automerge.change(gameState.current, 'update', doc => {
            update.playerUpdates.forEach(playerUpdate => {
              const player = doc.players.find(p => p.playerId === playerUpdate.playerId);
              if (player) {
                Object.assign(player, playerUpdate);
              }
            });
          });

          // Sync with server
          await gameApi.updateGameState({
            sessionId: session!.id,
            timestamp: now,
            playerUpdates: update.playerUpdates,
            environmentUpdates: update.environmentUpdates
          });

          // Update metrics
          setPerformance(prev => ({
            ...prev,
            updateLatency: performance.now() - now
          }));

          lastUpdate.current = now;
          animationFrameId.current = undefined;
        } catch (error) {
          console.error('[GameSession] Update error:', error);
        }
      });
    }
  }, [session]);

  /**
   * Predict game state for smooth updates
   */
  const predictGameState = (currentState: GameSession, update: GameStateUpdate) => {
    const predictedState = { ...currentState };
    const deltaTime = performance.now() - update.timestamp;

    update.playerUpdates.forEach(playerUpdate => {
      const player = predictedState.players.find(p => p.playerId === playerUpdate.playerId);
      if (player && playerUpdate.position) {
        // Apply velocity-based prediction
        player.position.x += player.velocity?.x * deltaTime || 0;
        player.position.y += player.velocity?.y * deltaTime || 0;
        player.position.z += player.velocity?.z * deltaTime || 0;
      }
    });

    return predictedState;
  };

  /**
   * Monitor game performance
   */
  const startPerformanceMonitoring = useCallback(() => {
    let lastFrameTime = performance.now();

    const monitorFrame = () => {
      const now = performance.now();
      frameCount.current++;

      // Calculate FPS every second
      if (now - lastFrameTime >= 1000) {
        setPerformance(prev => ({
          ...prev,
          fps: frameCount.current
        }));
        frameCount.current = 0;
        lastFrameTime = now;
      }

      requestAnimationFrame(monitorFrame);
    };

    requestAnimationFrame(monitorFrame);
  }, []);

  /**
   * Cleanup resources on unmount
   */
  useEffect(() => {
    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, []);

  // Initialize session on mount
  useEffect(() => {
    initializeSession();
  }, [initializeSession]);

  return {
    session,
    performance,
    updateGameState: handleStateUpdate,
    gameState: gameState.current
  };
};