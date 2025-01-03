import { createSlice, PayloadAction, createSelector } from '@reduxjs/toolkit'; // v1.9.5
import * as THREE from 'three'; // v0.150.0
import { 
    GameSession, 
    GameMode, 
    GameStatus, 
    PlayerState, 
    GameStateUpdate, 
    GameConfig, 
    EnvironmentState 
} from '../../types/game.types';
import { 
    initializeGameState, 
    updatePlayerState, 
    reconcileGameState, 
    validateStateUpdate 
} from '../../utils/gameState';

// Performance optimization constants
const STATE_UPDATE_INTERVAL_MS = 16; // 60 FPS state updates
const MAX_PLAYERS = 32;
const ENVIRONMENT_UPDATE_THRESHOLD_CM = 0.01;
const STATE_BATCH_SIZE = 10;

// Initial state interface
interface GameState {
    currentSession: GameSession | null;
    playerStates: Record<string, PlayerState>;
    environmentState: EnvironmentState | null;
    pendingUpdates: GameStateUpdate[];
    performance: {
        fps: number;
        updateLatency: number;
        lastUpdateTimestamp: number;
    };
    error: string | null;
}

// Initial state
const initialState: GameState = {
    currentSession: null,
    playerStates: {},
    environmentState: null,
    pendingUpdates: [],
    performance: {
        fps: 0,
        updateLatency: 0,
        lastUpdateTimestamp: 0
    },
    error: null
};

// Create the game slice
const gameSlice = createSlice({
    name: 'game',
    initialState,
    reducers: {
        // Initialize new game session
        initSession: (state, action: PayloadAction<GameSession>) => {
            state.currentSession = action.payload;
            state.playerStates = action.payload.players.reduce((acc, player) => ({
                ...acc,
                [player.playerId]: player
            }), {});
            state.pendingUpdates = [];
            state.error = null;
        },

        // Update environment state with validation
        updateEnvironment: (state, action: PayloadAction<EnvironmentState>) => {
            const { payload } = action;
            if (Math.abs(payload.resolution - (state.environmentState?.resolution || 0)) >= ENVIRONMENT_UPDATE_THRESHOLD_CM) {
                state.environmentState = payload;
            }
        },

        // Batch update player states with CRDT reconciliation
        updatePlayers: (state, action: PayloadAction<PlayerState[]>) => {
            action.payload.forEach(playerState => {
                state.playerStates[playerState.playerId] = {
                    ...state.playerStates[playerState.playerId],
                    ...playerState,
                    position: new THREE.Vector3().copy(playerState.position),
                    rotation: new THREE.Quaternion().copy(playerState.rotation)
                };
            });
        },

        // Queue state update for batch processing
        queueUpdate: (state, action: PayloadAction<GameStateUpdate>) => {
            state.pendingUpdates.push(action.payload);
            if (state.pendingUpdates.length >= STATE_BATCH_SIZE) {
                state.pendingUpdates = state.pendingUpdates.slice(-STATE_BATCH_SIZE);
            }
        },

        // Update performance metrics
        updatePerformance: (state, action: PayloadAction<Partial<GameState['performance']>>) => {
            state.performance = {
                ...state.performance,
                ...action.payload,
                lastUpdateTimestamp: Date.now()
            };
        },

        // Handle game session status changes
        updateSessionStatus: (state, action: PayloadAction<GameStatus>) => {
            if (state.currentSession) {
                state.currentSession.status = action.payload;
            }
        },

        // Set error state
        setError: (state, action: PayloadAction<string>) => {
            state.error = action.payload;
        },

        // Reset game state
        resetState: (state) => {
            Object.assign(state, initialState);
        }
    }
});

// Memoized selectors for optimal performance
export const selectCurrentSession = createSelector(
    [(state: { game: GameState }) => state.game.currentSession],
    (session) => session
);

export const selectPlayerStates = createSelector(
    [(state: { game: GameState }) => state.game.playerStates],
    (playerStates) => Object.values(playerStates)
);

export const selectEnvironmentState = createSelector(
    [(state: { game: GameState }) => state.game.environmentState],
    (environment) => environment
);

export const selectGamePerformance = createSelector(
    [(state: { game: GameState }) => state.game.performance],
    (performance) => performance
);

export const selectActivePlayerCount = createSelector(
    [selectPlayerStates],
    (players) => players.filter(player => player.status === 'active').length
);

export const selectIsGameActive = createSelector(
    [selectCurrentSession],
    (session) => session?.status === GameStatus.IN_PROGRESS
);

// Export actions and reducer
export const { 
    initSession,
    updateEnvironment,
    updatePlayers,
    queueUpdate,
    updatePerformance,
    updateSessionStatus,
    setError,
    resetState
} = gameSlice.actions;

export default gameSlice.reducer;