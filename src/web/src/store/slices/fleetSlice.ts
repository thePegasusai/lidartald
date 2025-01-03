import { createSlice, PayloadAction, createSelector, createAsyncThunk } from '@reduxjs/toolkit'; // v1.9.5
import { 
    Fleet, 
    FleetStatus, 
    FleetDevice, 
    FleetMember, 
    CreateFleetDTO, 
    UpdateFleetDTO, 
    FleetEnvironmentSync, 
    FleetMeshStatus, 
    FleetValidationError, 
    FleetProximityData 
} from '../../types/fleet.types';
import { 
    createFleet, 
    joinFleet, 
    leaveFleet, 
    getFleetState, 
    syncFleetState, 
    getNearbyFleets, 
    updateFleetEnvironment, 
    validateFleetOperation 
} from '../../api/fleetApi';
import { FLEET_CONSTANTS } from '../../config/constants';

// State interface
interface FleetState {
    currentFleet: Fleet | null;
    nearbyFleets: Fleet[];
    fleetStatus: FleetStatus;
    syncState: {
        lastSync: number;
        pendingUpdates: number;
        syncProgress: number;
    };
    meshNetwork: {
        connected: boolean;
        latency: number;
        activeConnections: number;
    };
    error: string | null;
    loading: boolean;
}

// Initial state
const initialState: FleetState = {
    currentFleet: null,
    nearbyFleets: [],
    fleetStatus: FleetStatus.DISCONNECTED,
    syncState: {
        lastSync: 0,
        pendingUpdates: 0,
        syncProgress: 0
    },
    meshNetwork: {
        connected: false,
        latency: 0,
        activeConnections: 0
    },
    error: null,
    loading: false
};

// Async thunks
export const createFleetThunk = createAsyncThunk<Fleet, CreateFleetDTO>(
    'fleet/create',
    async (fleetData, { rejectWithValue }) => {
        try {
            return await createFleet(fleetData);
        } catch (error) {
            return rejectWithValue((error as Error).message);
        }
    }
);

export const joinFleetThunk = createAsyncThunk<Fleet, string>(
    'fleet/join',
    async (fleetId, { rejectWithValue }) => {
        try {
            return await joinFleet(fleetId);
        } catch (error) {
            return rejectWithValue((error as Error).message);
        }
    }
);

export const syncFleetEnvironmentThunk = createAsyncThunk<void, FleetEnvironmentSync>(
    'fleet/syncEnvironment',
    async (environmentData, { getState, rejectWithValue }) => {
        try {
            const { currentFleet } = (getState() as { fleet: FleetState }).fleet;
            if (!currentFleet) throw new Error('No active fleet');
            
            await updateFleetEnvironment(currentFleet.id, environmentData);
        } catch (error) {
            return rejectWithValue((error as Error).message);
        }
    }
);

export const updateFleetProximityThunk = createAsyncThunk<void, FleetProximityData>(
    'fleet/updateProximity',
    async (proximityData, { getState, rejectWithValue }) => {
        try {
            const { currentFleet } = (getState() as { fleet: FleetState }).fleet;
            if (!currentFleet) throw new Error('No active fleet');
            
            await syncFleetState(currentFleet.id);
        } catch (error) {
            return rejectWithValue((error as Error).message);
        }
    }
);

// Slice definition
const fleetSlice = createSlice({
    name: 'fleet',
    initialState,
    reducers: {
        setFleetStatus(state, action: PayloadAction<FleetStatus>) {
            state.fleetStatus = action.payload;
        },
        updateSyncProgress(state, action: PayloadAction<number>) {
            state.syncState.syncProgress = action.payload;
        },
        updateMeshNetwork(state, action: PayloadAction<Partial<FleetState['meshNetwork']>>) {
            state.meshNetwork = { ...state.meshNetwork, ...action.payload };
        },
        clearFleetError(state) {
            state.error = null;
        }
    },
    extraReducers: (builder) => {
        builder
            // Create Fleet
            .addCase(createFleetThunk.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(createFleetThunk.fulfilled, (state, action) => {
                state.currentFleet = action.payload;
                state.fleetStatus = FleetStatus.ACTIVE;
                state.loading = false;
            })
            .addCase(createFleetThunk.rejected, (state, action) => {
                state.error = action.payload as string;
                state.loading = false;
            })
            // Join Fleet
            .addCase(joinFleetThunk.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(joinFleetThunk.fulfilled, (state, action) => {
                state.currentFleet = action.payload;
                state.fleetStatus = FleetStatus.ACTIVE;
                state.loading = false;
            })
            .addCase(joinFleetThunk.rejected, (state, action) => {
                state.error = action.payload as string;
                state.loading = false;
            })
            // Sync Environment
            .addCase(syncFleetEnvironmentThunk.pending, (state) => {
                state.fleetStatus = FleetStatus.SYNCING;
                state.syncState.pendingUpdates++;
            })
            .addCase(syncFleetEnvironmentThunk.fulfilled, (state) => {
                state.syncState.lastSync = Date.now();
                state.syncState.pendingUpdates = Math.max(0, state.syncState.pendingUpdates - 1);
                if (state.syncState.pendingUpdates === 0) {
                    state.fleetStatus = FleetStatus.ACTIVE;
                }
            })
            .addCase(syncFleetEnvironmentThunk.rejected, (state, action) => {
                state.error = action.payload as string;
                state.fleetStatus = FleetStatus.ACTIVE;
            });
    }
});

// Selectors
export const selectCurrentFleet = (state: { fleet: FleetState }) => state.fleet.currentFleet;
export const selectFleetStatus = (state: { fleet: FleetState }) => state.fleet.fleetStatus;
export const selectMeshNetwork = (state: { fleet: FleetState }) => state.fleet.meshNetwork;

export const selectActiveMemberCount = createSelector(
    [selectCurrentFleet],
    (fleet) => fleet?.members.length ?? 0
);

export const selectFleetSyncStatus = createSelector(
    [(state: { fleet: FleetState }) => state.fleet.syncState],
    (syncState) => ({
        isFullySynced: syncState.pendingUpdates === 0,
        lastSyncTime: syncState.lastSync,
        syncProgress: syncState.syncProgress
    })
);

// Export actions and reducer
export const { 
    setFleetStatus, 
    updateSyncProgress, 
    updateMeshNetwork, 
    clearFleetError 
} = fleetSlice.actions;

export default fleetSlice.reducer;