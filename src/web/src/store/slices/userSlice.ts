import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { User, UserProfile, UserPrivacySettings } from '../../types/user.types';
import { userApi } from '../../api/userApi';

// State interface with comprehensive user data management
interface UserState {
  user: User | null;
  profile: UserProfile | null;
  privacySettings: UserPrivacySettings;
  rolePermissions: Record<string, boolean>;
  fleetStatus: string | null;
  loading: boolean;
  error: string | null;
  lastActive: number | null;
}

// Initial state with default privacy settings
const initialState: UserState = {
  user: null,
  profile: null,
  privacySettings: {
    shareLocation: false,
    shareActivity: false,
    shareFleetHistory: false,
    dataRetentionDays: 7
  },
  rolePermissions: {},
  fleetStatus: null,
  loading: false,
  error: null,
  lastActive: null
};

// Async thunk for fetching current user with enhanced security
export const fetchCurrentUser = createAsyncThunk(
  'user/fetchCurrentUser',
  async (_, { rejectWithValue }) => {
    try {
      const user = await userApi.getCurrentUser();
      return user;
    } catch (error) {
      return rejectWithValue('Failed to fetch user data');
    }
  }
);

// Async thunk for fetching user profile with fleet integration
export const fetchUserProfile = createAsyncThunk(
  'user/fetchUserProfile',
  async (userId: string, { rejectWithValue }) => {
    try {
      const profile = await userApi.getUserProfile(userId);
      return profile;
    } catch (error) {
      return rejectWithValue('Failed to fetch user profile');
    }
  }
);

// Async thunk for updating privacy settings
export const updatePrivacySettings = createAsyncThunk(
  'user/updatePrivacySettings',
  async (settings: UserPrivacySettings, { rejectWithValue }) => {
    try {
      const updatedUser = await userApi.updatePrivacySettings(settings);
      return updatedUser;
    } catch (error) {
      return rejectWithValue('Failed to update privacy settings');
    }
  }
);

// User slice with comprehensive state management
const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    setLastActive: (state) => {
      state.lastActive = Date.now();
    },
    updateFleetStatus: (state, action: PayloadAction<string>) => {
      state.fleetStatus = action.payload;
    },
    clearUserState: (state) => {
      return initialState;
    },
    updateRolePermissions: (state, action: PayloadAction<Record<string, boolean>>) => {
      state.rolePermissions = action.payload;
    }
  },
  extraReducers: (builder) => {
    // Fetch current user reducers
    builder
      .addCase(fetchCurrentUser.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchCurrentUser.fulfilled, (state, action) => {
        state.loading = false;
        state.user = action.payload;
        state.privacySettings = action.payload.privacySettings;
        state.lastActive = Date.now();
      })
      .addCase(fetchCurrentUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      })

    // Fetch user profile reducers
      .addCase(fetchUserProfile.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchUserProfile.fulfilled, (state, action) => {
        state.loading = false;
        state.profile = action.payload;
        state.fleetStatus = action.payload.fleetStatus;
      })
      .addCase(fetchUserProfile.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      })

    // Update privacy settings reducers
      .addCase(updatePrivacySettings.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(updatePrivacySettings.fulfilled, (state, action) => {
        state.loading = false;
        state.user = action.payload;
        state.privacySettings = action.payload.privacySettings;
      })
      .addCase(updatePrivacySettings.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });
  }
});

// Export actions
export const {
  setLastActive,
  updateFleetStatus,
  clearUserState,
  updateRolePermissions
} = userSlice.actions;

// Memoized selectors
export const selectUser = (state: { user: UserState }) => state.user.user;
export const selectUserProfile = (state: { user: UserState }) => state.user.profile;
export const selectPrivacySettings = (state: { user: UserState }) => state.user.privacySettings;
export const selectRolePermissions = (state: { user: UserState }) => state.user.rolePermissions;
export const selectFleetStatus = (state: { user: UserState }) => state.user.fleetStatus;
export const selectUserLoading = (state: { user: UserState }) => state.user.loading;
export const selectUserError = (state: { user: UserState }) => state.user.error;
export const selectLastActive = (state: { user: UserState }) => state.user.lastActive;

// Export reducer
export default userSlice.reducer;