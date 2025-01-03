import React, { useMemo, useCallback, useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { debounce } from 'lodash'; // ^4.17.21
import {
  Container,
  Grid,
  Typography,
  Tooltip,
  Skeleton,
  Switch,
  FormControlLabel,
  Alert,
  Paper,
  Box
} from '@mui/material'; // ^5.13.0
import TaldButton from '../../components/common/Button';
import Select from '../../components/common/Select';
import { userSlice } from '../../store/slices/userSlice';
import { LIDAR_CONSTANTS, UI_CONSTANTS } from '../../config/constants';
import { UserProfile, UserRole } from '../../types/user.types';
import { useTheme } from '@mui/material';

// Resolution options based on LiDAR hardware capabilities
const RESOLUTION_OPTIONS = [
  {
    value: LIDAR_CONSTANTS.MIN_RESOLUTION_CM,
    label: 'Ultra High (0.01cm)',
    tooltip: 'Maximum detail for close-range scanning'
  },
  {
    value: 0.05,
    label: 'High (0.05cm)',
    tooltip: 'Balanced performance and detail'
  },
  {
    value: 0.1,
    label: 'Standard (0.1cm)',
    tooltip: 'Optimal for battery life'
  }
];

// Range options for LiDAR scanning
const RANGE_OPTIONS = [
  {
    value: LIDAR_CONSTANTS.MAX_RANGE_M,
    label: '5 meters',
    tooltip: 'Maximum range for open environments'
  },
  {
    value: 3,
    label: '3 meters',
    tooltip: 'Balanced range for most uses'
  },
  {
    value: 1,
    label: '1 meter',
    tooltip: 'Precise scanning for small objects'
  }
];

// Theme options with outdoor visibility support
const THEME_OPTIONS = [
  {
    value: 'auto',
    label: 'Auto (Based on ambient light)',
    tooltip: 'Automatically adjusts based on environment'
  },
  {
    value: 'light',
    label: 'Light',
    tooltip: 'Optimized for indoor use'
  },
  {
    value: 'dark',
    label: 'Dark',
    tooltip: 'Enhanced for LiDAR visualization'
  },
  {
    value: 'high-contrast',
    label: 'High Contrast (Outdoor)',
    tooltip: 'Maximum visibility in bright sunlight'
  }
];

// Debounce delay for preference updates
const PREFERENCE_UPDATE_DEBOUNCE = 300;

interface SettingsProps {
  className?: string;
}

const Settings: React.FC<SettingsProps> = React.memo(({ className }) => {
  const dispatch = useDispatch();
  const theme = useTheme();
  const userProfile = useSelector(userSlice.selectUserProfile);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outdoorMode, setOutdoorMode] = useState(false);

  // Detect outdoor mode based on ambient light
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: light)');
    const updateOutdoorMode = (e: MediaQueryListEvent) => {
      setOutdoorMode(e.matches && theme.palette.mode === 'light');
    };
    mediaQuery.addEventListener('change', updateOutdoorMode);
    return () => mediaQuery.removeEventListener('change', updateOutdoorMode);
  }, [theme]);

  // Memoized validation function
  const validatePreference = useCallback((key: string, value: any) => {
    switch (key) {
      case 'scanResolution':
        return value >= LIDAR_CONSTANTS.VALIDATION_RANGES.MIN_RESOLUTION_CM[0] &&
               value <= LIDAR_CONSTANTS.VALIDATION_RANGES.MIN_RESOLUTION_CM[1];
      case 'scanRange':
        return value >= LIDAR_CONSTANTS.VALIDATION_RANGES.MAX_RANGE_M[0] &&
               value <= LIDAR_CONSTANTS.VALIDATION_RANGES.MAX_RANGE_M[1];
      default:
        return true;
    }
  }, []);

  // Debounced preference update handler
  const updatePreference = useMemo(
    () =>
      debounce(async (key: string, value: any) => {
        try {
          setLoading(true);
          setError(null);

          if (!validatePreference(key, value)) {
            throw new Error(`Invalid ${key} value`);
          }

          await dispatch(
            userSlice.updatePreferences({
              ...userProfile?.preferences,
              [key]: value
            })
          );
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to update preference');
        } finally {
          setLoading(false);
        }
      }, PREFERENCE_UPDATE_DEBOUNCE),
    [dispatch, userProfile, validatePreference]
  );

  // Handle scan resolution change
  const handleScanResolutionChange = useCallback(
    (event: React.ChangeEvent<{ value: unknown }>) => {
      const value = event.target.value as number;
      updatePreference('scanResolution', value);
    },
    [updatePreference]
  );

  // Handle scan range change
  const handleScanRangeChange = useCallback(
    (event: React.ChangeEvent<{ value: unknown }>) => {
      const value = event.target.value as number;
      updatePreference('scanRange', value);
    },
    [updatePreference]
  );

  // Handle theme change
  const handleThemeChange = useCallback(
    (event: React.ChangeEvent<{ value: unknown }>) => {
      const value = event.target.value as string;
      updatePreference('theme', value);
    },
    [updatePreference]
  );

  // Handle accessibility toggle
  const handleAccessibilityChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const { name, checked } = event.target;
      updatePreference(`accessibility.${name}`, checked);
    },
    [updatePreference]
  );

  if (!userProfile) {
    return (
      <Container maxWidth="md" className={className}>
        <Skeleton variant="rectangular" height={400} />
      </Container>
    );
  }

  return (
    <Container maxWidth="md" className={className}>
      <Paper elevation={outdoorMode ? 4 : 2} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Settings
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Grid container spacing={3}>
          {/* LiDAR Settings Section */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              LiDAR Configuration
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Tooltip title="Adjust scanning resolution for detail vs. performance">
                  <Select
                    name="scanResolution"
                    label="Scan Resolution"
                    value={userProfile.preferences.scanResolution}
                    onChange={handleScanResolutionChange}
                    options={RESOLUTION_OPTIONS}
                    disabled={loading}
                    outdoorMode={outdoorMode}
                    fullWidth
                  />
                </Tooltip>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Tooltip title="Set maximum scanning range">
                  <Select
                    name="scanRange"
                    label="Scan Range"
                    value={userProfile.preferences.scanRange}
                    onChange={handleScanRangeChange}
                    options={RANGE_OPTIONS}
                    disabled={loading}
                    outdoorMode={outdoorMode}
                    fullWidth
                  />
                </Tooltip>
              </Grid>
            </Grid>
          </Grid>

          {/* Display Settings Section */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Display Settings
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Tooltip title="Choose theme based on environment">
                  <Select
                    name="theme"
                    label="Theme"
                    value={userProfile.preferences.theme}
                    onChange={handleThemeChange}
                    options={THEME_OPTIONS}
                    disabled={loading}
                    outdoorMode={outdoorMode}
                    fullWidth
                  />
                </Tooltip>
              </Grid>
            </Grid>
          </Grid>

          {/* Accessibility Settings Section */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Accessibility
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={userProfile.preferences.accessibility.highContrast}
                      onChange={handleAccessibilityChange}
                      name="highContrast"
                      disabled={loading}
                    />
                  }
                  label="High Contrast Mode"
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={userProfile.preferences.accessibility.reducedMotion}
                      onChange={handleAccessibilityChange}
                      name="reducedMotion"
                      disabled={loading}
                    />
                  }
                  label="Reduced Motion"
                />
              </Grid>
            </Grid>
          </Grid>

          {/* Action Buttons */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
              <TaldButton
                variant="outlined"
                color="primary"
                onClick={() => updatePreference('reset', true)}
                disabled={loading}
                outdoorMode={outdoorMode}
              >
                Reset to Defaults
              </TaldButton>
              <TaldButton
                variant="contained"
                color="primary"
                onClick={() => null}
                disabled={loading}
                outdoorMode={outdoorMode}
              >
                Save Changes
              </TaldButton>
            </Box>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
});

Settings.displayName = 'Settings';

export default Settings;