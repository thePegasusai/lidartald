import React, { useEffect, useMemo, useCallback } from 'react'; // ^18.2.0
import { useSelector, useDispatch } from 'react-redux'; // ^8.1.0
import { Box, Typography, LinearProgress, useTheme } from '@mui/material'; // ^5.13.0
import styled from '@emotion/styled'; // ^11.11.0
import { Fleet, FleetStatus, FleetDevice, SyncState } from '../../types/fleet.types';
import { fleetActions } from '../../store/slices/fleetSlice';
import { Loading } from '../common/Loading';
import { FLEET_CONSTANTS } from '../../config/constants';
import { TIMING_FUNCTIONS, ANIMATION_DURATIONS } from '../../styles/animations';

// GPU-accelerated styled components
const SyncContainer = styled(Box)`
  padding: ${({ theme }) => theme.spacing(2)};
  border-radius: ${({ theme }) => theme.shape.borderRadius}px;
  background-color: ${({ theme }) => theme.palette.background.paper};
  transform: translateZ(0);
  will-change: transform;
  transition: all ${ANIMATION_DURATIONS.normal} ${TIMING_FUNCTIONS.easeInOut};
`;

const ProgressBar = styled(LinearProgress)`
  margin-top: ${({ theme }) => theme.spacing(1)};
  height: 8px;
  border-radius: 4px;
  transform: translateZ(0);
  will-change: transform;
  
  & .MuiLinearProgress-bar {
    transition: transform ${ANIMATION_DURATIONS.fast} ${TIMING_FUNCTIONS.emphasized};
  }
`;

const DeviceStatus = styled(Typography)`
  margin-top: ${({ theme }) => theme.spacing(1)};
  color: ${({ theme }) => theme.palette.text.secondary};
  font-size: 0.875rem;
`;

interface SyncStatusProps {
  fleetId: string;
}

/**
 * Calculates the overall sync progress percentage for fleet devices
 * @param devices Array of fleet devices
 * @returns Progress percentage (0-100)
 */
const calculateSyncProgress = (devices: FleetDevice[]): number => {
  if (!devices.length) return 0;

  const activeDevices = devices.filter(device => 
    device.status === FleetStatus.ACTIVE || 
    device.status === FleetStatus.SYNCING
  );

  if (!activeDevices.length) return 0;

  const totalProgress = activeDevices.reduce((acc, device) => {
    const lastSeen = new Date(device.lastSeen).getTime();
    const timeSinceUpdate = Date.now() - lastSeen;

    // Consider device out of sync if not seen within timeout period
    if (timeSinceUpdate > FLEET_CONSTANTS.CONNECTION_TIMEOUT_MS) {
      return acc;
    }

    return acc + (device.status === FleetStatus.SYNCING ? 0.5 : 1);
  }, 0);

  return Math.min(100, (totalProgress / activeDevices.length) * 100);
};

/**
 * SyncStatus component displays real-time synchronization status of fleet devices
 * with GPU-accelerated animations and accessibility support.
 */
const SyncStatus: React.FC<SyncStatusProps> = React.memo(({ fleetId }) => {
  const dispatch = useDispatch();
  const theme = useTheme();

  // Select fleet data from Redux store
  const fleet = useSelector((state: any) => 
    state.fleet.fleets.find((f: Fleet) => f.id === fleetId)
  );

  const syncProgress = useMemo(() => 
    fleet ? calculateSyncProgress(fleet.devices) : 0,
    [fleet?.devices]
  );

  // Handle sync status updates with RAF for smooth animations
  useEffect(() => {
    if (!fleet) return;

    let rafId: number;
    let lastUpdate = 0;

    const updateSync = (timestamp: number) => {
      if (timestamp - lastUpdate >= 1000 / 60) { // 60 FPS target
        dispatch(fleetActions.updateSyncProgress(syncProgress));
        lastUpdate = timestamp;
      }
      rafId = requestAnimationFrame(updateSync);
    };

    rafId = requestAnimationFrame(updateSync);

    return () => {
      cancelAnimationFrame(rafId);
    };
  }, [fleet, syncProgress, dispatch]);

  // Monitor device connections and update fleet status
  useEffect(() => {
    if (!fleet) return;

    const checkDevices = () => {
      const activeDevices = fleet.devices.filter(device => 
        Date.now() - new Date(device.lastSeen).getTime() < FLEET_CONSTANTS.CONNECTION_TIMEOUT_MS
      );

      if (activeDevices.length !== fleet.devices.length) {
        dispatch(fleetActions.setFleetStatus(FleetStatus.SYNCING));
      } else if (syncProgress === 100) {
        dispatch(fleetActions.setFleetStatus(FleetStatus.ACTIVE));
      }
    };

    const intervalId = setInterval(checkDevices, FLEET_CONSTANTS.HEARTBEAT_INTERVAL_MS);

    return () => {
      clearInterval(intervalId);
    };
  }, [fleet, syncProgress, dispatch]);

  if (!fleet) {
    return <Loading size="medium" ariaLabel="Loading fleet status..." />;
  }

  const connectedDevices = fleet.devices.filter(device => 
    device.status !== FleetStatus.DISCONNECTED
  ).length;

  return (
    <SyncContainer
      role="region"
      aria-label="Fleet synchronization status"
      elevation={1}
    >
      <Typography variant="h6" component="h2" gutterBottom>
        Fleet Sync Status
      </Typography>
      
      <ProgressBar
        variant="determinate"
        value={syncProgress}
        aria-label="Synchronization progress"
        sx={{
          '& .MuiLinearProgress-bar': {
            backgroundColor: theme.palette.primary.main
          }
        }}
      />

      <DeviceStatus>
        {`${connectedDevices}/${FLEET_CONSTANTS.MAX_DEVICES} devices connected`}
      </DeviceStatus>

      <Typography 
        variant="body2" 
        color="textSecondary"
        sx={{ mt: 1 }}
      >
        {fleet.status === FleetStatus.SYNCING
          ? 'Synchronizing fleet data...'
          : 'Fleet fully synchronized'}
      </Typography>

      {fleet.lastSyncTimestamp && (
        <Typography 
          variant="caption" 
          color="textSecondary"
          sx={{ mt: 0.5, display: 'block' }}
        >
          {`Last sync: ${new Date(fleet.lastSyncTimestamp).toLocaleTimeString()}`}
        </Typography>
      )}
    </SyncContainer>
  );
});

SyncStatus.displayName = 'SyncStatus';

export default SyncStatus;