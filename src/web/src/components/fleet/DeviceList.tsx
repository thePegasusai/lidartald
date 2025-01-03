import React, { useEffect, useMemo, useCallback } from 'react';
import { useDispatch, useSelector } from 'react-redux'; // v8.1.0
import {
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Typography,
  Chip,
  Tooltip,
  Box,
  useTheme
} from '@mui/material'; // v5.13.0
import {
  DeviceHub,
  SignalCellular4Bar,
  SignalCellularAlt,
  SignalCellularConnectedNoInternet0Bar,
  Speed
} from '@mui/icons-material'; // v5.13.0

import { Fleet, FleetDevice, FleetStatus, DeviceCapabilities } from '../../types/fleet.types';
import { setFleetStatus, updateSyncProgress } from '../../store/slices/fleetSlice';
import { useWebSocket } from '../../hooks/useWebSocket';
import { FLEET_CONSTANTS } from '../../config/constants';

// Constants for performance optimization
const REFRESH_INTERVAL = 1000; // 1 second refresh rate
const MAX_DEVICES = 32; // Maximum devices in fleet
const LATENCY_THRESHOLD = 50; // 50ms latency threshold

interface DeviceListProps {
  fleet: Fleet;
  onDeviceSelect?: (device: FleetDevice) => void;
}

/**
 * Enhanced device list component with real-time updates and performance optimizations
 * Supports up to 32 connected devices with <50ms latency
 */
const DeviceList: React.FC<DeviceListProps> = React.memo(({ fleet, onDeviceSelect }) => {
  const dispatch = useDispatch();
  const theme = useTheme();

  // WebSocket connection for real-time updates
  const { connected, sendMessage, latency } = useWebSocket({
    url: `${process.env.REACT_APP_WS_URL}/fleet/${fleet.id}`,
    reconnectAttempts: 5,
    heartbeatInterval: 30000,
    compressionEnabled: true,
    latencyThreshold: LATENCY_THRESHOLD
  });

  // Memoized sorted device list
  const sortedDevices = useMemo(() => {
    return [...fleet.devices].sort((a, b) => {
      // Sort by status first
      if (a.status !== b.status) {
        return a.status === FleetStatus.ACTIVE ? -1 : 1;
      }
      // Then by latency
      return a.networkLatency - b.networkLatency;
    });
  }, [fleet.devices]);

  // Device selection handler with debounce
  const handleDeviceSelect = useCallback((device: FleetDevice) => {
    if (onDeviceSelect) {
      onDeviceSelect(device);
    }
  }, [onDeviceSelect]);

  // Update fleet status based on device states
  useEffect(() => {
    const activeDevices = fleet.devices.filter(d => d.status === FleetStatus.ACTIVE);
    const syncingDevices = fleet.devices.filter(d => d.status === FleetStatus.SYNCING);
    
    if (syncingDevices.length > 0) {
      dispatch(setFleetStatus(FleetStatus.SYNCING));
      const progress = (activeDevices.length / fleet.devices.length) * 100;
      dispatch(updateSyncProgress(progress));
    } else if (activeDevices.length === fleet.devices.length) {
      dispatch(setFleetStatus(FleetStatus.ACTIVE));
      dispatch(updateSyncProgress(100));
    }
  }, [fleet.devices, dispatch]);

  /**
   * Renders device status icon based on connection quality
   */
  const getDeviceStatusIcon = (status: FleetStatus, latency: number) => {
    if (status === FleetStatus.DISCONNECTED) {
      return (
        <Tooltip title="Disconnected">
          <SignalCellularConnectedNoInternet0Bar color="error" />
        </Tooltip>
      );
    }

    if (latency > LATENCY_THRESHOLD) {
      return (
        <Tooltip title={`High Latency: ${latency}ms`}>
          <SignalCellularAlt color="warning" />
        </Tooltip>
      );
    }

    return (
      <Tooltip title={`Connected: ${latency}ms`}>
        <SignalCellular4Bar color="success" />
      </Tooltip>
    );
  };

  /**
   * Renders device capabilities with performance metrics
   */
  const renderDeviceCapabilities = (capabilities: DeviceCapabilities) => (
    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
      <Tooltip title="Scan Rate">
        <Chip
          size="small"
          icon={<Speed />}
          label={`${capabilities.scanRate}Hz`}
          variant="outlined"
        />
      </Tooltip>
      <Tooltip title="Range">
        <Chip
          size="small"
          label={`${capabilities.scanRange}m`}
          variant="outlined"
        />
      </Tooltip>
      <Tooltip title="Resolution">
        <Chip
          size="small"
          label={`${capabilities.lidarResolution}cm`}
          variant="outlined"
        />
      </Tooltip>
    </Box>
  );

  return (
    <Box
      sx={{
        maxHeight: '400px',
        overflowY: 'auto',
        willChange: 'transform',
        transform: 'translateZ(0)', // GPU acceleration
      }}
    >
      <List>
        {sortedDevices.map((device) => (
          <ListItem
            key={device.deviceId}
            onClick={() => handleDeviceSelect(device)}
            sx={{
              cursor: 'pointer',
              transition: 'background-color 0.2s',
              '&:hover': {
                backgroundColor: theme.palette.action.hover,
              },
            }}
          >
            <ListItemIcon>
              {device.deviceId === fleet.hostDeviceId ? (
                <Tooltip title="Host Device">
                  <DeviceHub color="primary" />
                </Tooltip>
              ) : (
                getDeviceStatusIcon(device.status, device.networkLatency)
              )}
            </ListItemIcon>
            <ListItemText
              primary={
                <Typography variant="subtitle1">
                  {device.deviceId === fleet.hostDeviceId ? 'Host Device' : `Device ${device.deviceId.slice(-4)}`}
                </Typography>
              }
              secondary={renderDeviceCapabilities(device.capabilities)}
            />
            <Chip
              size="small"
              label={device.status}
              color={device.status === FleetStatus.ACTIVE ? 'success' : 'warning'}
              sx={{ ml: 1 }}
            />
          </ListItem>
        ))}
      </List>
    </Box>
  );
});

DeviceList.displayName = 'DeviceList';

export default DeviceList;