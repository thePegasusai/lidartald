import React from 'react';
import { Box, Typography, useTheme } from '@mui/material'; // v5.13.0
import { FleetStatus } from '../../types/fleet.types';
import { ScanParameters } from '../../types/lidar.types';
import { THEME_COLORS } from '../../styles/theme';

interface StatusBarProps {
  fleetStatus: FleetStatus;
  scanParameters: ScanParameters;
  networkLatency: number;
}

/**
 * Formats scan rate with Hz unit
 * @param rate - Scan rate in Hz
 */
const formatScanRate = (rate: number): string => {
  return `${Math.round(rate)}Hz`;
};

/**
 * Formats range with meter unit and one decimal place
 * @param range - Range in meters
 */
const formatRange = (range: number): string => {
  return `${range.toFixed(1)}m`;
};

/**
 * StatusBar component displays real-time system metrics and status information
 * with high-contrast styling for outdoor visibility.
 */
const StatusBar: React.FC<StatusBarProps> = React.memo(({ 
  fleetStatus, 
  scanParameters, 
  networkLatency 
}) => {
  const theme = useTheme();

  /**
   * Determines status indicator color based on fleet state
   */
  const getStatusColor = (status: FleetStatus): string => {
    switch (status) {
      case FleetStatus.ACTIVE:
        return theme.palette.primary.main;
      case FleetStatus.SYNCING:
        return theme.palette.secondary.main;
      default:
        return theme.palette.warning.main;
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: theme.spacing(1, 2),
        backgroundColor: theme.palette.background.paper,
        borderTop: `1px solid ${theme.palette.divider}`,
        minHeight: '48px',
        // High contrast styles for outdoor visibility
        '& .MuiTypography-root': {
          fontWeight: 500,
          textShadow: theme.palette.mode === 'light' 
            ? '0 0 1px rgba(0,0,0,0.2)'
            : '0 0 1px rgba(255,255,255,0.2)'
        }
      }}
    >
      {/* Fleet Status Indicator */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Box
          sx={{
            width: 12,
            height: 12,
            borderRadius: '50%',
            backgroundColor: getStatusColor(fleetStatus),
            boxShadow: `0 0 4px ${getStatusColor(fleetStatus)}`,
          }}
        />
        <Typography variant="body2">
          {`Fleet: ${fleetStatus}`}
        </Typography>
      </Box>

      {/* LiDAR Metrics */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Typography variant="body2">
          {`Scan: ${formatScanRate(scanParameters.scanRate)}`}
        </Typography>
        <Typography variant="body2">
          {`Range: ${formatRange(scanParameters.range)}`}
        </Typography>
        <Typography variant="body2">
          {`Res: ${scanParameters.resolution}cm`}
        </Typography>
      </Box>

      {/* Network Latency */}
      <Typography 
        variant="body2"
        sx={{
          color: networkLatency > 50 
            ? theme.palette.warning.main 
            : theme.palette.success.main
        }}
      >
        {`Latency: ${networkLatency}ms`}
      </Typography>
    </Box>
  );
});

StatusBar.displayName = 'StatusBar';

export default StatusBar;