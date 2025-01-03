import React, { useCallback, useEffect, useMemo } from 'react';
import { AppBar, Toolbar, IconButton, Typography } from '@mui/material'; // ^5.13.0
import { Settings, Help, AccountCircle } from '@mui/icons-material'; // ^5.13.0
import styled from '@emotion/styled'; // ^11.11.0
import TaldButton from '../common/Button';
import Navigation from './Navigation';
import { theme } from '../../styles/theme';

// Constants for header configuration
const HEADER_HEIGHT = '64px';
const HEADER_HEIGHT_LANDSCAPE = '56px';
const ANIMATION_DURATION = 300;
const MIN_CONTRAST_RATIO = 4.5;
const HAPTIC_FEEDBACK_DURATION = 50;

const STATUS_MESSAGES = {
  idle: 'Ready to Scan',
  scanning: 'Scanning Environment',
  processing: 'Processing Data',
  error: 'Scan Error',
  connected: 'Fleet Connected',
  disconnected: 'Fleet Disconnected',
} as const;

// Styled components with outdoor visibility optimization
const HeaderContainer = styled(AppBar)<{ elevation: number; scanStatus: string }>`
  width: 100%;
  position: fixed;
  top: 0;
  left: 0;
  z-index: ${({ theme }) => theme.zIndex.appBar};
  background-color: ${({ theme }) => theme.palette.background.paper};
  transition: all ${ANIMATION_DURATION}ms ease-in-out;
  backdrop-filter: blur(10px);
  box-shadow: ${({ theme, elevation }) => theme.shadows[elevation]};
  height: ${({ orientation }) =>
    orientation === 'landscape' ? HEADER_HEIGHT_LANDSCAPE : HEADER_HEIGHT};

  ${({ scanStatus, theme }) =>
    scanStatus === 'scanning' &&
    `
    &::after {
      content: '';
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      height: 2px;
      background: ${theme.palette.primary.main};
      animation: scanning 2s infinite linear;
    }
  `}

  @keyframes scanning {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }
`;

const StatusIndicator = styled.div<{ status: string }>`
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin-right: ${({ theme }) => theme.spacing(1)};
  transition: all ${ANIMATION_DURATION}ms ease-in-out;
  box-shadow: 0 0 10px rgba(0,0,0,0.2);
  background-color: ${({ status, theme }) => {
    switch (status) {
      case 'scanning':
        return theme.palette.primary.main;
      case 'processing':
        return theme.palette.warning.main;
      case 'error':
        return theme.palette.error.main;
      case 'connected':
        return theme.palette.success.main;
      default:
        return theme.palette.grey[400];
    }
  }};

  animation: ${({ status }) => {
    switch (status) {
      case 'scanning':
        return 'pulse 1.5s infinite';
      case 'processing':
        return 'rotate 2s infinite';
      case 'error':
        return 'shake 0.5s infinite';
      default:
        return 'none';
    }
  }};

  @keyframes pulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.2); opacity: 0.7; }
    100% { transform: scale(1); opacity: 1; }
  }

  @keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  @keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(2px); }
    75% { transform: translateX(-2px); }
  }
`;

const StatusText = styled(Typography)<{ highContrast: boolean }>`
  margin-left: ${({ theme }) => theme.spacing(1)};
  font-weight: ${({ highContrast }) => highContrast ? 600 : 500};
  text-shadow: ${({ highContrast }) => 
    highContrast ? '0 1px 2px rgba(0,0,0,0.3)' : 'none'};
`;

interface HeaderProps {
  className?: string;
  orientation: 'portrait' | 'landscape';
  scanStatus: 'idle' | 'scanning' | 'processing' | 'error';
  connectionStatus: 'connected' | 'disconnected';
  fleetSize: number;
  batteryLevel: number;
  signalStrength: number;
  onSettingsClick: (event: React.MouseEvent) => void;
  onHelpClick: (event: React.MouseEvent) => void;
  onProfileClick: (event: React.MouseEvent) => void;
}

const Header: React.FC<HeaderProps> = ({
  className,
  orientation,
  scanStatus,
  connectionStatus,
  fleetSize,
  batteryLevel,
  signalStrength,
  onSettingsClick,
  onHelpClick,
  onProfileClick,
}) => {
  // Calculate elevation based on scan status
  const elevation = useMemo(() => {
    switch (scanStatus) {
      case 'scanning':
        return 4;
      case 'processing':
        return 3;
      case 'error':
        return 2;
      default:
        return 1;
    }
  }, [scanStatus]);

  // Handle haptic feedback for status changes
  useEffect(() => {
    if ('vibrate' in navigator) {
      navigator.vibrate(HAPTIC_FEEDBACK_DURATION);
    }
  }, [scanStatus, connectionStatus]);

  // Determine if high contrast mode is needed based on battery and signal
  const highContrast = useMemo(() => {
    return batteryLevel < 20 || signalStrength < 30;
  }, [batteryLevel, signalStrength]);

  // Memoized status message
  const statusMessage = useMemo(() => {
    if (connectionStatus === 'disconnected') {
      return STATUS_MESSAGES.disconnected;
    }
    return STATUS_MESSAGES[scanStatus];
  }, [scanStatus, connectionStatus]);

  // Handle fleet size changes
  const handleFleetUpdate = useCallback(() => {
    if ('vibrate' in navigator) {
      navigator.vibrate([50, 100, 50]);
    }
  }, []);

  useEffect(() => {
    handleFleetUpdate();
  }, [fleetSize, handleFleetUpdate]);

  return (
    <HeaderContainer
      elevation={elevation}
      scanStatus={scanStatus}
      className={className}
      orientation={orientation}
    >
      <Toolbar>
        <StatusIndicator status={scanStatus} />
        <Typography
          variant="h6"
          component="h1"
          sx={{
            flexGrow: 1,
            fontWeight: highContrast ? 600 : 500,
            letterSpacing: '0.02em',
          }}
        >
          TALD UNIA
        </Typography>

        <StatusText
          variant="body2"
          highContrast={highContrast}
        >
          {statusMessage}
          {connectionStatus === 'connected' && fleetSize > 0 && 
            ` • Fleet: ${fleetSize}`}
        </StatusText>

        <TaldButton
          variant="text"
          size="small"
          color="primary"
          outdoorMode={highContrast}
          onClick={onHelpClick}
          aria-label="Help"
        >
          <Help />
        </TaldButton>

        <IconButton
          size="large"
          edge="start"
          color="inherit"
          aria-label="settings"
          onClick={onSettingsClick}
          sx={{ ml: 1 }}
        >
          <Settings />
        </IconButton>

        <IconButton
          size="large"
          edge="end"
          color="inherit"
          aria-label="profile"
          onClick={onProfileClick}
          sx={{ ml: 1 }}
        >
          <AccountCircle />
        </IconButton>
      </Toolbar>

      <Navigation
        orientation={orientation}
        highContrast={highContrast}
        fleetSyncEnabled={connectionStatus === 'connected'}
        hapticFeedback={true}
        onThemeChange={() => {}}
      />
    </HeaderContainer>
  );
};

export default React.memo(Header);