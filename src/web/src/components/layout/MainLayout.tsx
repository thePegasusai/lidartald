import React, { useCallback, useMemo, useState } from 'react';
import { Box, Container, useTheme, useMediaQuery } from '@mui/material'; // ^5.13.0
import styled from '@emotion/styled'; // ^11.11.0
import Header from './Header';
import StatusBar from './StatusBar';
import { theme } from '../../styles/theme';

// Constants for layout configuration
const HEADER_HEIGHT = '64px';
const STATUS_BAR_HEIGHT = '48px';
const CONTENT_MAX_WIDTH = '1200px';
const ANIMATION_DURATION = '300ms';
const ELEVATION_LEVELS = {
  lidar: 24,
  content: 1,
  status: 4
};

// Interface for performance configuration
interface PerformanceConfig {
  enableHardwareAcceleration: boolean;
  optimizeReflows: boolean;
  enableTouchOptimization: boolean;
}

// Props interface for MainLayout
interface MainLayoutProps {
  children: React.ReactNode;
  className?: string;
  fleetSync?: boolean;
  highContrast?: boolean;
  performance?: PerformanceConfig;
}

// Styled components with hardware acceleration and outdoor visibility optimizations
const LayoutContainer = styled(Box)<{ highContrast: boolean }>`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: ${({ theme }) => theme.palette.background.default};
  transform: translateZ(0);
  backface-visibility: hidden;
  perspective: 1000;
  transition: all ${ANIMATION_DURATION} cubic-bezier(0.4, 0, 0.2, 1);

  ${({ highContrast }) =>
    highContrast &&
    `
    background-color: ${theme.palette.background.paper};
    color: ${theme.palette.text.primary};
  `}
`;

const MainContent = styled(Container)<{ orientation: 'portrait' | 'landscape' }>`
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: ${({ theme }) => theme.spacing(2)};
  margin-top: ${HEADER_HEIGHT};
  margin-bottom: ${STATUS_BAR_HEIGHT};
  will-change: transform;
  transform: translateZ(0);
  touch-action: manipulation;

  ${({ orientation }) =>
    orientation === 'landscape' &&
    `
    padding-right: 80px; // Account for side navigation in landscape
  `}
`;

// Memoized MainLayout component
const MainLayout: React.FC<MainLayoutProps> = React.memo(({
  children,
  className,
  fleetSync = false,
  highContrast = false,
  performance = {
    enableHardwareAcceleration: true,
    optimizeReflows: true,
    enableTouchOptimization: true
  }
}) => {
  const theme = useTheme();
  const isLandscape = useMediaQuery(theme.breakpoints.up('md'));
  const [scanStatus, setScanStatus] = useState<'idle' | 'scanning' | 'processing' | 'error'>('idle');
  const [fleetSize, setFleetSize] = useState(0);

  // Memoized performance styles
  const performanceStyles = useMemo(() => ({
    ...(performance.enableHardwareAcceleration && {
      transform: 'translateZ(0)',
      backfaceVisibility: 'hidden' as const,
      perspective: '1000',
    }),
    ...(performance.optimizeReflows && {
      willChange: 'transform',
      isolation: 'isolate',
    }),
    ...(performance.enableTouchOptimization && {
      touchAction: 'manipulation',
      WebkitTapHighlightColor: 'transparent',
    }),
  }), [performance]);

  // Callbacks for header actions
  const handleSettingsClick = useCallback(() => {
    // Settings implementation
  }, []);

  const handleHelpClick = useCallback(() => {
    // Help implementation
  }, []);

  const handleProfileClick = useCallback(() => {
    // Profile implementation
  }, []);

  return (
    <LayoutContainer
      className={className}
      highContrast={highContrast}
      sx={performanceStyles}
    >
      <Header
        orientation={isLandscape ? 'landscape' : 'portrait'}
        scanStatus={scanStatus}
        connectionStatus={fleetSync ? 'connected' : 'disconnected'}
        fleetSize={fleetSize}
        batteryLevel={100}
        signalStrength={100}
        onSettingsClick={handleSettingsClick}
        onHelpClick={handleHelpClick}
        onProfileClick={handleProfileClick}
      />

      <MainContent
        maxWidth={false}
        orientation={isLandscape ? 'landscape' : 'portrait'}
        sx={{
          maxWidth: CONTENT_MAX_WIDTH,
          mx: 'auto',
          position: 'relative',
          zIndex: theme.zIndex.appBar - 1,
          boxShadow: theme.shadows[ELEVATION_LEVELS.content],
        }}
      >
        {children}
      </MainContent>

      <StatusBar
        fleetStatus={fleetSync ? 'ACTIVE' : 'DISCONNECTED'}
        scanParameters={{
          scanRate: 30,
          range: 5.0,
          resolution: 0.01
        }}
        networkLatency={25}
      />
    </LayoutContainer>
  );
});

MainLayout.displayName = 'MainLayout';

export default MainLayout;