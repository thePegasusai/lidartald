import styled from '@emotion/styled'; // ^11.11.0
import { css } from '@emotion/react'; // ^11.11.0
import { Paper } from '@mui/material'; // ^5.13.0
import { lightTheme, darkTheme } from './theme';
import { fadeIn, scanProgress } from './animations';

// Elevation constants for Material Design 3.0 compliance
const ELEVATION_LEVELS = {
  low: 1,
  medium: 2,
  high: 3,
  floating: 4,
  lidarView: 5,
  fleetCard: 3,
  outdoorMode: 6
} as const;

// Responsive breakpoints with orientation support
const GRID_BREAKPOINTS = {
  mobile: '320px',
  tablet: '768px',
  desktop: '1024px',
  wide: '1440px',
  portrait: '(orientation: portrait)',
  landscape: '(orientation: landscape)',
  highContrast: '(prefers-contrast: high)'
} as const;

// Component size definitions
const COMPONENT_SIZES = {
  small: '24px',
  medium: '40px',
  large: '56px',
  lidarViewport: 'calc(100vh - 120px)',
  fleetCard: '160px',
  touchTarget: '48px'
} as const;

// Create elevation shadow with outdoor visibility optimization
const createElevation = (level: number, isOutdoorMode: boolean = false) => {
  const baseElevation = level * 2;
  const ambientOpacity = isOutdoorMode ? 0.4 : 0.2;
  const directionalOpacity = isOutdoorMode ? 0.6 : 0.3;
  
  return `
    ${isOutdoorMode ? 'box-shadow: 0 0 ${baseElevation * 2}px rgba(0, 0, 0, 0.8);' : ''}
    box-shadow: 
      0px ${baseElevation}px ${baseElevation * 2}px rgba(0, 0, 0, ${ambientOpacity}),
      0px ${baseElevation / 2}px ${baseElevation}px rgba(0, 0, 0, ${directionalOpacity});
  `;
};

// LiDAR viewport with 3D perspective and GPU acceleration
export const LidarViewport = styled.div`
  position: relative;
  width: 100%;
  height: ${COMPONENT_SIZES.lidarViewport};
  perspective: 1000px;
  transform-style: preserve-3d;
  background-color: ${({ theme }) => theme.palette.lidar.background};
  ${createElevation(ELEVATION_LEVELS.lidarView)}
  contain: layout paint size;
  will-change: transform;
  
  @media ${GRID_BREAKPOINTS.portrait} {
    height: calc(100vh - 160px);
  }

  @media ${GRID_BREAKPOINTS.landscape} {
    height: 100vh;
  }

  @media ${GRID_BREAKPOINTS.highContrast} {
    ${createElevation(ELEVATION_LEVELS.lidarView, true)}
  }
`;

// Fleet member card with outdoor visibility optimization
export const FleetCard = styled(Paper)`
  width: ${COMPONENT_SIZES.fleetCard};
  padding: ${({ theme }) => theme.spacing(2)};
  background-color: ${({ theme }) => theme.palette.background.paper};
  backdrop-filter: blur(8px);
  transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  ${createElevation(ELEVATION_LEVELS.fleetCard)}
  contain: content;
  
  &:hover {
    transform: translateY(-2px);
  }

  @media ${GRID_BREAKPOINTS.highContrast} {
    ${createElevation(ELEVATION_LEVELS.fleetCard, true)}
  }
`;

// Scan progress indicator with GPU-accelerated animation
export const ScanProgressBar = styled.div`
  height: 4px;
  background: linear-gradient(
    90deg,
    ${({ theme }) => theme.palette.lidar.point},
    ${({ theme }) => theme.palette.lidar.highlight}
  );
  animation: ${scanProgress} 2s cubic-bezier(0.4, 0, 0.2, 1) infinite;
  transform: translateZ(0);
  will-change: transform;
`;

// Point cloud container with GPU optimization
export const PointCloudContainer = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  transform-style: preserve-3d;
  animation: ${fadeIn} 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  contain: layout paint size;
  will-change: transform;
`;

// Feature highlight overlay
export const FeatureHighlight = styled.div`
  position: absolute;
  border: 2px solid ${({ theme }) => theme.palette.lidar.feature};
  border-radius: ${({ theme }) => theme.shape.borderRadius}px;
  pointer-events: none;
  transform: translateZ(0);
  will-change: transform;
  
  @media ${GRID_BREAKPOINTS.highContrast} {
    border-width: 3px;
  }
`;

// Touch-optimized control button
export const ControlButton = styled.button`
  width: ${COMPONENT_SIZES.touchTarget};
  height: ${COMPONENT_SIZES.touchTarget};
  border-radius: 50%;
  background-color: ${({ theme }) => theme.palette.primary.main};
  color: ${({ theme }) => theme.palette.primary.contrastText};
  border: none;
  ${createElevation(ELEVATION_LEVELS.floating)}
  transition: background-color 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  contain: layout paint;
  
  &:active {
    background-color: ${({ theme }) => theme.palette.primary.dark};
  }

  @media ${GRID_BREAKPOINTS.highContrast} {
    ${createElevation(ELEVATION_LEVELS.floating, true)}
  }
`;

// Grid overlay for environment mapping
export const GridOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-image: linear-gradient(
    0deg,
    ${({ theme }) => theme.palette.lidar.grid} 1px,
    transparent 1px
  ),
  linear-gradient(
    90deg,
    ${({ theme }) => theme.palette.lidar.grid} 1px,
    transparent 1px
  );
  background-size: 50px 50px;
  opacity: 0.2;
  pointer-events: none;
  transform: translateZ(0);
`;

// Surface mapping visualization
export const SurfaceMap = styled.div`
  position: absolute;
  background-color: ${({ theme }) => theme.palette.lidar.surface};
  opacity: 0.3;
  transition: opacity 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  pointer-events: none;
  transform: translateZ(0);
  will-change: opacity;
  
  &:hover {
    opacity: 0.5;
  }
`;

// Fleet member indicator with distance-based scaling
export const FleetMemberIndicator = styled.div<{ distance: number }>`
  width: ${COMPONENT_SIZES.medium};
  height: ${COMPONENT_SIZES.medium};
  border-radius: 50%;
  background-color: ${({ theme }) => theme.palette.secondary.main};
  transform: scale(${({ distance }) => Math.max(0.5, 1 - distance / 10)}) translateZ(0);
  transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  contain: layout paint;
  will-change: transform;
`;