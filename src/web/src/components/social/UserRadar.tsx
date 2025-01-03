import React, { useEffect, useCallback, useState, useRef } from 'react';
import styled from '@emotion/styled';
import { useTheme } from '@mui/material';
import { WebGLRenderer } from 'three';
import { interpolate } from 'popmotion';
import { ErrorBoundary } from 'react-error-boundary';
import { usePerformanceMonitor } from 'react-performance-monitor';

import { UserLocation, UserProfile } from '../../types/user.types';
import { useFleetConnection } from '../../hooks/useFleetConnection';
import { useLidarScanner } from '../../hooks/useLidarScanner';

// Constants based on technical specifications
const RADAR_UPDATE_INTERVAL = 33; // ~30Hz to match LiDAR scan rate
const MAX_USERS_DISPLAYED = 32; // Maximum fleet size
const RADAR_ANIMATION_DURATION = 2000;
const MIN_FPS_THRESHOLD = 55; // Target 60 FPS with tolerance

// WebGL shader sources
const VERTEX_SHADER_SOURCE = `
  attribute vec3 position;
  attribute float userDistance;
  uniform mat4 projectionMatrix;
  uniform mat4 viewMatrix;
  varying float vDistance;

  void main() {
    vDistance = userDistance;
    gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0);
    gl_PointSize = max(10.0 * (1.0 - userDistance / 5.0), 3.0);
  }
`;

const FRAGMENT_SHADER_SOURCE = `
  precision highp float;
  varying float vDistance;
  uniform vec3 userColor;

  void main() {
    float alpha = 1.0 - (vDistance / 5.0);
    gl_FragColor = vec4(userColor, alpha);
  }
`;

interface UserRadarProps {
  size: number;
  onUserSelect?: (userId: string, event: React.SyntheticEvent) => void;
  maxRange?: number;
  ariaLabel?: string;
}

// Styled components
const RadarContainer = styled.div<{ size: number }>`
  width: ${props => props.size}px;
  height: ${props => props.size}px;
  position: relative;
  border-radius: 50%;
  overflow: hidden;
`;

const RadarCanvas = styled.canvas`
  width: 100%;
  height: 100%;
`;

const RadarOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
`;

export const UserRadar: React.FC<UserRadarProps> = ({
  size,
  onUserSelect,
  maxRange = 5.0,
  ariaLabel = 'User proximity radar'
}) => {
  const theme = useTheme();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<WebGLRenderer | null>(null);
  const animationFrameRef = useRef<number>();
  const { fleet, isConnected } = useFleetConnection();
  const { scanResult, isScanning } = useLidarScanner();
  const [selectedUser, setSelectedUser] = useState<string | null>(null);
  const performanceMonitor = usePerformanceMonitor();

  // Initialize WebGL renderer
  const initializeWebGL = useCallback(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const gl = canvas.getContext('webgl', {
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });

    if (!gl) {
      throw new Error('WebGL not supported');
    }

    rendererRef.current = new WebGLRenderer({
      canvas,
      context: gl,
      antialias: true,
      alpha: true
    });

    rendererRef.current.setSize(size, size, false);
    rendererRef.current.setPixelRatio(window.devicePixelRatio);

    // Initialize shaders and buffers
    const program = createShaderProgram(gl, VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE);
    gl.useProgram(program);

    // Set up vertex buffers and attributes
    setupBuffers(gl, program);
  }, [size]);

  // Calculate user position with WebGL optimization
  const calculateUserPosition = useCallback((distance: number, angle: number) => {
    const x = distance * Math.cos(angle);
    const y = distance * Math.sin(angle);
    const z = 0;

    return { x, y, z };
  }, []);

  // Handle user selection with accessibility support
  const handleUserSelect = useCallback((userId: string, event: React.SyntheticEvent) => {
    setSelectedUser(userId);
    if (onUserSelect) {
      onUserSelect(userId, event);
    }

    // Announce selection to screen readers
    const user = fleet?.participants.find(p => p.participantId === userId);
    if (user) {
      const announcement = `Selected user ${user.displayName} at ${user.proximityData.distance.toFixed(1)} meters`;
      announceToScreenReader(announcement);
    }
  }, [fleet, onUserSelect]);

  // Update radar visualization
  const updateRadar = useCallback(() => {
    if (!rendererRef.current || !fleet?.participants) return;

    const gl = rendererRef.current.getContext();
    const users = fleet.participants
      .filter(p => p.proximityData.distance <= maxRange)
      .slice(0, MAX_USERS_DISPLAYED);

    // Update vertex data
    const vertexData = new Float32Array(users.length * 3);
    const distanceData = new Float32Array(users.length);

    users.forEach((user, index) => {
      const angle = (index / users.length) * Math.PI * 2;
      const position = calculateUserPosition(user.proximityData.distance, angle);
      
      vertexData[index * 3] = position.x;
      vertexData[index * 3 + 1] = position.y;
      vertexData[index * 3 + 2] = position.z;
      distanceData[index] = user.proximityData.distance;
    });

    // Update buffers
    updateBuffers(gl, vertexData, distanceData);

    // Render frame
    rendererRef.current.render(scene, camera);

    // Monitor performance
    performanceMonitor.measure();
    if (performanceMonitor.getFPS() < MIN_FPS_THRESHOLD) {
      console.warn('Radar performance degradation detected');
    }

    animationFrameRef.current = requestAnimationFrame(updateRadar);
  }, [fleet, maxRange, calculateUserPosition, performanceMonitor]);

  // Setup and cleanup effects
  useEffect(() => {
    initializeWebGL();
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, [initializeWebGL]);

  useEffect(() => {
    if (isConnected && isScanning) {
      updateRadar();
    }
  }, [isConnected, isScanning, updateRadar]);

  return (
    <ErrorBoundary fallback={<div>Error loading radar</div>}>
      <RadarContainer size={size} role="region" aria-label={ariaLabel}>
        <RadarCanvas ref={canvasRef} />
        <RadarOverlay>
          {fleet?.participants.map(user => (
            <UserIndicator
              key={user.participantId}
              user={user}
              selected={selectedUser === user.participantId}
              onSelect={handleUserSelect}
            />
          ))}
        </RadarOverlay>
      </RadarContainer>
    </ErrorBoundary>
  );
};

// Helper components
const UserIndicator = styled.div<{ position: { x: number; y: number }, selected: boolean }>`
  position: absolute;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: ${props => props.theme.palette.primary.main};
  opacity: ${props => props.selected ? 1 : 0.8};
  transform: translate(-50%, -50%);
  left: ${props => props.position.x}%;
  top: ${props => props.position.y}%;
  transition: all 0.3s ease-out;
  cursor: pointer;
`;

// Helper functions
const createShaderProgram = (gl: WebGLRenderingContext, vertexSource: string, fragmentSource: string) => {
  // WebGL shader program creation implementation
  // ...implementation omitted for brevity
  return {} as WebGLProgram;
};

const setupBuffers = (gl: WebGLRenderingContext, program: WebGLProgram) => {
  // WebGL buffer setup implementation
  // ...implementation omitted for brevity
};

const updateBuffers = (gl: WebGLRenderingContext, vertexData: Float32Array, distanceData: Float32Array) => {
  // WebGL buffer update implementation
  // ...implementation omitted for brevity
};

const announceToScreenReader = (message: string) => {
  const announcement = document.createElement('div');
  announcement.setAttribute('aria-live', 'polite');
  announcement.setAttribute('role', 'status');
  announcement.textContent = message;
  document.body.appendChild(announcement);
  setTimeout(() => announcement.remove(), 1000);
};

export default UserRadar;