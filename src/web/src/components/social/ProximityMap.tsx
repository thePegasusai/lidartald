import React, { useEffect, useCallback, useRef, useState } from 'react';
import * as THREE from 'three'; // v0.150.0
import styled from '@emotion/styled'; // v11.10.6

import { UserLocation, UserProfile } from '../../types/user.types';
import { usePointCloud } from '../../hooks/usePointCloud';
import { LIDAR_CONSTANTS, UI_CONSTANTS } from '../../config/constants';

// Constants for radar visualization
const RADAR_SIZE = 400; // pixels
const RADAR_CENTER = RADAR_SIZE / 2;
const MAX_RANGE = LIDAR_CONSTANTS.MAX_RANGE_M;
const RADAR_SCALE = RADAR_SIZE / (MAX_RANGE * 2);
const USER_MARKER_SIZE = 12;

// WebGL settings for optimal performance
const WEBGL_SETTINGS = {
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance'
} as const;

interface ProximityMapProps {
    userLocations: UserLocation[];
    onUserSelect?: (userId: string) => void;
    performanceMode?: 'high' | 'balanced' | 'power-saver';
}

// Styled components for radar visualization
const RadarContainer = styled.div`
    position: relative;
    width: ${RADAR_SIZE}px;
    height: ${RADAR_SIZE}px;
    border-radius: 50%;
    background: rgba(0, 30, 60, 0.1);
    overflow: hidden;
`;

const Canvas = styled.canvas`
    position: absolute;
    top: 0;
    left: 0;
`;

const UserMarker = styled.div<{ x: number; y: number; isActive: boolean }>`
    position: absolute;
    width: ${USER_MARKER_SIZE}px;
    height: ${USER_MARKER_SIZE}px;
    border-radius: 50%;
    background: ${props => props.isActive ? '#4CAF50' : '#2196F3'};
    transform: translate(${props => props.x}px, ${props => props.y}px);
    transition: all ${UI_CONSTANTS.ANIMATION_DURATION_MS}ms ease-out;
    cursor: pointer;
    
    &:hover {
        transform: translate(${props => props.x}px, ${props => props.y}px) scale(1.2);
        box-shadow: 0 0 10px rgba(33, 150, 243, 0.5);
    }
`;

const RadarRing = styled.div<{ scale: number }>`
    position: absolute;
    top: 50%;
    left: 50%;
    width: ${props => RADAR_SIZE * props.scale}px;
    height: ${props => RADAR_SIZE * props.scale}px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 50%;
    transform: translate(-50%, -50%);
`;

export const ProximityMap: React.FC<ProximityMapProps> = ({
    userLocations,
    onUserSelect,
    performanceMode = 'balanced'
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const rendererRef = useRef<THREE.WebGLRenderer>();
    const sceneRef = useRef<THREE.Scene>();
    const cameraRef = useRef<THREE.OrthographicCamera>();
    const frameIdRef = useRef<number>();
    const [selectedUserId, setSelectedUserId] = useState<string | null>(null);

    const { pointCloud, isScanning } = usePointCloud({
        resolution: LIDAR_CONSTANTS.MIN_RESOLUTION_CM,
        range: MAX_RANGE,
        scanRate: performanceMode === 'high' ? 60 : 30
    });

    // Initialize WebGL renderer and scene
    const initializeRenderer = useCallback(() => {
        if (!containerRef.current) return;

        // Create WebGL renderer
        rendererRef.current = new THREE.WebGLRenderer({
            ...WEBGL_SETTINGS,
            canvas: containerRef.current.querySelector('canvas') as HTMLCanvasElement
        });
        rendererRef.current.setSize(RADAR_SIZE, RADAR_SIZE);
        rendererRef.current.setPixelRatio(window.devicePixelRatio);

        // Create scene
        sceneRef.current = new THREE.Scene();
        
        // Create orthographic camera
        cameraRef.current = new THREE.OrthographicCamera(
            -MAX_RANGE, MAX_RANGE,
            MAX_RANGE, -MAX_RANGE,
            0.1, 1000
        );
        cameraRef.current.position.z = 5;
    }, []);

    // Update user positions based on point cloud data
    const updateUserPositions = useCallback(() => {
        if (!sceneRef.current || !pointCloud) return;

        // Clear previous markers
        sceneRef.current.children = [];

        userLocations.forEach(user => {
            // Create user marker geometry
            const geometry = new THREE.CircleGeometry(USER_MARKER_SIZE / RADAR_SCALE, 32);
            const material = new THREE.MeshBasicMaterial({
                color: user.userId === selectedUserId ? 0x4CAF50 : 0x2196F3,
                transparent: true,
                opacity: 0.8
            });
            const marker = new THREE.Mesh(geometry, material);

            // Position marker based on user location
            const x = user.distance * Math.cos(user.coordinates.lat);
            const y = user.distance * Math.sin(user.coordinates.lng);
            marker.position.set(x, y, 0);

            // Add to scene
            sceneRef.current?.add(marker);
        });
    }, [pointCloud, userLocations, selectedUserId]);

    // Animation loop
    const animate = useCallback(() => {
        if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;

        updateUserPositions();
        rendererRef.current.render(sceneRef.current, cameraRef.current);

        // Schedule next frame based on performance mode
        const frameRate = performanceMode === 'high' ? 1000 / 60 : 1000 / 30;
        frameIdRef.current = setTimeout(() => {
            frameIdRef.current = requestAnimationFrame(animate);
        }, frameRate);
    }, [updateUserPositions, performanceMode]);

    // Handle user selection
    const handleUserClick = useCallback((userId: string) => {
        setSelectedUserId(userId);
        onUserSelect?.(userId);
    }, [onUserSelect]);

    // Initialize WebGL context
    useEffect(() => {
        initializeRenderer();
        animate();

        return () => {
            if (frameIdRef.current) {
                cancelAnimationFrame(frameIdRef.current);
            }
            rendererRef.current?.dispose();
        };
    }, [initializeRenderer, animate]);

    return (
        <RadarContainer ref={containerRef}>
            <Canvas />
            {/* Radar rings for distance reference */}
            {[0.25, 0.5, 0.75, 1].map(scale => (
                <RadarRing key={scale} scale={scale} />
            ))}
            {/* User markers */}
            {userLocations.map(user => {
                const x = RADAR_CENTER + (user.distance * Math.cos(user.coordinates.lat) * RADAR_SCALE);
                const y = RADAR_CENTER + (user.distance * Math.sin(user.coordinates.lng) * RADAR_SCALE);
                return (
                    <UserMarker
                        key={user.userId}
                        x={x}
                        y={y}
                        isActive={user.userId === selectedUserId}
                        onClick={() => handleUserClick(user.userId)}
                    />
                );
            })}
        </RadarContainer>
    );
};

export default ProximityMap;