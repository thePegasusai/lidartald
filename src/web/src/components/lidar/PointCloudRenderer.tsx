import React, { useEffect, useRef, useMemo, useCallback } from 'react';
import * as THREE from 'three'; // v0.150.0
import styled from '@emotion/styled'; // v11.10.6
import { Point3D, PointCloud, ScanParameters, PerformanceMetrics } from '../../types/lidar.types';
import { usePointCloud } from '../../hooks/usePointCloud';

// Constants for rendering configuration
const POINT_SIZE_MIN = 0.005;
const POINT_SIZE_MAX = 0.02;
const CAMERA_FOV = 75;
const CAMERA_NEAR = 0.1;
const CAMERA_FAR = 1000;
const FRUSTUM_CULL_MARGIN = 1.2;
const BUFFER_POOL_SIZE = 3;
const PERFORMANCE_SAMPLE_SIZE = 60;
const TARGET_FPS = 60;

// WebGL context attributes for optimal performance
const GL_CONTEXT_ATTRIBUTES: WebGLContextAttributes = {
    alpha: false,
    antialias: true,
    depth: true,
    desynchronized: true,
    failIfMajorPerformanceCaveat: true,
    powerPreference: 'high-performance',
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    stencil: false
};

// Styled components for canvas and container
const CanvasContainer = styled.div`
    position: relative;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: #000;
`;

const StyledCanvas = styled.canvas`
    width: 100%;
    height: 100%;
    display: block;
`;

const PerformanceOverlay = styled.div`
    position: absolute;
    top: 8px;
    right: 8px;
    padding: 4px 8px;
    background: rgba(0, 0, 0, 0.7);
    color: #fff;
    font-family: monospace;
    font-size: 12px;
    border-radius: 4px;
    pointer-events: none;
`;

// Interface for component props
interface PointCloudRendererProps {
    width: number;
    height: number;
    performanceMode?: 'quality' | 'balanced' | 'performance';
    onPerformanceMetrics?: (metrics: PerformanceMetrics) => void;
}

// Geometry buffer pool for memory optimization
class GeometryBufferPool {
    private pool: THREE.BufferGeometry[];
    private inUse: Set<THREE.BufferGeometry>;

    constructor(size: number) {
        this.pool = Array(size).fill(null).map(() => new THREE.BufferGeometry());
        this.inUse = new Set();
    }

    acquire(): THREE.BufferGeometry {
        const geometry = this.pool.find(g => !this.inUse.has(g));
        if (geometry) {
            this.inUse.add(geometry);
            return geometry;
        }
        return new THREE.BufferGeometry();
    }

    release(geometry: THREE.BufferGeometry): void {
        if (this.inUse.has(geometry)) {
            geometry.dispose();
            this.inUse.delete(geometry);
        }
    }

    dispose(): void {
        this.pool.forEach(geometry => geometry.dispose());
        this.pool = [];
        this.inUse.clear();
    }
}

// Performance monitoring system
class PerformanceMonitor {
    private times: number[] = [];
    private frameCount = 0;
    private lastTime = performance.now();

    update(): PerformanceMetrics {
        const currentTime = performance.now();
        const deltaTime = currentTime - this.lastTime;
        this.lastTime = currentTime;

        this.times.push(deltaTime);
        if (this.times.length > PERFORMANCE_SAMPLE_SIZE) {
            this.times.shift();
        }

        this.frameCount++;

        const averageFrameTime = this.times.reduce((a, b) => a + b, 0) / this.times.length;
        const fps = 1000 / averageFrameTime;
        const droppedFrames = Math.max(0, TARGET_FPS - fps);

        return {
            fps,
            frameTime: averageFrameTime,
            droppedFrames,
            frameCount: this.frameCount
        };
    }

    reset(): void {
        this.times = [];
        this.frameCount = 0;
        this.lastTime = performance.now();
    }
}

export const PointCloudRenderer: React.FC<PointCloudRendererProps> = ({
    width,
    height,
    performanceMode = 'balanced',
    onPerformanceMetrics
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const sceneRef = useRef<THREE.Scene>();
    const cameraRef = useRef<THREE.PerspectiveCamera>();
    const rendererRef = useRef<THREE.WebGLRenderer>();
    const bufferPoolRef = useRef<GeometryBufferPool>();
    const performanceMonitorRef = useRef<PerformanceMonitor>();
    const animationFrameRef = useRef<number>();

    const { pointCloud, isScanning, scanError, performanceMetrics } = usePointCloud();

    // Initialize Three.js scene
    const initializeScene = useCallback(() => {
        if (!canvasRef.current) return;

        // Create scene with fog for depth perception
        const scene = new THREE.Scene();
        scene.fog = new THREE.Fog(0x000000, 2, CAMERA_FAR * 0.8);

        // Configure camera
        const camera = new THREE.PerspectiveCamera(
            CAMERA_FOV,
            width / height,
            CAMERA_NEAR,
            CAMERA_FAR
        );
        camera.position.set(0, 0, 5);

        // Initialize WebGL2 renderer
        const renderer = new THREE.WebGLRenderer({
            canvas: canvasRef.current,
            context: canvasRef.current.getContext('webgl2', GL_CONTEXT_ATTRIBUTES),
            ...GL_CONTEXT_ATTRIBUTES
        });

        renderer.setSize(width, height, false);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setClearColor(0x000000);
        renderer.info.autoReset = false;

        // Add lighting
        const ambientLight = new THREE.AmbientLight(0x404040);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        scene.add(ambientLight, directionalLight);

        sceneRef.current = scene;
        cameraRef.current = camera;
        rendererRef.current = renderer;
        bufferPoolRef.current = new GeometryBufferPool(BUFFER_POOL_SIZE);
        performanceMonitorRef.current = new PerformanceMonitor();

    }, [width, height]);

    // Update point cloud visualization
    const updatePointCloud = useCallback((points: Point3D[]) => {
        if (!sceneRef.current || !bufferPoolRef.current) return;

        const geometry = bufferPoolRef.current.acquire();
        const positions = new Float32Array(points.length * 3);
        const colors = new Float32Array(points.length * 3);
        const intensities = new Float32Array(points.length);

        points.forEach((point, index) => {
            positions[index * 3] = point.x;
            positions[index * 3 + 1] = point.y;
            positions[index * 3 + 2] = point.z;
            intensities[index] = point.intensity;

            // Color mapping based on intensity
            const color = new THREE.Color().setHSL(
                0.6 - point.intensity * 0.5,
                1.0,
                0.5 + point.intensity * 0.5
            );
            colors[index * 3] = color.r;
            colors[index * 3 + 1] = color.g;
            colors[index * 3 + 2] = color.b;
        });

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('intensity', new THREE.BufferAttribute(intensities, 1));

        // Point material with size scaling based on performance mode
        const material = new THREE.PointsMaterial({
            size: performanceMode === 'quality' ? POINT_SIZE_MAX : POINT_SIZE_MIN,
            vertexColors: true,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.8
        });

        const pointCloud = new THREE.Points(geometry, material);
        pointCloud.frustumCulled = true;
        pointCloud.renderOrder = 1;

        // Clear previous point cloud
        sceneRef.current.children
            .filter(child => child instanceof THREE.Points)
            .forEach(child => {
                sceneRef.current?.remove(child);
                const geometry = (child as THREE.Points).geometry;
                bufferPoolRef.current?.release(geometry);
            });

        sceneRef.current.add(pointCloud);

    }, [performanceMode]);

    // Animation loop
    const animate = useCallback(() => {
        if (!sceneRef.current || !cameraRef.current || !rendererRef.current) return;

        animationFrameRef.current = requestAnimationFrame(animate);

        // Update camera rotation for dynamic view
        if (isScanning) {
            cameraRef.current.rotation.y += 0.001;
        }

        rendererRef.current.render(sceneRef.current, cameraRef.current);

        // Performance monitoring
        if (performanceMonitorRef.current && onPerformanceMetrics) {
            const metrics = performanceMonitorRef.current.update();
            onPerformanceMetrics(metrics);
        }

        rendererRef.current.info.reset();

    }, [isScanning, onPerformanceMetrics]);

    // Initialize scene
    useEffect(() => {
        initializeScene();
        animate();

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
            bufferPoolRef.current?.dispose();
            rendererRef.current?.dispose();
        };
    }, [initializeScene, animate]);

    // Update point cloud when data changes
    useEffect(() => {
        if (pointCloud?.points) {
            updatePointCloud(pointCloud.points);
        }
    }, [pointCloud, updatePointCloud]);

    // Handle resize
    useEffect(() => {
        if (cameraRef.current && rendererRef.current) {
            cameraRef.current.aspect = width / height;
            cameraRef.current.updateProjectionMatrix();
            rendererRef.current.setSize(width, height, false);
        }
    }, [width, height]);

    return (
        <CanvasContainer>
            <StyledCanvas ref={canvasRef} />
            {performanceMetrics && (
                <PerformanceOverlay>
                    FPS: {Math.round(performanceMetrics.fps)}
                    <br />
                    Points: {pointCloud?.points.length ?? 0}
                    <br />
                    Frame Time: {performanceMetrics.frameTime.toFixed(2)}ms
                </PerformanceOverlay>
            )}
        </CanvasContainer>
    );
};

export default PointCloudRenderer;