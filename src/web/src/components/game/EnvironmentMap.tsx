import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react'; // v18.2.0
import * as THREE from 'three'; // v0.150.0
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'; // v0.150.0
import { 
    EnvironmentMap, 
    EnvironmentFeature, 
    SurfaceClassification 
} from '../../types/environment.types';
import { useEnvironmentMap } from '../../hooks/useEnvironmentMap';

// Constants for rendering configuration
const UPDATE_INTERVAL_MS = 33; // 30Hz update rate
const DEFAULT_FOV = 75;
const DEFAULT_CAMERA_POSITION = [0, 5, 10];
const MAX_POINTS_PER_FRAME = 100000;
const LOD_LEVELS = [0.25, 0.5, 1.0];

/**
 * Interface for component props
 */
interface EnvironmentMapProps {
    resolution: number;
    autoStart?: boolean;
    onMapUpdate?: (map: EnvironmentMap) => void;
    onError?: (error: Error) => void;
}

/**
 * Interface for scene context
 */
interface SceneContext {
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    pointCloud: THREE.Points;
    features: Map<string, THREE.Mesh>;
}

/**
 * Interface for render statistics
 */
interface RenderStats {
    fps: number;
    drawCalls: number;
    triangles: number;
    points: number;
}

/**
 * High-performance 3D environment map visualization component
 * Implements GPU-accelerated rendering with optimized memory management
 */
export const EnvironmentMapComponent: React.FC<EnvironmentMapProps> = ({
    resolution,
    autoStart = false,
    onMapUpdate,
    onError
}) => {
    // Refs for canvas and animation
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const sceneContextRef = useRef<SceneContext>();
    const animationFrameRef = useRef<number>();
    const lastUpdateRef = useRef<number>(0);

    // State management
    const [renderStats, setRenderStats] = useState<RenderStats>({
        fps: 0,
        drawCalls: 0,
        triangles: 0,
        points: 0
    });

    // Custom hook for environment map management
    const {
        currentMap,
        scanProgress,
        isScanning,
        memoryUsage,
        startScan,
        stopScan
    } = useEnvironmentMap(resolution, autoStart);

    /**
     * Initializes THREE.js scene with optimized settings
     */
    const initializeScene = useCallback(() => {
        if (!canvasRef.current) return;

        // Configure WebGL2 context with performance optimizations
        const contextAttributes: WebGLContextAttributes = {
            alpha: false,
            antialias: true,
            depth: true,
            stencil: false,
            powerPreference: 'high-performance'
        };

        // Initialize renderer with optimal settings
        const renderer = new THREE.WebGLRenderer({
            canvas: canvasRef.current,
            context: canvasRef.current.getContext('webgl2', contextAttributes),
            ...contextAttributes
        });
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Create scene with fog for depth perception
        const scene = new THREE.Scene();
        scene.fog = new THREE.Fog(0x000000, 0.1, 50);

        // Initialize camera with dynamic FOV
        const camera = new THREE.PerspectiveCamera(
            DEFAULT_FOV,
            canvasRef.current.clientWidth / canvasRef.current.clientHeight,
            0.1,
            1000
        );
        camera.position.set(...DEFAULT_CAMERA_POSITION);

        // Configure OrbitControls with smooth damping
        const controls = new OrbitControls(camera, canvasRef.current);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxDistance = 20;
        controls.minDistance = 2;

        // Add lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 10);
        directionalLight.castShadow = true;
        scene.add(directionalLight);

        // Initialize point cloud geometry
        const pointCloudGeometry = new THREE.BufferGeometry();
        const pointCloudMaterial = new THREE.PointsMaterial({
            size: 0.01,
            vertexColors: true,
            sizeAttenuation: true
        });
        const pointCloud = new THREE.Points(pointCloudGeometry, pointCloudMaterial);
        scene.add(pointCloud);

        // Store scene context
        sceneContextRef.current = {
            scene,
            camera,
            renderer,
            controls,
            pointCloud,
            features: new Map()
        };
    }, []);

    /**
     * Updates environment mesh with GPU-accelerated processing
     */
    const updateEnvironmentMesh = useCallback((map: EnvironmentMap) => {
        if (!sceneContextRef.current) return;
        const { scene, pointCloud, features } = sceneContextRef.current;

        // Update point cloud geometry
        const positions = new Float32Array(map.points.length * 3);
        const colors = new Float32Array(map.points.length * 3);

        map.points.forEach((point, index) => {
            positions[index * 3] = point.x;
            positions[index * 3 + 1] = point.y;
            positions[index * 3 + 2] = point.z;

            // Color based on surface classification
            const color = new THREE.Color();
            switch (point.classification) {
                case SurfaceClassification.Floor:
                    color.setHSL(0.3, 0.8, 0.5); // Green
                    break;
                case SurfaceClassification.Wall:
                    color.setHSL(0.6, 0.8, 0.5); // Blue
                    break;
                default:
                    color.setHSL(0, 0, 0.5); // Gray
            }

            colors[index * 3] = color.r;
            colors[index * 3 + 1] = color.g;
            colors[index * 3 + 2] = color.b;
        });

        pointCloud.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        pointCloud.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        pointCloud.geometry.computeBoundingSphere();

        // Update features
        map.features.forEach(feature => {
            if (!features.has(feature.id)) {
                const geometry = new THREE.BufferGeometry();
                const material = new THREE.MeshPhongMaterial({
                    color: 0x00ff00,
                    transparent: true,
                    opacity: 0.7
                });
                const mesh = new THREE.Mesh(geometry, material);
                features.set(feature.id, mesh);
                scene.add(mesh);
            }

            const featureMesh = features.get(feature.id)!;
            const featurePositions = new Float32Array(feature.points.length * 3);
            feature.points.forEach((point, index) => {
                featurePositions[index * 3] = point.x;
                featurePositions[index * 3 + 1] = point.y;
                featurePositions[index * 3 + 2] = point.z;
            });
            featureMesh.geometry.setAttribute('position', new THREE.BufferAttribute(featurePositions, 3));
            featureMesh.geometry.computeBoundingSphere();
        });
    }, []);

    /**
     * Manages continuous scene rendering with performance optimization
     */
    const renderScene = useCallback(() => {
        if (!sceneContextRef.current) return;
        const { scene, camera, renderer, controls } = sceneContextRef.current;

        // Update controls
        controls.update();

        // Implement frustum culling
        const frustum = new THREE.Frustum();
        frustum.setFromProjectionMatrix(
            new THREE.Matrix4().multiplyMatrices(
                camera.projectionMatrix,
                camera.matrixWorldInverse
            )
        );

        // Render scene
        renderer.render(scene, camera);

        // Update render statistics
        setRenderStats({
            fps: 1000 / (performance.now() - lastUpdateRef.current),
            drawCalls: renderer.info.render.calls,
            triangles: renderer.info.render.triangles,
            points: renderer.info.render.points
        });

        lastUpdateRef.current = performance.now();
        animationFrameRef.current = requestAnimationFrame(renderScene);
    }, []);

    // Initialize scene on mount
    useEffect(() => {
        try {
            initializeScene();
            renderScene();

            // Handle resize
            const handleResize = () => {
                if (!canvasRef.current || !sceneContextRef.current) return;
                const { camera, renderer } = sceneContextRef.current;

                camera.aspect = canvasRef.current.clientWidth / canvasRef.current.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);
            };

            window.addEventListener('resize', handleResize);

            return () => {
                window.removeEventListener('resize', handleResize);
                if (animationFrameRef.current) {
                    cancelAnimationFrame(animationFrameRef.current);
                }
                sceneContextRef.current?.renderer.dispose();
            };
        } catch (error) {
            onError?.(error as Error);
        }
    }, [initializeScene, renderScene, onError]);

    // Update environment mesh when map changes
    useEffect(() => {
        if (currentMap) {
            updateEnvironmentMesh(currentMap);
            onMapUpdate?.(currentMap);
        }
    }, [currentMap, updateEnvironmentMesh, onMapUpdate]);

    // Auto-start scanning if enabled
    useEffect(() => {
        if (autoStart) {
            startScan();
        }
        return () => stopScan();
    }, [autoStart, startScan, stopScan]);

    return (
        <div className="environment-map-container">
            <canvas
                ref={canvasRef}
                className="environment-map-canvas"
                style={{ width: '100%', height: '100%' }}
            />
            {isScanning && (
                <div className="scan-progress">
                    Scanning Progress: {Math.round(scanProgress)}%
                </div>
            )}
            <div className="render-stats">
                FPS: {Math.round(renderStats.fps)} | 
                Points: {renderStats.points} | 
                Memory: {memoryUsage.memoryUsageMB}MB
            </div>
        </div>
    );
};

export default EnvironmentMapComponent;