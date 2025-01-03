import React, { useEffect, useRef, useMemo, useCallback } from 'react';
import { useSelector } from 'react-redux';
import * as THREE from 'three'; // v0.150.0
import { Point3D, Feature, FeatureType, ConfidenceScore } from '../../types/lidar.types';
import { detectFeatures, processPointCloud } from '../../utils/pointCloud';
import { selectScanResult, selectIsScanning, selectProcessingMetrics } from '../../store/slices/lidarSlice';

// Constants for visualization configuration
const CONFIDENCE_COLORS = {
    HIGH: new THREE.Color(0x00ff00),    // Green
    MEDIUM: new THREE.Color(0xffff00),   // Yellow
    LOW: new THREE.Color(0xff0000)       // Red
};

const FEATURE_COLORS = {
    [FeatureType.SURFACE]: new THREE.Color(0x4287f5),
    [FeatureType.OBSTACLE]: new THREE.Color(0xf54242),
    [FeatureType.BOUNDARY]: new THREE.Color(0x42f5a7)
};

interface FeatureDetectionProps {
    className?: string;
    style?: React.CSSProperties;
    confidenceThreshold?: number;
    frameRateLimit?: number;
}

/**
 * Custom hook for managing feature detection state and GPU processing
 */
const useFeatureDetection = (confidenceThreshold: number = 0.6) => {
    const scanResult = useSelector(selectScanResult);
    const isScanning = useSelector(selectIsScanning);
    const metrics = useSelector(selectProcessingMetrics);
    const [features, setFeatures] = React.useState<Feature[]>([]);
    const [error, setError] = React.useState<Error | null>(null);
    const lastProcessTime = useRef<number>(0);

    useEffect(() => {
        if (!scanResult || !isScanning) return;

        const processFeatures = async () => {
            try {
                // Enforce 30Hz frame rate limit
                const now = performance.now();
                if (now - lastProcessTime.current < 33.33) return;
                lastProcessTime.current = now;

                // GPU-accelerated point cloud processing
                const processedCloud = await processPointCloud(
                    scanResult.pointCloud,
                    scanResult.scanParameters
                );

                // Feature detection with confidence filtering
                const detectedFeatures = await detectFeatures(processedCloud);
                const filteredFeatures = detectedFeatures.filter(
                    feature => feature.confidence >= confidenceThreshold
                );

                setFeatures(filteredFeatures);
                setError(null);
            } catch (err) {
                setError(err as Error);
            }
        };

        processFeatures();
    }, [scanResult, isScanning, confidenceThreshold]);

    return { features, isProcessing: isScanning, error, metrics };
};

/**
 * Renders detected features with visual confidence indicators
 */
const renderFeatures = (
    features: Feature[],
    scene: THREE.Scene,
    renderer: THREE.WebGLRenderer
) => {
    features.forEach(feature => {
        // Create geometry for feature points
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(
            feature.coordinates.flatMap(point => [point.x, point.y, point.z])
        );
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        // Apply feature-specific material with confidence color
        const material = new THREE.PointsMaterial({
            size: 0.02,
            color: FEATURE_COLORS[feature.type],
            opacity: feature.confidence,
            transparent: true,
            blending: THREE.AdditiveBlending
        });

        // Create point cloud mesh
        const points = new THREE.Points(geometry, material);
        scene.add(points);

        // Add confidence indicator
        const confidenceSprite = new THREE.Sprite(
            new THREE.SpriteMaterial({
                map: createConfidenceTexture(feature.confidence),
                color: getConfidenceColor(feature.confidence)
            })
        );
        
        // Position confidence indicator above feature center
        const center = calculateFeatureCenter(feature.coordinates);
        confidenceSprite.position.set(center.x, center.y + 0.1, center.z);
        confidenceSprite.scale.set(0.1, 0.1, 1);
        scene.add(confidenceSprite);
    });

    renderer.render(scene, camera);
};

/**
 * Feature detection visualization component with GPU acceleration
 */
const FeatureDetection: React.FC<FeatureDetectionProps> = ({
    className,
    style,
    confidenceThreshold = 0.6,
    frameRateLimit = 30
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);

    const { features, isProcessing, error, metrics } = useFeatureDetection(confidenceThreshold);

    // Initialize THREE.js scene
    useEffect(() => {
        if (!containerRef.current) return;

        // Setup renderer
        rendererRef.current = new THREE.WebGLRenderer({ antialias: true });
        rendererRef.current.setPixelRatio(window.devicePixelRatio);
        rendererRef.current.setSize(
            containerRef.current.clientWidth,
            containerRef.current.clientHeight
        );
        containerRef.current.appendChild(rendererRef.current.domElement);

        // Setup scene
        sceneRef.current = new THREE.Scene();
        sceneRef.current.background = new THREE.Color(0x000000);

        // Setup camera
        cameraRef.current = new THREE.PerspectiveCamera(
            75,
            containerRef.current.clientWidth / containerRef.current.clientHeight,
            0.1,
            1000
        );
        cameraRef.current.position.z = 5;

        return () => {
            if (rendererRef.current) {
                rendererRef.current.dispose();
            }
        };
    }, []);

    // Handle window resize
    useEffect(() => {
        const handleResize = () => {
            if (!containerRef.current || !rendererRef.current || !cameraRef.current) return;

            cameraRef.current.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
            cameraRef.current.updateProjectionMatrix();
            rendererRef.current.setSize(
                containerRef.current.clientWidth,
                containerRef.current.clientHeight
            );
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    // Render features when updated
    useEffect(() => {
        if (!sceneRef.current || !rendererRef.current || !features.length) return;

        sceneRef.current.clear();
        renderFeatures(features, sceneRef.current, rendererRef.current);
    }, [features]);

    return (
        <div
            ref={containerRef}
            className={className}
            style={{
                width: '100%',
                height: '100%',
                position: 'relative',
                ...style
            }}
        >
            {isProcessing && (
                <div className="processing-overlay">
                    Processing at {metrics.averageTime.toFixed(1)}ms/frame
                </div>
            )}
            {error && (
                <div className="error-overlay">
                    Error: {error.message}
                </div>
            )}
        </div>
    );
};

// Helper functions
const createConfidenceTexture = (confidence: number): THREE.Texture => {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to get canvas context');

    ctx.beginPath();
    ctx.arc(32, 32, 30, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255, 255, 255, ${confidence})`;
    ctx.fill();

    return new THREE.CanvasTexture(canvas);
};

const getConfidenceColor = (confidence: number): THREE.Color => {
    if (confidence >= 0.8) return CONFIDENCE_COLORS.HIGH;
    if (confidence >= 0.6) return CONFIDENCE_COLORS.MEDIUM;
    return CONFIDENCE_COLORS.LOW;
};

const calculateFeatureCenter = (points: Point3D[]): Point3D => {
    const sum = points.reduce(
        (acc, point) => ({
            x: acc.x + point.x,
            y: acc.y + point.y,
            z: acc.z + point.z,
            intensity: 0
        }),
        { x: 0, y: 0, z: 0, intensity: 0 }
    );

    return {
        x: sum.x / points.length,
        y: sum.y / points.length,
        z: sum.z / points.length,
        intensity: 0
    };
};

export default FeatureDetection;