import React, { useEffect, useCallback, useState, useRef, useMemo } from 'react'; // v18.2
import ForceGraph2D from 'react-force-graph'; // v1.43
import { styled } from '@mui/material/styles'; // v5.0

import { Fleet, FleetDevice, FleetStatus, NetworkMetrics } from '../../types/fleet.types';
import { useFleetConnection } from '../../hooks/useFleetConnection';
import { MeshNetwork } from '../../utils/meshNetwork';
import { FLEET_CONSTANTS } from '../../config/constants';

// Styled components for enhanced visualization
const GraphContainer = styled('div')(({ theme }) => ({
    width: '100%',
    height: '400px',
    backgroundColor: theme.palette.background.paper,
    borderRadius: theme.shape.borderRadius,
    boxShadow: theme.shadows[4],
    overflow: 'hidden',
    position: 'relative'
}));

const MetricsOverlay = styled('div')(({ theme }) => ({
    position: 'absolute',
    top: theme.spacing(2),
    right: theme.spacing(2),
    padding: theme.spacing(1),
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    color: theme.palette.common.white,
    borderRadius: theme.shape.borderRadius,
    zIndex: 1000,
    fontSize: '0.875rem'
}));

interface Node {
    id: string;
    label: string;
    status: FleetStatus;
    role: 'host' | 'member';
    x?: number;
    y?: number;
}

interface Link {
    source: string;
    target: string;
    quality: number;
    latency: number;
    encrypted: boolean;
}

interface GraphData {
    nodes: Node[];
    links: Link[];
}

interface MeshNetworkProps {
    fleet: Fleet;
    onDeviceSelect?: (deviceId: string) => void;
    onMetricsUpdate?: (metrics: NetworkMetrics) => void;
}

const GRAPH_CONFIG = {
    nodeRelSize: 6,
    linkWidth: 2,
    linkColor: '#cccccc',
    nodeColor: '#1976d2',
    backgroundColor: '#ffffff',
    highlightColor: '#ff4081'
};

const UPDATE_INTERVAL = 1000;
const METRICS_INTERVAL = 500;

export const MeshNetworkVisualization: React.FC<MeshNetworkProps> = ({
    fleet,
    onDeviceSelect,
    onMetricsUpdate
}) => {
    const graphRef = useRef<any>(null);
    const { status, metrics } = useFleetConnection();
    const [selectedNode, setSelectedNode] = useState<string | null>(null);
    const [networkQuality, setNetworkQuality] = useState<number>(100);

    // Generate optimized graph data with security indicators
    const generateGraphData = useCallback((fleet: Fleet, metrics: NetworkMetrics): GraphData => {
        const nodes: Node[] = fleet.devices.map(device => ({
            id: device.deviceId,
            label: device.participantId,
            status: device.status,
            role: device.deviceId === fleet.hostDeviceId ? 'host' : 'member'
        }));

        const links: Link[] = [];
        fleet.devices.forEach(source => {
            fleet.devices.forEach(target => {
                if (source.deviceId < target.deviceId) {
                    const quality = Math.min(100, 100 - (source.networkLatency + target.networkLatency) / 2);
                    links.push({
                        source: source.deviceId,
                        target: target.deviceId,
                        quality,
                        latency: (source.networkLatency + target.networkLatency) / 2,
                        encrypted: true // WebRTC connections are encrypted by default
                    });
                }
            });
        });

        return { nodes, links };
    }, []);

    // Memoized graph data for performance
    const graphData = useMemo(() => 
        generateGraphData(fleet, metrics), 
        [fleet, metrics, generateGraphData]
    );

    // Handle node selection with security context
    const handleNodeClick = useCallback((node: Node) => {
        setSelectedNode(node.id);
        onDeviceSelect?.(node.id);
    }, [onDeviceSelect]);

    // Update network quality metrics
    useEffect(() => {
        const updateNetworkQuality = () => {
            const quality = Math.min(100, metrics.healthScore);
            setNetworkQuality(quality);
            onMetricsUpdate?.(metrics);
        };

        const interval = setInterval(updateNetworkQuality, METRICS_INTERVAL);
        return () => clearInterval(interval);
    }, [metrics, onMetricsUpdate]);

    // Optimize graph rendering
    useEffect(() => {
        if (graphRef.current) {
            graphRef.current.d3Force('charge').strength(-50);
            graphRef.current.d3Force('link').distance(50);
        }
    }, []);

    // Custom node painting with status indicators
    const paintNode = useCallback((node: Node, ctx: CanvasRenderingContext2D) => {
        const size = GRAPH_CONFIG.nodeRelSize;
        ctx.beginPath();
        ctx.arc(node.x || 0, node.y || 0, size, 0, 2 * Math.PI);
        
        // Color based on role and status
        ctx.fillStyle = node.role === 'host' ? '#ff4081' : 
            node.status === FleetStatus.ACTIVE ? '#4caf50' : '#ff9800';
        ctx.fill();

        // Security indicator ring
        ctx.strokeStyle = '#1976d2';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label
        ctx.fillStyle = '#000000';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.label, node.x || 0, (node.y || 0) + size + 5);
    }, []);

    // Custom link painting with quality indicators
    const paintLink = useCallback((link: Link, ctx: CanvasRenderingContext2D) => {
        ctx.beginPath();
        ctx.moveTo(link.source.x || 0, link.source.y || 0);
        ctx.lineTo(link.target.x || 0, link.target.y || 0);
        
        // Color based on connection quality
        const alpha = Math.max(0.2, link.quality / 100);
        ctx.strokeStyle = `rgba(25, 118, 210, ${alpha})`;
        
        // Width based on latency
        ctx.lineWidth = Math.max(1, GRAPH_CONFIG.linkWidth * (1 - link.latency / 1000));
        
        // Encryption indicator
        if (link.encrypted) {
            ctx.setLineDash([5, 5]);
        }
        
        ctx.stroke();
        ctx.setLineDash([]);
    }, []);

    return (
        <GraphContainer>
            <MetricsOverlay>
                <div>Status: {status}</div>
                <div>Network Quality: {networkQuality}%</div>
                <div>Active Connections: {metrics.activeConnections}</div>
                <div>Avg Latency: {metrics.averageLatency.toFixed(2)}ms</div>
            </MetricsOverlay>
            <ForceGraph2D
                ref={graphRef}
                graphData={graphData}
                nodeRelSize={GRAPH_CONFIG.nodeRelSize}
                linkWidth={GRAPH_CONFIG.linkWidth}
                backgroundColor={GRAPH_CONFIG.backgroundColor}
                nodeCanvasObject={paintNode}
                linkCanvasObject={paintLink}
                onNodeClick={handleNodeClick}
                enableNodeDrag={false}
                enableZoomInteraction={true}
                cooldownTicks={50}
                d3VelocityDecay={0.1}
            />
        </GraphContainer>
    );
};

export default MeshNetworkVisualization;