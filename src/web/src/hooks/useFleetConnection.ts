import { useEffect, useCallback, useState, useRef } from 'react'; // v18.2
import { useDispatch, useSelector } from 'react-redux'; // v8.1
import * as Y from 'yjs'; // v13.6
import adapter from 'webrtc-adapter'; // v8.2

import { Fleet, FleetStatus, FleetMetrics } from '../types/fleet.types';
import { createMeshNetwork, MeshNetwork } from '../utils/meshNetwork';
import { 
    setFleetStatus, 
    updateSyncProgress, 
    updateMeshNetwork, 
    selectCurrentFleet,
    selectFleetStatus,
    selectMeshNetwork 
} from '../store/slices/fleetSlice';
import { 
    createFleet as apiCreateFleet,
    joinFleet as apiJoinFleet,
    leaveFleet as apiLeaveFleet,
    FleetEncryption 
} from '../api/fleetApi';
import { FLEET_CONSTANTS } from '../config/constants';

interface FleetConnectionOptions {
    autoReconnect?: boolean;
    syncInterval?: number;
    encryption?: boolean;
    compressionLevel?: number;
    maxRetries?: number;
    performanceMode?: 'high' | 'balanced' | 'powersave';
}

const DEFAULT_OPTIONS: Required<FleetConnectionOptions> = {
    autoReconnect: true,
    syncInterval: FLEET_CONSTANTS.SYNC_INTERVAL_MS,
    encryption: true,
    compressionLevel: 6,
    maxRetries: FLEET_CONSTANTS.MAX_RETRY_ATTEMPTS,
    performanceMode: 'balanced'
};

export function useFleetConnection(options: FleetConnectionOptions = {}) {
    const dispatch = useDispatch();
    const fleet = useSelector(selectCurrentFleet);
    const status = useSelector(selectFleetStatus);
    const meshNetworkState = useSelector(selectMeshNetwork);

    const [isConnected, setIsConnected] = useState(false);
    const [metrics, setMetrics] = useState<FleetMetrics>({
        averageLatency: 0,
        activeConnections: 0,
        topology: 'mesh',
        healthScore: 100
    });

    const meshNetworkRef = useRef<MeshNetwork | null>(null);
    const yDocRef = useRef<Y.Doc | null>(null);
    const encryptionRef = useRef<FleetEncryption | null>(null);
    const optionsRef = useRef<Required<FleetConnectionOptions>>({
        ...DEFAULT_OPTIONS,
        ...options
    });

    // Initialize CRDT document and encryption
    useEffect(() => {
        yDocRef.current = new Y.Doc();
        if (optionsRef.current.encryption) {
            encryptionRef.current = new FleetEncryption();
        }

        return () => {
            yDocRef.current?.destroy();
            meshNetworkRef.current?.removePeer(fleet?.hostDeviceId || '');
        };
    }, []);

    // Setup performance monitoring
    useEffect(() => {
        const monitorInterval = setInterval(() => {
            if (meshNetworkRef.current) {
                const currentMetrics = meshNetworkRef.current.getPerformanceMetrics();
                setMetrics(currentMetrics);
                dispatch(updateMeshNetwork({
                    latency: currentMetrics.averageLatency,
                    activeConnections: currentMetrics.activeConnections
                }));
            }
        }, optionsRef.current.syncInterval);

        return () => clearInterval(monitorInterval);
    }, [dispatch]);

    // Handle fleet status changes
    useEffect(() => {
        if (status === FleetStatus.ACTIVE && !isConnected) {
            initializeMeshNetwork();
        } else if (status === FleetStatus.DISCONNECTED && isConnected) {
            cleanupMeshNetwork();
        }
    }, [status, isConnected]);

    const initializeMeshNetwork = useCallback(async () => {
        if (!fleet) return;

        try {
            const networkConfig = {
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
                maxRetries: optionsRef.current.maxRetries,
                compressionLevel: optionsRef.current.compressionLevel,
                encryptionKey: encryptionRef.current?.getKey() || new Uint8Array()
            };

            meshNetworkRef.current = createMeshNetwork(fleet, networkConfig);
            
            meshNetworkRef.current.on('peerConnected', (deviceId: string) => {
                dispatch(updateSyncProgress(100));
                setIsConnected(true);
            });

            meshNetworkRef.current.on('peerDisconnected', (deviceId: string) => {
                if (optionsRef.current.autoReconnect) {
                    meshNetworkRef.current?.addPeer(deviceId, {
                        deviceId,
                        capabilities: {
                            processingPower: 100,
                            networkLatency: 0
                        }
                    });
                }
            });

            meshNetworkRef.current.on('dataChannelMessage', async (deviceId: string, data: ArrayBuffer) => {
                if (yDocRef.current) {
                    Y.applyUpdate(yDocRef.current, new Uint8Array(data));
                }
            });

            await meshNetworkRef.current.addPeer(fleet.hostDeviceId, {
                deviceId: fleet.hostDeviceId,
                capabilities: {
                    processingPower: 100,
                    networkLatency: 0
                }
            });

        } catch (error) {
            console.error('[Fleet Connection Error]', error);
            dispatch(setFleetStatus(FleetStatus.DISCONNECTED));
            setIsConnected(false);
        }
    }, [fleet, dispatch]);

    const cleanupMeshNetwork = useCallback(() => {
        if (meshNetworkRef.current) {
            meshNetworkRef.current.removeAllListeners();
            meshNetworkRef.current = null;
        }
        setIsConnected(false);
    }, []);

    const createFleet = useCallback(async (name: string) => {
        try {
            const result = await apiCreateFleet({ name });
            dispatch(setFleetStatus(FleetStatus.ACTIVE));
            return result;
        } catch (error) {
            console.error('[Create Fleet Error]', error);
            throw error;
        }
    }, [dispatch]);

    const joinFleet = useCallback(async (fleetId: string) => {
        try {
            const result = await apiJoinFleet(fleetId);
            dispatch(setFleetStatus(FleetStatus.ACTIVE));
            return result;
        } catch (error) {
            console.error('[Join Fleet Error]', error);
            throw error;
        }
    }, [dispatch]);

    const leaveFleet = useCallback(async () => {
        try {
            await apiLeaveFleet();
            cleanupMeshNetwork();
            dispatch(setFleetStatus(FleetStatus.DISCONNECTED));
        } catch (error) {
            console.error('[Leave Fleet Error]', error);
            throw error;
        }
    }, [dispatch, cleanupMeshNetwork]);

    const syncState = useCallback(async () => {
        if (!meshNetworkRef.current || !yDocRef.current) return;

        try {
            dispatch(setFleetStatus(FleetStatus.SYNCING));
            const update = Y.encodeStateAsUpdate(yDocRef.current);
            await meshNetworkRef.current.synchronizeState(
                fleet?.hostDeviceId || '',
                update.buffer
            );
            dispatch(setFleetStatus(FleetStatus.ACTIVE));
        } catch (error) {
            console.error('[Sync State Error]', error);
            dispatch(setFleetStatus(FleetStatus.ACTIVE));
        }
    }, [fleet, dispatch]);

    const getNetworkStats = useCallback(() => {
        return meshNetworkRef.current?.getPerformanceMetrics() || metrics;
    }, [metrics]);

    const setEncryption = useCallback((enabled: boolean) => {
        optionsRef.current.encryption = enabled;
        if (enabled && !encryptionRef.current) {
            encryptionRef.current = new FleetEncryption();
        } else if (!enabled) {
            encryptionRef.current = null;
        }
    }, []);

    const setCompressionLevel = useCallback((level: number) => {
        optionsRef.current.compressionLevel = Math.max(0, Math.min(9, level));
        if (meshNetworkRef.current) {
            meshNetworkRef.current.setCompressionLevel(optionsRef.current.compressionLevel);
        }
    }, []);

    return {
        fleet,
        status,
        isConnected,
        metrics,
        createFleet,
        joinFleet,
        leaveFleet,
        syncState,
        getNetworkStats,
        setEncryption,
        setCompressionLevel
    };
}