import React, { useEffect, useState, useCallback, useMemo } from 'react'; // v18.2.0
import { useDispatch, useSelector } from 'react-redux'; // v8.1.0
import { 
    Box, 
    Typography, 
    Button, 
    Grid, 
    Paper, 
    CircularProgress 
} from '@mui/material'; // v5.13.0
import { 
    Fleet, 
    FleetStatus, 
    FleetRole, 
    FleetDevice, 
    FleetMember, 
    FleetNetworkMetrics 
} from '../../types/fleet.types';

// Constants for fleet management
const FLEET_UPDATE_INTERVAL = 1000; // 1 second refresh
const MAX_FLEET_SIZE = 32;
const SYNC_TIMEOUT = 5000;
const MAX_RETRY_ATTEMPTS = 3;
const COMPRESSION_THRESHOLD = 1024;
const PERFORMANCE_SAMPLE_RATE = 100;

interface FleetManagerProps {
    deviceId: string;
    onFleetUpdate?: (fleet: Fleet) => void;
    onError?: (error: Error) => void;
}

interface FleetMetrics {
    networkLatency: number;
    syncProgress: number;
    deviceCount: number;
    healthScore: number;
}

// Styled components for accessibility and theme consistency
const FleetManagerContainer = styled(Box)(({ theme }) => ({
    padding: theme.spacing(3),
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    role: 'region',
    'aria-label': 'Fleet Management Interface',
    '& .MuiPaper-root': {
        marginBottom: theme.spacing(2)
    }
}));

const FleetManager: React.FC<FleetManagerProps> = ({
    deviceId,
    onFleetUpdate,
    onError
}) => {
    const dispatch = useDispatch();
    const [fleet, setFleet] = useState<Fleet | null>(null);
    const [metrics, setMetrics] = useState<FleetMetrics>({
        networkLatency: 0,
        syncProgress: 0,
        deviceCount: 0,
        healthScore: 100
    });
    const [isInitializing, setIsInitializing] = useState(true);

    // Memoized fleet status for performance
    const fleetStatus = useMemo(() => fleet?.status || FleetStatus.INITIALIZING, [fleet]);

    // Initialize WebRTC connection pool and mesh network
    const initializeFleet = useCallback(async (role: FleetRole = FleetRole.MEMBER) => {
        try {
            setIsInitializing(true);
            const config = {
                maxDevices: MAX_FLEET_SIZE,
                updateInterval: FLEET_UPDATE_INTERVAL,
                retryAttempts: MAX_RETRY_ATTEMPTS,
                compressionThreshold: COMPRESSION_THRESHOLD
            };

            // Initialize secure WebRTC connections
            const fleetInstance = await initializeSecureFleet(deviceId, role, config);
            setFleet(fleetInstance);
            
            // Start performance monitoring
            initializeMetricsCollection();
            
            onFleetUpdate?.(fleetInstance);
        } catch (error) {
            onError?.(error as Error);
            setIsInitializing(false);
        }
    }, [deviceId, onFleetUpdate, onError]);

    // Handle fleet synchronization with optimized data transfer
    const handleFleetSync = useCallback(async () => {
        if (!fleet) return;

        try {
            const syncOptions = {
                compression: true,
                timeout: SYNC_TIMEOUT,
                validateChecksum: true
            };

            setMetrics(prev => ({ ...prev, syncProgress: 0 }));
            
            // Perform incremental sync with compression
            const syncResult = await synchronizeFleetState(fleet, syncOptions);
            
            setMetrics(prev => ({
                ...prev,
                syncProgress: 100,
                networkLatency: syncResult.latency
            }));

            // Update fleet state
            setFleet(syncResult.updatedFleet);
            onFleetUpdate?.(syncResult.updatedFleet);
        } catch (error) {
            onError?.(error as Error);
        }
    }, [fleet, onFleetUpdate, onError]);

    // Monitor fleet health and performance
    const monitorFleetHealth = useCallback(() => {
        if (!fleet) return;

        const healthMetrics = calculateFleetHealth(fleet);
        setMetrics(prev => ({
            ...prev,
            healthScore: healthMetrics.score,
            deviceCount: fleet.devices.length
        }));
    }, [fleet]);

    // Effect for fleet initialization
    useEffect(() => {
        if (!deviceId) return;
        initializeFleet();
    }, [deviceId, initializeFleet]);

    // Effect for periodic fleet synchronization
    useEffect(() => {
        if (!fleet || fleetStatus === FleetStatus.INITIALIZING) return;

        const syncInterval = setInterval(handleFleetSync, FLEET_UPDATE_INTERVAL);
        return () => clearInterval(syncInterval);
    }, [fleet, fleetStatus, handleFleetSync]);

    // Effect for health monitoring
    useEffect(() => {
        if (!fleet) return;

        const healthInterval = setInterval(monitorFleetHealth, PERFORMANCE_SAMPLE_RATE);
        return () => clearInterval(healthInterval);
    }, [fleet, monitorFleetHealth]);

    // Render fleet status and controls
    return (
        <FleetManagerContainer>
            <Paper elevation={2}>
                <Box p={2}>
                    <Typography variant="h6" component="h2">
                        Fleet Status: {fleetStatus}
                    </Typography>
                    {isInitializing ? (
                        <CircularProgress 
                            aria-label="Initializing fleet"
                            size={24}
                        />
                    ) : (
                        <Grid container spacing={2}>
                            <Grid item xs={12} md={6}>
                                <Typography>
                                    Devices: {metrics.deviceCount}/{MAX_FLEET_SIZE}
                                </Typography>
                                <Typography>
                                    Network Latency: {metrics.networkLatency}ms
                                </Typography>
                            </Grid>
                            <Grid item xs={12} md={6}>
                                <Typography>
                                    Sync Progress: {metrics.syncProgress}%
                                </Typography>
                                <Typography>
                                    Health Score: {metrics.healthScore}%
                                </Typography>
                            </Grid>
                        </Grid>
                    )}
                </Box>
            </Paper>

            {fleet?.devices.map((device: FleetDevice) => (
                <Paper 
                    key={device.deviceId}
                    elevation={1}
                    role="listitem"
                    aria-label={`Device ${device.deviceId}`}
                >
                    <Box p={2}>
                        <Typography variant="subtitle1">
                            Device ID: {device.deviceId}
                        </Typography>
                        <Typography>
                            Status: {device.status}
                        </Typography>
                        <Typography>
                            Latency: {device.networkLatency}ms
                        </Typography>
                    </Box>
                </Paper>
            ))}

            <Box mt={2}>
                <Button
                    variant="contained"
                    color="primary"
                    onClick={handleFleetSync}
                    disabled={isInitializing || !fleet}
                    aria-label="Synchronize Fleet"
                >
                    Sync Fleet
                </Button>
            </Box>
        </FleetManagerContainer>
    );
};

export default FleetManager;