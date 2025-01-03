import React from 'react'; // v18.2.0
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'; // v14.0.0
import userEvent from '@testing-library/user-event'; // v14.4.3
import { Provider } from 'react-redux'; // v8.1.0
import { configureStore } from '@reduxjs/toolkit'; // v1.9.5
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'; // v0.32.0
import { setupServer, rest } from 'msw'; // v1.2.1

import FleetManager from '../../../components/fleet/FleetManager';
import { Fleet, FleetStatus, FleetDevice } from '../../../types/fleet.types';
import { 
    setFleetStatus, 
    createFleetThunk, 
    selectFleetMetrics 
} from '../../../store/slices/fleetSlice';
import { FLEET_CONSTANTS } from '../../../config/constants';

// Mock server setup
const server = setupServer(
    rest.post('/api/v1/fleet/create', (req, res, ctx) => {
        return res(ctx.json({
            id: 'test-fleet-123',
            status: FleetStatus.ACTIVE,
            devices: []
        }));
    }),
    rest.put('/api/v1/fleet/sync/:fleetId', (req, res, ctx) => {
        return res(ctx.json({
            syncProgress: 100,
            networkLatency: 45
        }));
    })
);

// Helper function to setup test environment
const renderWithRedux = (
    ui: React.ReactElement,
    {
        initialState = {},
        store = configureStore({
            reducer: {
                fleet: (state = initialState, action) => state
            }
        })
    } = {}
) => {
    const user = userEvent.setup();
    return {
        user,
        store,
        ...render(
            <Provider store={store}>
                {ui}
            </Provider>
        )
    };
};

// Mock fleet data generator
const createMockFleet = (status: FleetStatus = FleetStatus.ACTIVE): Fleet => ({
    id: 'test-fleet-123',
    name: 'Test Fleet',
    hostDeviceId: 'device-456',
    status,
    devices: [],
    members: [],
    participants: [],
    environmentMapId: 'env-789',
    lastSyncTimestamp: Date.now(),
    maxDevices: FLEET_CONSTANTS.MAX_DEVICES,
    meshNetworkStatus: {
        connected: true,
        latency: 45
    },
    createdAt: new Date(),
    updatedAt: new Date()
});

describe('FleetManager Component', () => {
    beforeEach(() => {
        server.listen();
        vi.useFakeTimers();
    });

    afterEach(() => {
        server.resetHandlers();
        vi.clearAllTimers();
        vi.clearAllMocks();
    });

    it('renders fleet manager with accessibility support', async () => {
        const { user } = renderWithRedux(
            <FleetManager deviceId="device-456" />
        );

        // Verify ARIA landmarks and roles
        expect(screen.getByRole('region')).toHaveAttribute('aria-label', 'Fleet Management Interface');
        
        // Test keyboard navigation
        const syncButton = screen.getByRole('button', { name: /sync fleet/i });
        await user.tab();
        expect(syncButton).toHaveFocus();
        
        // Verify screen reader content
        expect(screen.getByText(/fleet status/i)).toBeInTheDocument();
    });

    it('handles fleet creation with performance validation', async () => {
        const onFleetUpdate = vi.fn();
        const { store } = renderWithRedux(
            <FleetManager 
                deviceId="device-456" 
                onFleetUpdate={onFleetUpdate}
            />
        );

        // Mock fleet creation thunk
        const createFleetSpy = vi.spyOn(store, 'dispatch');
        
        // Measure operation latency
        const startTime = performance.now();
        await waitFor(() => {
            expect(createFleetSpy).toHaveBeenCalledWith(
                expect.any(Function)
            );
        });
        const endTime = performance.now();
        
        // Verify performance meets requirements
        expect(endTime - startTime).toBeLessThan(FLEET_CONSTANTS.SYNC_INTERVAL_MS);
        
        // Verify fleet creation and status update
        expect(onFleetUpdate).toHaveBeenCalledWith(
            expect.objectContaining({
                id: 'test-fleet-123',
                status: FleetStatus.ACTIVE
            })
        );
    });

    it('manages fleet capacity limits', async () => {
        const mockFleet = createMockFleet();
        mockFleet.devices = Array(FLEET_CONSTANTS.MAX_DEVICES).fill(null).map((_, i) => ({
            deviceId: `device-${i}`,
            status: FleetStatus.ACTIVE,
            networkLatency: 45
        }));

        const { user } = renderWithRedux(
            <FleetManager deviceId="new-device" />,
            {
                initialState: {
                    fleet: {
                        currentFleet: mockFleet
                    }
                }
            }
        );

        // Attempt to add device beyond limit
        const joinButton = screen.getByRole('button', { name: /join fleet/i });
        await user.click(joinButton);

        // Verify error handling
        expect(await screen.findByText(/maximum fleet capacity reached/i)).toBeInTheDocument();
    });

    it('handles network failures and recovery', async () => {
        server.use(
            rest.put('/api/v1/fleet/sync/:fleetId', (req, res, ctx) => {
                return res(ctx.status(500));
            })
        );

        const { user } = renderWithRedux(
            <FleetManager deviceId="device-456" />
        );

        // Trigger sync operation
        const syncButton = screen.getByRole('button', { name: /sync fleet/i });
        await user.click(syncButton);

        // Verify error state
        expect(await screen.findByText(/network error/i)).toBeInTheDocument();

        // Test recovery
        server.resetHandlers();
        await user.click(syncButton);
        expect(await screen.findByText(/sync complete/i)).toBeInTheDocument();
    });

    it('validates mesh network performance', async () => {
        const mockFleet = createMockFleet();
        const { rerender } = renderWithRedux(
            <FleetManager deviceId="device-456" />,
            {
                initialState: {
                    fleet: {
                        currentFleet: mockFleet
                    }
                }
            }
        );

        // Monitor network latency
        await waitFor(() => {
            const latencyElement = screen.getByText(/network latency/i);
            const latency = parseInt(latencyElement.textContent!.match(/\d+/)![0]);
            expect(latency).toBeLessThan(FLEET_CONSTANTS.SYNC_INTERVAL_MS);
        });

        // Test mesh network scaling
        for (let i = 0; i < 5; i++) {
            mockFleet.devices.push({
                deviceId: `device-${i}`,
                status: FleetStatus.ACTIVE,
                networkLatency: 45
            });
            rerender(<FleetManager deviceId="device-456" />);
            
            // Verify performance remains within bounds
            await waitFor(() => {
                const latencyElement = screen.getByText(/network latency/i);
                const latency = parseInt(latencyElement.textContent!.match(/\d+/)![0]);
                expect(latency).toBeLessThan(FLEET_CONSTANTS.SYNC_INTERVAL_MS);
            });
        }
    });

    it('maintains fleet state synchronization', async () => {
        const mockFleet = createMockFleet();
        const { user } = renderWithRedux(
            <FleetManager deviceId="device-456" />,
            {
                initialState: {
                    fleet: {
                        currentFleet: mockFleet
                    }
                }
            }
        );

        // Monitor sync progress
        const syncButton = screen.getByRole('button', { name: /sync fleet/i });
        await user.click(syncButton);

        await waitFor(() => {
            const progressElement = screen.getByText(/sync progress/i);
            const progress = parseInt(progressElement.textContent!.match(/\d+/)![0]);
            expect(progress).toBe(100);
        });

        // Verify state consistency
        expect(screen.getByText(/fleet status: active/i)).toBeInTheDocument();
    });
});