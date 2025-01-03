import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { jest } from '@jest/globals';
import { Provider } from 'react-redux';
import userEvent from '@testing-library/user-event';
import 'jest-webgl-canvas-mock';

import UserRadar from '../../../components/social/UserRadar';
import { useFleetConnection } from '../../../hooks/useFleetConnection';
import { FLEET_CONSTANTS, UI_CONSTANTS } from '../../../config/constants';

// Mock dependencies
jest.mock('../../../hooks/useFleetConnection');
jest.mock('three', () => ({
  WebGLRenderer: jest.fn().mockImplementation(() => ({
    setSize: jest.fn(),
    setPixelRatio: jest.fn(),
    render: jest.fn(),
    dispose: jest.fn(),
    getContext: jest.fn().mockReturnValue({
      canvas: document.createElement('canvas')
    })
  }))
}));

// Test constants based on technical specifications
const TEST_CONSTANTS = {
  SCAN_RATE: 30, // 30Hz scan rate
  MAX_RANGE: 5, // 5-meter range
  UPDATE_INTERVAL: 33.33, // ~30Hz update interval
  FRAME_TIME: 16.67 // ~60 FPS target
};

// Mock fleet data
const mockFleetData = {
  id: 'test-fleet',
  name: 'Test Fleet',
  participants: [],
  maxUsers: FLEET_CONSTANTS.MAX_DEVICES
};

// Mock user data
const mockUserData = [
  {
    participantId: 'user1',
    displayName: 'Test User 1',
    proximityData: {
      distance: 2.5,
      lastUpdate: Date.now()
    }
  },
  {
    participantId: 'user2',
    displayName: 'Test User 2',
    proximityData: {
      distance: 1.5,
      lastUpdate: Date.now()
    }
  }
];

// Helper function to render with providers
const renderWithProviders = (ui: React.ReactNode, options = {}) => {
  const mockStore = {
    getState: () => ({}),
    subscribe: jest.fn(),
    dispatch: jest.fn()
  };

  return {
    ...render(
      <Provider store={mockStore}>
        {ui}
      </Provider>
    ),
    store: mockStore
  };
};

// Helper function to mock performance.now for animation testing
const mockPerformanceNow = () => {
  let count = 0;
  return jest.spyOn(performance, 'now').mockImplementation(() => {
    count += TEST_CONSTANTS.FRAME_TIME;
    return count;
  });
};

describe('UserRadar Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (useFleetConnection as jest.Mock).mockReturnValue({
      fleet: mockFleetData,
      isConnected: true
    });
  });

  it('renders radar visualization correctly', async () => {
    const { container } = renderWithProviders(
      <UserRadar 
        size={400}
        maxRange={TEST_CONSTANTS.MAX_RANGE}
        ariaLabel="Test radar"
      />
    );

    // Verify radar container dimensions
    const radarContainer = container.firstChild;
    expect(radarContainer).toHaveStyle({
      width: '400px',
      height: '400px'
    });

    // Verify WebGL canvas initialization
    const canvas = screen.getByRole('presentation');
    expect(canvas).toBeInTheDocument();
    expect(canvas.tagName.toLowerCase()).toBe('canvas');

    // Verify ARIA labels
    expect(radarContainer).toHaveAttribute('role', 'region');
    expect(radarContainer).toHaveAttribute('aria-label', 'Test radar');
  });

  it('displays nearby users with correct positions', async () => {
    (useFleetConnection as jest.Mock).mockReturnValue({
      fleet: { ...mockFleetData, participants: mockUserData },
      isConnected: true
    });

    renderWithProviders(
      <UserRadar 
        size={400}
        maxRange={TEST_CONSTANTS.MAX_RANGE}
      />
    );

    // Verify user indicators are rendered
    const userIndicators = screen.getAllByRole('button');
    expect(userIndicators).toHaveLength(mockUserData.length);

    // Verify user information tooltips
    await userEvent.hover(userIndicators[0]);
    await waitFor(() => {
      expect(screen.getByText('Test User 1')).toBeInTheDocument();
      expect(screen.getByText('2.5m')).toBeInTheDocument();
    });
  });

  it('handles real-time updates at 30Hz', async () => {
    const mockNow = mockPerformanceNow();
    const onUserSelect = jest.fn();

    renderWithProviders(
      <UserRadar 
        size={400}
        maxRange={TEST_CONSTANTS.MAX_RANGE}
        onUserSelect={onUserSelect}
      />
    );

    // Verify update rate
    await waitFor(() => {
      expect(mockNow).toHaveBeenCalledTimes(Math.floor(1000 / TEST_CONSTANTS.FRAME_TIME));
    }, { timeout: 1000 });

    mockNow.mockRestore();
  });

  it('supports keyboard and screen reader navigation', async () => {
    (useFleetConnection as jest.Mock).mockReturnValue({
      fleet: { ...mockFleetData, participants: mockUserData },
      isConnected: true
    });

    renderWithProviders(
      <UserRadar 
        size={400}
        maxRange={TEST_CONSTANTS.MAX_RANGE}
      />
    );

    const userIndicators = screen.getAllByRole('button');

    // Test keyboard navigation
    await userEvent.tab();
    expect(userIndicators[0]).toHaveFocus();

    await userEvent.keyboard('{Enter}');
    expect(screen.getByText('Selected user Test User 1 at 2.5 meters')).toBeInTheDocument();

    // Test arrow key navigation
    await userEvent.keyboard('{ArrowRight}');
    expect(userIndicators[1]).toHaveFocus();
  });

  it('handles performance degradation gracefully', async () => {
    const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
    const mockNow = mockPerformanceNow();

    // Simulate slow frame times
    mockNow.mockImplementation(() => Date.now() + 20); // >16.67ms frame time

    renderWithProviders(
      <UserRadar 
        size={400}
        maxRange={TEST_CONSTANTS.MAX_RANGE}
      />
    );

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Radar performance degradation detected')
      );
    });

    consoleSpy.mockRestore();
    mockNow.mockRestore();
  });

  it('maintains WebGL context correctly', () => {
    const { unmount } = renderWithProviders(
      <UserRadar 
        size={400}
        maxRange={TEST_CONSTANTS.MAX_RANGE}
      />
    );

    // Verify WebGL cleanup on unmount
    unmount();
    expect(screen.queryByRole('presentation')).not.toBeInTheDocument();
  });
});