import { mockScanResult } from '../fixtures/pointCloud';
import '@testing-library/cypress'; // v9.0.0
import { FeatureType } from '../../../src/types/lidar.types';

// Constants for test configuration
const ENVIRONMENT_UPDATE_INTERVAL = 33; // ~30Hz
const UI_UPDATE_INTERVAL = 16; // ~60 FPS
const MAX_FLEET_SIZE = 32;
const POINT_CLOUD_RESOLUTION = 0.01;

describe('Gaming Mode', () => {
  beforeEach(() => {
    // Configure browser for WebGL support
    cy.visit('/game', {
      onBeforeLoad: (win) => {
        win.WebGL2RenderingContext = true;
      }
    });

    // Mock API responses
    cy.intercept('GET', '/api/environment/current', {
      statusCode: 200,
      body: mockScanResult
    }).as('getEnvironment');

    cy.intercept('POST', '/api/fleet/sync', {
      statusCode: 200,
      body: { status: 'synchronized' }
    }).as('fleetSync');

    // Wait for environment initialization
    cy.get('[data-testid="environment-map"]', { timeout: 10000 })
      .should('be.visible')
      .and('have.attr', 'data-initialized', 'true');
  });

  it('should render environment map with performance metrics', () => {
    // Verify WebGL canvas initialization
    cy.get('[data-testid="environment-map-canvas"]')
      .should('have.attr', 'data-webgl', 'true');

    // Test point cloud rendering
    cy.window().then((win) => {
      const canvas = win.document.querySelector('[data-testid="environment-map-canvas"]');
      const gl = canvas.getContext('webgl2');
      expect(gl).to.not.be.null;

      // Monitor frame rate
      let frames = 0;
      const startTime = performance.now();
      
      const checkPerformance = () => {
        frames++;
        if (performance.now() - startTime >= 1000) {
          const fps = Math.round(frames * 1000 / (performance.now() - startTime));
          expect(fps).to.be.at.least(60, 'Should maintain 60 FPS minimum');
          return;
        }
        requestAnimationFrame(checkPerformance);
      };
      
      requestAnimationFrame(checkPerformance);
    });

    // Verify point cloud resolution
    cy.get('[data-testid="resolution-display"]')
      .should('contain', POINT_CLOUD_RESOLUTION)
      .and('have.attr', 'data-quality', 'ultra-high');

    // Test feature detection accuracy
    cy.get('[data-testid="detected-features"]').within(() => {
      // Verify surface detection
      cy.get(`[data-feature-type="${FeatureType.SURFACE}"]`)
        .should('have.length.at.least', 1)
        .and('have.attr', 'data-confidence')
        .and('be.gt', 0.9);

      // Verify obstacle detection
      cy.get(`[data-feature-type="${FeatureType.OBSTACLE}"]`)
        .should('have.length.at.least', 1)
        .and('have.attr', 'data-confidence')
        .and('be.gt', 0.9);
    });

    // Verify environment update rate
    cy.get('[data-testid="update-rate"]')
      .should('contain', '30')
      .and('have.attr', 'data-status', 'optimal');
  });

  it('should manage fleet synchronization', () => {
    // Initialize fleet network
    cy.get('[data-testid="fleet-manager"]')
      .should('be.visible')
      .and('have.attr', 'data-status', 'ready');

    // Test device connections
    const mockDevices = Array.from({ length: MAX_FLEET_SIZE }, (_, i) => ({
      id: `device-${i + 1}`,
      status: 'connected'
    }));

    cy.window().then((win) => {
      win.postMessage({ type: 'MOCK_FLEET_DEVICES', devices: mockDevices }, '*');
    });

    // Verify fleet size limit
    cy.get('[data-testid="connected-devices"]')
      .should('have.length', MAX_FLEET_SIZE)
      .and('have.attr', 'data-network-status', 'optimal');

    // Test environment synchronization
    cy.get('[data-testid="sync-status"]')
      .should('contain', 'Synchronized')
      .and('have.attr', 'data-latency')
      .and('be.lt', 50); // Less than 50ms latency

    // Verify mesh network topology
    cy.get('[data-testid="network-topology"]').within(() => {
      cy.get('[data-testid="mesh-connection"]')
        .should('have.length.at.least', MAX_FLEET_SIZE - 1);
      
      cy.get('[data-testid="network-latency"]')
        .each(($el) => {
          const latency = parseInt($el.attr('data-value'));
          expect(latency).to.be.lessThan(50, 'Mesh network latency should be under 50ms');
        });
    });

    // Test reconnection handling
    cy.window().then((win) => {
      win.postMessage({ type: 'MOCK_DEVICE_DISCONNECT', deviceId: 'device-1' }, '*');
    });

    cy.get('[data-testid="device-device-1"]')
      .should('have.attr', 'data-status', 'reconnecting');

    cy.window().then((win) => {
      win.postMessage({ type: 'MOCK_DEVICE_RECONNECT', deviceId: 'device-1' }, '*');
    });

    cy.get('[data-testid="device-device-1"]')
      .should('have.attr', 'data-status', 'connected');
  });

  it('should handle game session management', () => {
    // Initialize game session
    cy.get('[data-testid="start-game"]').click();

    // Verify environment preparation
    cy.get('[data-testid="environment-status"]')
      .should('contain', 'Ready')
      .and('have.attr', 'data-scan-quality')
      .and('be.gt', 0.95);

    // Test player spawning
    cy.get('[data-testid="player-spawn-points"]')
      .should('have.length.at.least', 1)
      .and('have.attr', 'data-validation', 'complete');

    // Verify game state synchronization
    cy.get('[data-testid="game-state"]')
      .should('have.attr', 'data-sync-status', 'active')
      .and('have.attr', 'data-update-rate')
      .and('be.gte', 30);

    // Test game session persistence
    cy.window().then((win) => {
      win.postMessage({ type: 'MOCK_CONNECTION_INTERRUPT', duration: 1000 }, '*');
    });

    cy.get('[data-testid="game-state"]')
      .should('have.attr', 'data-sync-status', 'reconnecting')
      .and('have.attr', 'data-state-preserved', 'true');

    // Verify session recovery
    cy.get('[data-testid="game-state"]', { timeout: 5000 })
      .should('have.attr', 'data-sync-status', 'active')
      .and('have.attr', 'data-state-recovered', 'true');
  });
});