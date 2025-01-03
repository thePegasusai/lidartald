import { mockScanResult } from '../fixtures/pointCloud';
import { DEFAULT_SCAN_PARAMETERS } from '../../src/types/lidar.types';

// Constants for test validation
const SCAN_RESOLUTION = 0.01; // cm
const SCAN_RANGE = 5.0; // meters
const SCAN_RATE = 30; // Hz
const MIN_FPS = 60;
const MAX_MEMORY_USAGE = 1024; // MB
const GPU_UTILIZATION_THRESHOLD = 80; // percent

describe('LiDAR Scanning Interface', () => {
  beforeEach(() => {
    // Visit scanner page and initialize test environment
    cy.visit('/scanner');
    cy.intercept('GET', '/api/scan/status', { fixture: 'pointCloud.json' }).as('scanStatus');
    
    // Wait for viewport initialization
    cy.get('[data-cy=lidar-viewport]', { timeout: 10000 })
      .should('be.visible')
      .and('have.attr', 'data-initialized', 'true');

    // Verify WebGL context
    cy.window().then((win) => {
      expect(win.WebGLRenderingContext || win.WebGL2RenderingContext).to.exist;
    });
  });

  it('should display LidarViewport component with correct specifications', () => {
    // Verify viewport container
    cy.get('[data-cy=lidar-viewport]')
      .should('have.css', 'width')
      .and('not.eq', '0px');

    // Check WebGL context initialization
    cy.get('canvas[data-cy=webgl-canvas]')
      .should('exist')
      .and('have.attr', 'data-context-type', 'webgl2');

    // Validate viewport resolution settings
    cy.get('[data-cy=viewport-settings]')
      .should('contain', `${SCAN_RESOLUTION}cm`)
      .and('contain', `${SCAN_RANGE}m`);
  });

  it('should handle scan controls correctly with 30Hz refresh rate', () => {
    // Start scan with default parameters
    cy.get('[data-cy=start-scan]').click();

    // Verify scanning state
    cy.get('[data-cy=scan-status]')
      .should('contain', 'Scanning')
      .and('have.class', 'active');

    // Monitor progress updates at 30Hz
    let lastUpdateTime = Date.now();
    cy.get('[data-cy=scan-progress]')
      .should('be.visible')
      .then(() => {
        const updateInterval = Date.now() - lastUpdateTime;
        expect(updateInterval).to.be.closeTo(1000 / SCAN_RATE, 5);
      });

    // Verify scan parameter controls are disabled during scan
    cy.get('[data-cy=scan-resolution]').should('be.disabled');
    cy.get('[data-cy=scan-range]').should('be.disabled');

    // Stop scan
    cy.get('[data-cy=stop-scan]').click();

    // Verify scan data persistence
    cy.get('[data-cy=point-cloud-stats]')
      .should('contain', 'Points:')
      .and('not.contain', '0');
  });

  it('should render point cloud data with 60 FPS minimum', () => {
    // Load test point cloud data
    cy.window().then((win) => {
      win.postMessage({ type: 'LOAD_POINT_CLOUD', data: mockScanResult.pointCloud }, '*');
    });

    // Verify point cloud buffer allocation
    cy.get('[data-cy=point-cloud-viewer]')
      .should('have.attr', 'data-points-loaded', 'true');

    // Monitor frame rate
    cy.get('[data-cy=performance-stats]')
      .should('be.visible')
      .and((stats) => {
        const fps = parseInt(stats.text().match(/FPS: (\d+)/)[1]);
        expect(fps).to.be.at.least(MIN_FPS);
      });

    // Verify point cloud transformation
    cy.get('[data-cy=transform-matrix]')
      .should('have.length', 16)
      .and('deep.equal', mockScanResult.pointCloud.transformMatrix);
  });

  it('should detect and classify features accurately', () => {
    // Load feature detection test data
    cy.window().then((win) => {
      win.postMessage({ type: 'LOAD_FEATURES', data: mockScanResult.features }, '*');
    });

    // Verify feature detection initialization
    cy.get('[data-cy=feature-overlay]')
      .should('be.visible')
      .and('have.attr', 'data-features-loaded', 'true');

    // Check feature classification accuracy
    cy.get('[data-cy=feature-list] .feature-item').each(($feature, index) => {
      const expectedFeature = mockScanResult.features[index];
      cy.wrap($feature)
        .should('contain', expectedFeature.type)
        .and('contain', `${(expectedFeature.confidence * 100).toFixed(1)}%`);
    });
  });

  it('should maintain system performance requirements', () => {
    // Initialize performance monitoring
    cy.window().then((win) => {
      win.performance.mark('scan-start');
    });

    // Start continuous scanning
    cy.performLidarScan(SCAN_RESOLUTION, SCAN_RANGE, SCAN_RATE);

    // Monitor frame rate consistency
    cy.get('[data-cy=performance-monitor]', { timeout: 10000 })
      .should('be.visible')
      .and((monitor) => {
        const metrics = JSON.parse(monitor.attr('data-metrics'));
        expect(metrics.fps).to.be.at.least(MIN_FPS);
        expect(metrics.memoryUsage).to.be.below(MAX_MEMORY_USAGE);
        expect(metrics.gpuUtilization).to.be.below(GPU_UTILIZATION_THRESHOLD);
      });

    // Verify scan rate consistency
    cy.get('[data-cy=scan-rate-monitor]')
      .should('contain', `${SCAN_RATE}Hz`);
  });

  it('should handle error conditions gracefully', () => {
    // Simulate hardware disconnection
    cy.window().then((win) => {
      win.postMessage({ type: 'SIMULATE_ERROR', error: 'HARDWARE_DISCONNECTED' }, '*');
    });

    // Verify error handling
    cy.get('[data-cy=error-message]')
      .should('be.visible')
      .and('contain', 'Hardware disconnected');

    // Check recovery UI
    cy.get('[data-cy=retry-scan]')
      .should('be.visible')
      .and('not.be.disabled');

    // Verify error logging
    cy.get('[data-cy=error-log]')
      .should('contain', 'Hardware disconnected')
      .and('contain', new Date().toLocaleDateString());
  });
});