import { users, authStates, fleetConfigs } from '../fixtures/userData';
import { UserRole, UserLocation } from '../../../src/types/user.types';
import { FleetStatus, FleetRole } from '../../../src/types/fleet.types';
import { UPDATE_INTERVAL_MS } from '../../../src/types/environment.types';

// Constants for test configuration
const SCAN_RATE = 30; // 30Hz scan rate
const SCAN_RESOLUTION = 0.01; // 0.01cm resolution
const WS_LATENCY = 50; // 50ms network latency
const RADAR_UPDATE_INTERVAL = UPDATE_INTERVAL_MS; // ~30Hz updates

describe('Social Mode Interface', () => {
  beforeEach(() => {
    // Reset application state
    cy.clearLocalStorage();
    cy.clearCookies();

    // Mock WebSocket connection
    mockWebSocket(WS_LATENCY);

    // Mock LiDAR scanner
    mockLidarScanner(SCAN_RATE, SCAN_RESOLUTION);

    // Set authenticated user state
    cy.window().then((win) => {
      win.localStorage.setItem('authState', JSON.stringify(authStates.authenticated));
    });

    // Visit social mode page
    cy.visit('/social');
    cy.wait('@wsConnection');
  });

  it('displays user radar with nearby users', () => {
    // Verify radar component initialization
    cy.get('[data-testid=user-radar]')
      .should('be.visible')
      .and('have.attr', 'data-update-rate', SCAN_RATE.toString());

    // Check user indicators
    users.forEach(user => {
      if (user.location.isVisible) {
        cy.get(`[data-testid=user-indicator-${user.id}]`)
          .should('be.visible')
          .and('have.attr', 'data-distance', user.location.distance.toString())
          .and('have.css', 'transform', `translate(${user.location.coordinates.lat}px, ${user.location.coordinates.lng}px)`);
      }
    });

    // Verify radar zoom controls
    cy.get('[data-testid=radar-zoom-in]').click();
    cy.get('[data-testid=user-radar]')
      .should('have.attr', 'data-zoom-level', '2');
  });

  it('handles real-time position updates', () => {
    // Monitor update frequency
    let lastUpdateTime = Date.now();
    let updateCount = 0;

    cy.get('[data-testid=user-radar]')
      .then($radar => {
        const observer = new MutationObserver(() => {
          const now = Date.now();
          const interval = now - lastUpdateTime;
          expect(interval).to.be.closeTo(RADAR_UPDATE_INTERVAL, 5);
          lastUpdateTime = now;
          updateCount++;
        });

        observer.observe($radar[0], {
          attributes: true,
          attributeFilter: ['data-last-update']
        });

        // Wait for multiple updates
        cy.wait(1000).then(() => {
          expect(updateCount).to.be.closeTo(30, 3); // ~30 updates per second
          observer.disconnect();
        });
      });
  });

  it('displays user profile cards on selection', () => {
    const testUser = users[0];

    cy.get(`[data-testid=user-indicator-${testUser.id}]`).click();

    cy.get('[data-testid=profile-card]')
      .should('be.visible')
      .within(() => {
        cy.get('[data-testid=display-name]')
          .should('have.text', testUser.profile.displayName);
        cy.get('[data-testid=user-level]')
          .should('have.text', `Level ${testUser.profile.level}`);
        cy.get('[data-testid=last-active]')
          .should('contain', 'Last active');
      });
  });

  it('supports fleet formation with nearby users', () => {
    const hostUser = users[0];
    const memberUser = users[1];

    // Select users and form fleet
    cy.get(`[data-testid=user-indicator-${memberUser.id}]`).click();
    cy.get('[data-testid=invite-to-fleet]').click();

    // Verify fleet formation request
    cy.wait('@wsMessage').its('request.body').should('deep.equal', {
      type: 'FLEET_INVITE',
      targetUserId: memberUser.id,
      fleetConfig: {
        hostId: hostUser.id,
        maxDevices: 32,
        scanRate: SCAN_RATE
      }
    });

    // Mock fleet formation success
    cy.window().then((win) => {
      win.postMessage({
        type: 'FLEET_FORMED',
        fleetId: 'test-fleet-1',
        status: FleetStatus.ACTIVE,
        members: [
          { userId: hostUser.id, role: FleetRole.HOST },
          { userId: memberUser.id, role: FleetRole.MEMBER }
        ]
      }, '*');
    });

    // Verify fleet UI updates
    cy.get('[data-testid=fleet-status]')
      .should('be.visible')
      .and('contain', 'Fleet Active');
  });

  it('maintains performance under load with multiple users', () => {
    // Generate additional mock users
    const mockUsers = Array.from({ length: 20 }, (_, i) => ({
      ...users[0],
      id: `load-test-user-${i}`,
      location: {
        ...users[0].location,
        distance: Math.random() * 5,
        coordinates: {
          lat: Math.random() * 10 - 5,
          lng: Math.random() * 10 - 5
        }
      }
    }));

    // Inject mock users into radar
    cy.window().then((win) => {
      win.postMessage({
        type: 'USERS_DETECTED',
        users: mockUsers
      }, '*');
    });

    // Verify radar performance
    cy.get('[data-testid=user-radar]')
      .should('have.attr', 'data-user-count', mockUsers.length.toString())
      .and('have.attr', 'data-frame-time')
      .then((frameTime) => {
        expect(parseInt(frameTime)).to.be.lessThan(16); // Maintain 60fps
      });
  });
});

// Mock WebSocket server for real-time updates
function mockWebSocket(latency: number) {
  cy.intercept('ws://*/social', (req) => {
    req.reply({
      websocket: true,
      onMessage: (msg) => {
        // Simulate network latency
        setTimeout(() => {
          const data = JSON.parse(msg.toString());
          handleWebSocketMessage(data);
        }, latency);
      }
    });
  }).as('wsConnection');
}

// Mock LiDAR scanner functionality
function mockLidarScanner(scanRate: number, resolution: number) {
  cy.intercept('/api/v1/lidar/scan', (req) => {
    req.reply({
      statusCode: 200,
      body: {
        scanId: `scan-${Date.now()}`,
        timestamp: Date.now(),
        resolution,
        users: users.map(u => ({
          userId: u.id,
          distance: u.location.distance,
          coordinates: u.location.coordinates,
          lastUpdate: Date.now()
        })),
        updateRate: scanRate
      },
      headers: {
        'Content-Type': 'application/json'
      },
      delay: Math.floor(1000 / scanRate)
    });
  }).as('lidarScan');
}

// WebSocket message handler
function handleWebSocketMessage(data: any) {
  switch (data.type) {
    case 'USER_UPDATE':
      cy.get('[data-testid=user-radar]')
        .trigger('user-update', { detail: data });
      break;
    case 'FLEET_SYNC':
      cy.get('[data-testid=fleet-status]')
        .trigger('fleet-sync', { detail: data });
      break;
    case 'ERROR':
      cy.get('[data-testid=error-toast]')
        .trigger('error', { detail: data });
      break;
  }
}