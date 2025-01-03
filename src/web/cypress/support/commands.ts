import { UserRole, UserPrivacySettings } from '../../src/types/user.types';
import { Fleet, FleetStatus } from '../../src/types/fleet.types';
import { validateUserProfile } from '../../src/utils/validation';
import { DEFAULT_SCAN_PARAMETERS } from '../../src/types/lidar.types';
import '@faker-js/faker/locale/en'; // v8.x
import { faker } from '@faker-js/faker';

// Type definitions for custom Cypress commands
declare global {
  namespace Cypress {
    interface Chainable {
      login(email: string, password: string, role?: UserRole, privacySettings?: UserPrivacySettings): Chainable<void>;
      createFleet(fleetName: string, maxDevices: number, meshConfig: any): Chainable<void>;
      performLidarScan(resolution?: number, range?: number, scanRate?: number): Chainable<void>;
    }
  }
}

// Default test data
const DEFAULT_USER = {
  email: 'test@tald.com',
  password: 'testpass123',
  role: UserRole.BASIC_USER,
  privacySettings: {
    shareLocation: false,
    shareActivity: false,
    shareFleetHistory: false,
    dataRetentionDays: 7
  }
};

const DEFAULT_FLEET = {
  name: 'Test Fleet',
  maxDevices: 4,
  meshConfig: {
    topology: 'mesh',
    encryption: 'aes256'
  }
};

/**
 * Custom command for user authentication with comprehensive security validation
 */
Cypress.Commands.add('login', (
  email = DEFAULT_USER.email,
  password = DEFAULT_USER.password,
  role = DEFAULT_USER.role,
  privacySettings = DEFAULT_USER.privacySettings
) => {
  // Validate user profile data
  validateUserProfile({ email, role, privacySettings });

  cy.intercept('POST', '/api/auth/login').as('loginRequest');

  cy.visit('/login', {
    headers: {
      'X-CSRF-Token': 'test-csrf-token',
      'X-Frame-Options': 'DENY'
    }
  });

  // Input validation and XSS prevention
  cy.get('[data-cy=email-input]')
    .should('be.visible')
    .type(email, { delay: 50 });

  cy.get('[data-cy=password-input]')
    .should('be.visible')
    .type(password, { delay: 50 });

  // Role-based access validation
  cy.get('[data-cy=role-select]')
    .should('be.visible')
    .select(role);

  // Privacy settings configuration
  Object.entries(privacySettings).forEach(([setting, value]) => {
    cy.get(`[data-cy=privacy-${setting}]`).click();
  });

  cy.get('[data-cy=login-submit]').click();

  // Validate authentication response and security context
  cy.wait('@loginRequest').then((interception) => {
    expect(interception.response?.statusCode).to.eq(200);
    expect(interception.response?.headers['x-frame-options']).to.eq('DENY');
    expect(interception.response?.body).to.have.property('token');
  });

  // Verify successful authentication
  cy.url().should('not.include', '/login');
  cy.getCookie('auth_token').should('exist');
});

/**
 * Custom command for fleet creation and mesh network validation
 */
Cypress.Commands.add('createFleet', (
  fleetName = DEFAULT_FLEET.name,
  maxDevices = DEFAULT_FLEET.maxDevices,
  meshConfig = DEFAULT_FLEET.meshConfig
) => {
  cy.intercept('POST', '/api/fleet/create').as('fleetCreate');

  // Navigate to fleet creation
  cy.get('[data-cy=fleet-menu]').click();
  cy.get('[data-cy=create-fleet]').click();

  // Validate fleet parameters
  cy.get('[data-cy=fleet-name]')
    .should('be.visible')
    .type(fleetName);

  cy.get('[data-cy=max-devices]')
    .should('be.visible')
    .type(maxDevices.toString())
    .should('have.value', maxDevices)
    .and('have.attr', 'max', '32');

  // Configure mesh network
  cy.get('[data-cy=mesh-topology]').select(meshConfig.topology);
  cy.get('[data-cy=mesh-encryption]').select(meshConfig.encryption);

  cy.get('[data-cy=create-fleet-submit]').click();

  // Validate fleet creation and mesh network establishment
  cy.wait('@fleetCreate').then((interception) => {
    expect(interception.response?.statusCode).to.eq(200);
    const fleet: Fleet = interception.response?.body;
    expect(fleet.status).to.eq(FleetStatus.INITIALIZING);
    expect(fleet.maxDevices).to.be.lte(32);
  });

  // Verify fleet status
  cy.get('[data-cy=fleet-status]')
    .should('be.visible')
    .and('contain', FleetStatus.ACTIVE);
});

/**
 * Custom command for LiDAR scanning with parameter validation
 */
Cypress.Commands.add('performLidarScan', (
  resolution = DEFAULT_SCAN_PARAMETERS.resolution,
  range = DEFAULT_SCAN_PARAMETERS.range,
  scanRate = DEFAULT_SCAN_PARAMETERS.scanRate
) => {
  cy.intercept('POST', '/api/scan/start').as('startScan');
  cy.intercept('GET', '/api/scan/status').as('scanStatus');

  // Navigate to scanner interface
  cy.get('[data-cy=scanner-menu]').click();

  // Configure scan parameters
  cy.get('[data-cy=scan-resolution]')
    .should('be.visible')
    .type(resolution.toString())
    .should('have.value', resolution)
    .and('have.attr', 'min', '0.01');

  cy.get('[data-cy=scan-range]')
    .should('be.visible')
    .type(range.toString())
    .should('have.value', range)
    .and('have.attr', 'max', '5.0');

  cy.get('[data-cy=scan-rate]')
    .should('be.visible')
    .type(scanRate.toString())
    .should('have.value', scanRate);

  // Start scan
  cy.get('[data-cy=start-scan]').click();

  // Monitor scan progress
  cy.wait('@startScan').then((interception) => {
    expect(interception.response?.statusCode).to.eq(200);
  });

  // Validate scan completion and data quality
  cy.get('[data-cy=scan-progress]')
    .should('be.visible')
    .and('have.attr', 'aria-valuenow', '100');

  cy.get('[data-cy=point-cloud-viewer]')
    .should('be.visible')
    .and('have.attr', 'data-points-count')
    .and('be.gt', 0);

  // Verify scan quality metrics
  cy.get('[data-cy=scan-quality]')
    .should('be.visible')
    .and('have.text', 'High Quality');
});