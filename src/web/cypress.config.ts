import { defineConfig } from 'cypress'; // v12.14.0

export default defineConfig({
  // End-to-end testing configuration
  e2e: {
    // Base URL for e2e tests
    baseUrl: 'http://localhost:3000',
    
    // Test file pattern for e2e tests
    specPattern: 'cypress/e2e/**/*.cy.ts',
    
    // Support file containing custom commands
    supportFile: 'cypress/support/commands.ts',
    
    // Folders for test artifacts
    videosFolder: 'cypress/videos',
    screenshotsFolder: 'cypress/screenshots',
    fixturesFolder: 'cypress/fixtures',
    
    // Viewport configuration for responsive testing
    viewportWidth: 1280,
    viewportHeight: 720,
    
    // Timeouts
    defaultCommandTimeout: 10000,
    pageLoadTimeout: 30000,
    
    // Video recording settings
    video: true,
    videoCompression: 32,
    
    // Retry configuration
    retries: {
      runMode: 2,      // Retries in CI/headless mode
      openMode: 0      // No retries in interactive mode
    },
    
    // Additional e2e-specific settings
    setupNodeEvents(on, config) {
      // Register event listeners and plugins
      on('before:browser:launch', (browser, launchOptions) => {
        // Configure browser-specific settings
        if (browser.name === 'chrome' && browser.isHeadless) {
          launchOptions.args.push('--disable-gpu');
          launchOptions.args.push('--no-sandbox');
        }
        return launchOptions;
      });

      // Configure environment-specific settings
      config.env = {
        ...config.env,
        // Environment variables for LiDAR testing
        MOCK_LIDAR_DATA: true,
        POINT_CLOUD_FIXTURE: 'pointCloud.json',
        // Environment variables for fleet testing
        MOCK_FLEET_DATA: true,
        FLEET_DATA_FIXTURE: 'fleetData.json',
        // Environment variables for user testing
        MOCK_USER_DATA: true,
        USER_DATA_FIXTURE: 'userData.json'
      };

      return config;
    }
  },

  // Component testing configuration
  component: {
    // Development server configuration
    devServer: {
      framework: 'react',
      bundler: 'vite'
    },
    
    // Test file pattern for component tests
    specPattern: 'src/**/*.cy.tsx',
    
    // Support file for component testing
    supportFile: 'cypress/support/component.ts',
    
    // Component-specific settings
    setupNodeEvents(on, config) {
      // Register component test specific plugins
      on('dev-server:start', (options) => {
        // Configure component testing server
        return options;
      });

      return config;
    }
  },

  // Global configuration
  screenshotOnRunFailure: true,
  trashAssetsBeforeRuns: true,
  watchForFileChanges: true,
  
  // Reporter configuration
  reporter: 'mochawesome',
  reporterOptions: {
    reportDir: 'cypress/reports',
    overwrite: false,
    html: true,
    json: true
  }
});