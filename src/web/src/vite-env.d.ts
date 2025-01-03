/// <reference types="vite/client" /> // vite@4.1.0

/**
 * Extended environment variable interface for TALD UNIA platform
 * Augments Vite's default ImportMetaEnv with application-specific configuration
 */
interface ImportMetaEnv extends Vite.ImportMetaEnv {
  /**
   * Backend API endpoint URL for TALD UNIA services
   */
  readonly VITE_API_URL: string;

  /**
   * WebSocket connection URL for real-time fleet communication
   */
  readonly VITE_WS_URL: string;

  /**
   * LiDAR scanning frequency in Hz (default: 30)
   */
  readonly VITE_LIDAR_SCAN_RATE: number;

  /**
   * Maximum number of devices in a fleet (default: 32)
   */
  readonly VITE_FLEET_MAX_SIZE: number;

  /**
   * Environment scanning resolution in cm (default: 0.01)
   */
  readonly VITE_ENVIRONMENT_RESOLUTION: number;

  /**
   * Current deployment environment mode
   */
  readonly MODE: 'development' | 'production' | 'staging';

  /**
   * Development mode flag
   */
  readonly DEV: boolean;

  /**
   * Production mode flag
   */
  readonly PROD: boolean;

  /**
   * Server-side rendering flag
   */
  readonly SSR: boolean;

  /**
   * Base URL for the application
   */
  readonly BASE_URL: string;
}

/**
 * Augment the ImportMeta interface to ensure type safety
 * when accessing environment variables through import.meta.env
 */
interface ImportMeta {
  readonly env: ImportMetaEnv;
}