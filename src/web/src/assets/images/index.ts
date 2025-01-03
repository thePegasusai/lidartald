// React 18.2.0 - Core React for image component types and lazy loading
import { lazy } from 'react';

// Base path for all image assets
const IMAGE_BASE_PATH = '/assets/images/';

/**
 * LiDAR scanning status and feature detection icons
 * Material Design 3.0 compliant with elevation system support
 */
export const lidarIcons = {
  scanActive: `${IMAGE_BASE_PATH}lidar/scan-active.svg`,
  scanPaused: `${IMAGE_BASE_PATH}lidar/scan-paused.svg`,
  featureDetected: `${IMAGE_BASE_PATH}lidar/feature-detected.svg`,
  scanError: `${IMAGE_BASE_PATH}lidar/scan-error.svg`,
  scanComplete: `${IMAGE_BASE_PATH}lidar/scan-complete.svg`
} as const;

/**
 * Fleet management and device connection status icons
 * Includes real-time sync and error state indicators
 */
export const fleetIcons = {
  deviceConnected: `${IMAGE_BASE_PATH}fleet/device-connected.svg`,
  deviceDisconnected: `${IMAGE_BASE_PATH}fleet/device-disconnected.svg`,
  fleetSync: `${IMAGE_BASE_PATH}fleet/fleet-sync.svg`,
  fleetHost: `${IMAGE_BASE_PATH}fleet/fleet-host.svg`,
  fleetMember: `${IMAGE_BASE_PATH}fleet/fleet-member.svg`,
  fleetError: `${IMAGE_BASE_PATH}fleet/fleet-error.svg`
} as const;

/**
 * Social features and user status indicators
 * Includes achievement and level progression icons
 */
export const socialIcons = {
  userOnline: `${IMAGE_BASE_PATH}social/user-online.svg`,
  userOffline: `${IMAGE_BASE_PATH}social/user-offline.svg`,
  matchFound: `${IMAGE_BASE_PATH}social/match-found.svg`,
  userProfile: `${IMAGE_BASE_PATH}social/user-profile.svg`,
  userLevel: `${IMAGE_BASE_PATH}social/user-level.svg`,
  userAchievement: `${IMAGE_BASE_PATH}social/user-achievement.svg`
} as const;

/**
 * Game state and player status indicators
 * Environment-aware game session icons
 */
export const gameIcons = {
  gameActive: `${IMAGE_BASE_PATH}game/game-active.svg`,
  gamePaused: `${IMAGE_BASE_PATH}game/game-paused.svg`,
  playerStatus: `${IMAGE_BASE_PATH}game/player-status.svg`,
  gameEnvironment: `${IMAGE_BASE_PATH}game/game-environment.svg`,
  gameScore: `${IMAGE_BASE_PATH}game/game-score.svg`,
  gameError: `${IMAGE_BASE_PATH}game/game-error.svg`
} as const;

/**
 * Common UI control and status icons
 * Supports both light and dark theme variants
 */
export const uiIcons = {
  settings: `${IMAGE_BASE_PATH}ui/settings.svg`,
  help: `${IMAGE_BASE_PATH}ui/help.svg`,
  close: `${IMAGE_BASE_PATH}ui/close.svg`,
  alert: `${IMAGE_BASE_PATH}ui/alert.svg`,
  darkMode: `${IMAGE_BASE_PATH}ui/dark-mode.svg`,
  lightMode: `${IMAGE_BASE_PATH}ui/light-mode.svg`,
  loading: `${IMAGE_BASE_PATH}ui/loading.svg`,
  error: `${IMAGE_BASE_PATH}ui/error.svg`
} as const;

/**
 * Application logos and branding assets
 * Includes variants for different sizes and themes
 */
export const logos = {
  taldUnia: `${IMAGE_BASE_PATH}logos/tald-unia.svg`,
  taldUniaSmall: `${IMAGE_BASE_PATH}logos/tald-unia-small.svg`,
  taldUniaDark: `${IMAGE_BASE_PATH}logos/tald-unia-dark.svg`,
  taldUniaSmallDark: `${IMAGE_BASE_PATH}logos/tald-unia-small-dark.svg`
} as const;

// Type definitions for strict type checking
export type LidarIconType = keyof typeof lidarIcons;
export type FleetIconType = keyof typeof fleetIcons;
export type SocialIconType = keyof typeof socialIcons;
export type GameIconType = keyof typeof gameIcons;
export type UiIconType = keyof typeof uiIcons;
export type LogoType = keyof typeof logos;

// Lazy-loaded image components for optimized loading
export const LazyImage = lazy(() => import('./components/LazyImage'));