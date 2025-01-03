/**
 * @fileoverview Central index file for Material Design icons used across TALD UNIA platform
 * @version Material Icons v5.13.0
 */

import {
  // Navigation icons
  Menu,
  ArrowBack,
  ArrowForward,
  
  // Mode icons
  Scanner,
  Games,
  Group,
  Map,
  
  // Status icons
  NetworkCheck,
  SignalWifi4Bar,
  Battery90,
  
  // Action icons
  Add,
  Close,
  Refresh,
  Save,
  Share,
  Upload,
  Download,
  
  // Feedback icons
  Info,
  Warning,
  Error,
  Check,
  
  // Theme icons
  LightMode,
  DarkMode,
  Settings,
  
  // Base type
  SvgIconComponent
} from '@mui/icons-material';

/**
 * Type definition for grouped icon exports ensuring type safety
 */
type IconGroup = Record<string, SvgIconComponent>;

/**
 * Navigation-related icon components for consistent UI navigation patterns
 */
export const navigationIcons: IconGroup = {
  Menu,
  ArrowBack,
  ArrowForward,
};

/**
 * Mode-specific icon components for different application features
 */
export const modeIcons: IconGroup = {
  Scanner,
  Games,
  Group,
  Map,
};

/**
 * Status indicator icon components for system state visualization
 */
export const statusIcons: IconGroup = {
  NetworkCheck,
  SignalWifi4Bar,
  Battery90,
};

/**
 * Action-related icon components for user interactions
 */
export const actionIcons: IconGroup = {
  Add,
  Close,
  Refresh,
  Save,
  Share,
  Upload,
  Download,
};

/**
 * Feedback and status icon components for user notifications
 */
export const feedbackIcons: IconGroup = {
  Info,
  Warning,
  Error,
  Check,
};

/**
 * Theme and settings icon components for appearance customization
 */
export const themeIcons: IconGroup = {
  LightMode,
  DarkMode,
  Settings,
};

// Re-export individual icons for direct access if needed
export {
  Menu,
  ArrowBack,
  ArrowForward,
  Scanner,
  Games,
  Group,
  Map,
  NetworkCheck,
  SignalWifi4Bar,
  Battery90,
  Add,
  Close,
  Refresh,
  Save,
  Share,
  Upload,
  Download,
  Info,
  Warning,
  Error,
  Check,
  LightMode,
  DarkMode,
  Settings,
};