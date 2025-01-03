import { createTheme, ThemeOptions, useMediaQuery } from '@mui/material'; // ^5.13.0

// Core color palette constants including LiDAR-specific colors
export const THEME_COLORS = {
  primary: {
    main: '#1976d2',
    light: '#42a5f5',
    dark: '#1565c0',
    contrastText: '#ffffff'
  },
  secondary: {
    main: '#9c27b0',
    light: '#ba68c8',
    dark: '#7b1fa2',
    contrastText: '#ffffff'
  },
  lidar: {
    point: '#00ff00',      // High visibility green for point cloud
    surface: '#4caf50',    // Surface mapping
    feature: '#ff9800',    // Feature highlighting
    highlight: '#ff4081',  // Interactive elements
    background: '#000000', // OLED optimized background
    grid: '#424242'        // Reference grid
  }
};

// Typography configuration optimized for outdoor visibility
const TYPOGRAPHY_CONFIG = {
  fontFamily: 'Roboto, system-ui, sans-serif',
  fontSize: 16,
  fontWeightLight: 300,
  fontWeightRegular: 400,
  fontWeightMedium: 500,
  fontWeightBold: 700,
  h1: {
    fontSize: '2.5rem',
    fontWeight: 500,
    letterSpacing: '-0.01562em'
  }
};

// Create base theme configuration shared between modes
const createBaseTheme = (): ThemeOptions => ({
  typography: TYPOGRAPHY_CONFIG,
  spacing: (factor: number) => `${8 * factor}px`,
  breakpoints: {
    values: {
      xs: 0,
      sm: 600,
      md: 960,
      lg: 1280,
      xl: 1920
    }
  },
  shape: {
    borderRadius: 4
  },
  transitions: {
    duration: {
      shortest: 150,
      shorter: 200,
      short: 250,
      standard: 300,
      complex: 375,
      enteringScreen: 225,
      leavingScreen: 195
    }
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarWidth: 'thin',
          '&::-webkit-scrollbar': {
            width: '6px',
            height: '6px'
          }
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500
        }
      }
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          fontSize: '0.875rem'
        }
      }
    }
  }
});

// Create light theme with outdoor visibility optimizations
export const lightTheme = createTheme({
  ...createBaseTheme(),
  palette: {
    mode: 'light',
    primary: THEME_COLORS.primary,
    secondary: THEME_COLORS.secondary,
    background: {
      default: '#ffffff',
      paper: '#f5f5f5'
    },
    text: {
      primary: 'rgba(0, 0, 0, 0.87)',
      secondary: 'rgba(0, 0, 0, 0.6)'
    },
    lidar: {
      ...THEME_COLORS.lidar,
      point: '#00cc00',    // Adjusted for daylight visibility
      surface: '#2e7d32',  // Darker green for better contrast
      grid: '#9e9e9e'      // Lighter grid for daylight
    }
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none'
        }
      }
    }
  }
});

// Create dark theme optimized for LiDAR visualization
export const darkTheme = createTheme({
  ...createBaseTheme(),
  palette: {
    mode: 'dark',
    primary: {
      ...THEME_COLORS.primary,
      main: '#90caf9'  // Lighter primary for dark mode contrast
    },
    secondary: {
      ...THEME_COLORS.secondary,
      main: '#ce93d8'  // Lighter secondary for dark mode contrast
    },
    background: {
      default: '#000000',  // OLED black
      paper: '#121212'
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.7)'
    },
    lidar: THEME_COLORS.lidar  // Original LiDAR colors optimal for dark mode
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: '#121212'
        }
      }
    }
  }
});

// Custom type declarations for LiDAR-specific theme extensions
declare module '@mui/material/styles' {
  interface Palette {
    lidar: typeof THEME_COLORS.lidar;
  }
  interface PaletteOptions {
    lidar?: Partial<typeof THEME_COLORS.lidar>;
  }
}

// Export theme types for TypeScript support
export type AppTheme = typeof lightTheme;
export type AppPalette = typeof lightTheme.palette;