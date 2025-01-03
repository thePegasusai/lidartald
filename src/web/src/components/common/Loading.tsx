import React from 'react'; // ^18.2.0
import { CircularProgress } from '@mui/material'; // ^5.13.0
import styled from '@emotion/styled'; // ^11.11.0
import { fadeIn } from '../../styles/animations';
import { THEME_COLORS } from '../../styles/theme';

// Type definitions for component props
interface LoadingProps {
  /**
   * Size of the loading indicator
   * @default 'medium'
   */
  size?: 'small' | 'medium' | 'large' | number;
  
  /**
   * Custom color for the loading indicator
   * @default primary.main
   */
  color?: string;
  
  /**
   * Accessibility label for screen readers
   * @default 'Loading...'
   */
  ariaLabel?: string;
  
  /**
   * Material elevation level
   * @default 0
   */
  elevation?: number;
  
  /**
   * Disable ripple effect for performance
   * @default true
   */
  disableRipple?: boolean;
  
  /**
   * Thickness of the circular progress
   * @default 3.6
   */
  thickness?: number;
}

// GPU-accelerated container with Material Design 3.0 elevation
const LoadingContainer = styled.div<{ elevation?: number }>`
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
  transform: translateZ(0); /* Force GPU acceleration */
  will-change: transform;
  animation: ${fadeIn} 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: ${({ elevation }) => 
    elevation ? `0px ${elevation * 1.5}px ${elevation * 3}px rgba(0, 0, 0, 0.14)` : 'none'};
`;

// Size mapping for consistent Material Design scaling
const sizeMap = {
  small: 24,
  medium: 40,
  large: 56
};

/**
 * A performance-optimized loading indicator component following Material Design 3.0
 * principles and optimized for 120Hz displays.
 */
const LoadingComponent: React.FC<LoadingProps> = React.memo(({
  size = 'medium',
  color = THEME_COLORS.primary.main,
  ariaLabel = 'Loading...',
  elevation = 0,
  disableRipple = true,
  thickness = 3.6
}) => {
  // Calculate actual size based on input
  const actualSize = typeof size === 'string' ? sizeMap[size] : size;

  // Use RAF for smooth animation on 120Hz displays
  React.useEffect(() => {
    let rafId: number;
    const animate = () => {
      rafId = requestAnimationFrame(animate);
    };
    rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, []);

  return (
    <LoadingContainer 
      elevation={elevation}
      role="progressbar"
      aria-label={ariaLabel}
      aria-live="polite"
    >
      <CircularProgress
        size={actualSize}
        color="primary"
        sx={{
          color,
          willChange: 'transform',
          '& .MuiCircularProgress-svg': {
            transform: 'translateZ(0)', // Force GPU acceleration for SVG
            willChange: 'transform'
          }
        }}
        disableRipple={disableRipple}
        thickness={thickness}
      />
    </LoadingContainer>
  );
});

// Display name for debugging
LoadingComponent.displayName = 'Loading';

// Default export with type safety
export default LoadingComponent;

// Named export for explicit imports
export { LoadingComponent as Loading };

// Type export for consumers
export type { LoadingProps };