import { keyframes } from '@emotion/react'; // @version ^11.11.0

// Animation duration constants optimized for 120Hz displays (8.33ms frame budget)
export const ANIMATION_DURATIONS = {
  fast: '150ms',
  normal: '300ms',
  slow: '450ms'
} as const;

// Material Design timing functions optimized for 120Hz displays
export const TIMING_FUNCTIONS = {
  easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
  easeOut: 'cubic-bezier(0, 0, 0.2, 1)', 
  easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
  emphasized: 'cubic-bezier(0.2, 0, 0, 1)',
  emphasizedDecelerate: 'cubic-bezier(0.05, 0.7, 0.1, 1)',
  emphasizedAccelerate: 'cubic-bezier(0.3, 0, 0.8, 0.15)'
} as const;

// LiDAR scan progress states
export const SCAN_PROGRESS_STEPS = {
  start: '0%',
  processing: '50%',
  complete: '100%',
  error: '-10%'
} as const;

// Helper function to create optimized timing functions
export const createTimingFunction = (type: string, duration: number): string => {
  // Validate animation type against Material Design standards
  const validTypes = ['standard', 'emphasized', 'decelerated', 'accelerated'];
  if (!validTypes.includes(type)) {
    type = 'standard';
  }

  // Calculate optimal control points for 120Hz
  const frameTime = 8.33; // ms per frame at 120Hz
  const frames = Math.round(duration / frameTime);
  
  switch(type) {
    case 'emphasized':
      return `cubic-bezier(0.2, 0, 0, 1)`;
    case 'decelerated': 
      return `cubic-bezier(0.05, 0.7, 0.1, 1)`;
    case 'accelerated':
      return `cubic-bezier(0.3, 0, 0.8, 0.15)`;
    default:
      return `cubic-bezier(0.4, 0, 0.2, 1)`;
  }
};

// GPU-accelerated fade in animation
export const fadeIn = keyframes`
  from {
    opacity: 0;
    transform: translateZ(0);
  }
  to {
    opacity: 1;
    transform: translateZ(0);
  }
`;

// GPU-accelerated fade out animation
export const fadeOut = keyframes`
  from {
    opacity: 1;
    transform: translateZ(0);
  }
  to {
    opacity: 0;
    transform: translateZ(0);
  }
`;

// Material Design compliant scan progress animation
export const scanProgress = keyframes`
  0% {
    transform: translateX(${SCAN_PROGRESS_STEPS.start}) translateZ(0);
  }
  50% {
    transform: translateX(${SCAN_PROGRESS_STEPS.processing}) translateZ(0);
  }
  100% {
    transform: translateX(${SCAN_PROGRESS_STEPS.complete}) translateZ(0);
  }
`;

// Smooth pulsing animation for proximity indicators
export const pulseEffect = keyframes`
  0% {
    transform: scale(1) translateZ(0);
    opacity: 1;
  }
  50% {
    transform: scale(1.1) translateZ(0);
    opacity: 0.7;
  }
  100% {
    transform: scale(1) translateZ(0);
    opacity: 1;
  }
`;

// 3D rotation for point cloud visualization
export const rotatePoint = keyframes`
  from {
    transform: rotate3d(0, 1, 0, 0deg) translateZ(0);
  }
  to {
    transform: rotate3d(0, 1, 0, 360deg) translateZ(0);
  }
`;

// Transform-based slide in animation
export const slideIn = keyframes`
  from {
    transform: translateX(-100%) translateZ(0);
  }
  to {
    transform: translateX(0) translateZ(0);
  }
`;

// Transform-based slide out animation
export const slideOut = keyframes`
  from {
    transform: translateX(0) translateZ(0);
  }
  to {
    transform: translateX(100%) translateZ(0);
  }
`;