import React from 'react';
import { Button as MuiButton } from '@mui/material'; // ^5.13.0
import styled from '@emotion/styled'; // ^11.11.0
import { useTheme } from '@mui/material'; // ^5.13.0
import { THEME_COLORS } from '../../styles/theme';

// Button size configurations with elevation mapping
const BUTTON_SIZES = {
  small: {
    height: '32px',
    padding: '0 12px',
    fontSize: '14px',
    elevation: 2,
  },
  medium: {
    height: '40px',
    padding: '0 16px',
    fontSize: '16px',
    elevation: 4,
  },
  large: {
    height: '48px',
    padding: '0 24px',
    fontSize: '18px',
    elevation: 6,
  },
} as const;

// Contrast ratios for outdoor visibility optimization
const OUTDOOR_CONTRAST_RATIOS = {
  normal: 1.5,  // 4.5:1 ratio
  high: 2,      // 7:1 ratio
  ultra: 2.5,   // 10:1 ratio
};

interface TaldButtonProps {
  variant?: 'contained' | 'outlined' | 'text';
  size?: keyof typeof BUTTON_SIZES;
  color?: 'primary' | 'secondary' | 'lidar';
  disabled?: boolean;
  fullWidth?: boolean;
  outdoorMode?: boolean;
  fleetSync?: boolean;
  elevation?: number;
  children: React.ReactNode;
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  className?: string;
}

// Enhanced styled button with outdoor visibility optimizations
const StyledButton = styled(MuiButton)<TaldButtonProps>(
  ({ theme, variant, size = 'medium', color = 'primary', outdoorMode, fleetSync, elevation }) => {
    const sizeConfig = BUTTON_SIZES[size];
    const contrastMultiplier = outdoorMode ? OUTDOOR_CONTRAST_RATIOS.high : 1;
    const baseColor = THEME_COLORS[color];

    return {
      height: sizeConfig.height,
      padding: sizeConfig.padding,
      fontSize: sizeConfig.fontSize,
      fontWeight: 500,
      letterSpacing: '0.02857em',
      boxShadow: theme.shadows[elevation ?? sizeConfig.elevation],
      transition: theme.transitions.create(
        ['background-color', 'box-shadow', 'border-color', 'color'],
        { duration: theme.transitions.duration.short }
      ),

      // Variant-specific styles with outdoor visibility enhancement
      ...(variant === 'contained' && {
        backgroundColor: baseColor.main,
        color: baseColor.contrastText,
        '&:hover': {
          backgroundColor: baseColor.dark,
          boxShadow: theme.shadows[(elevation ?? sizeConfig.elevation) + 1],
        },
      }),

      ...(variant === 'outlined' && {
        borderColor: baseColor.main,
        borderWidth: outdoorMode ? '2px' : '1px',
        color: baseColor.main,
        '&:hover': {
          backgroundColor: `${baseColor.main}1A`, // 10% opacity
          borderColor: baseColor.dark,
        },
      }),

      // Enhanced contrast for outdoor visibility
      ...(outdoorMode && {
        fontWeight: 600,
        color: variant === 'contained' 
          ? baseColor.contrastText
          : baseColor.dark,
        textShadow: variant === 'contained'
          ? '0 1px 2px rgba(0,0,0,0.3)'
          : 'none',
      }),

      // Fleet sync visual indicator
      ...(fleetSync && {
        '&::after': {
          content: '""',
          position: 'absolute',
          top: '4px',
          right: '4px',
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor: THEME_COLORS.lidar.point,
          animation: 'pulse 1.5s infinite',
        },
        '@keyframes pulse': {
          '0%': { opacity: 1 },
          '50%': { opacity: 0.5 },
          '100%': { opacity: 1 },
        },
      }),

      // Disabled state with maintained visibility
      '&.Mui-disabled': {
        backgroundColor: variant === 'contained' 
          ? theme.palette.action.disabledBackground 
          : 'transparent',
        color: theme.palette.action.disabled,
        opacity: outdoorMode ? 0.7 : 0.5,
      },

      // High-performance ripple effect
      '& .MuiTouchRipple-root': {
        color: variant === 'contained' 
          ? baseColor.light 
          : baseColor.main,
      },
    };
  }
);

// Memoized button component for performance optimization
export const TaldButton = React.memo(
  React.forwardRef<HTMLButtonElement, TaldButtonProps>((props, ref) => {
    const {
      variant = 'contained',
      size = 'medium',
      color = 'primary',
      disabled = false,
      fullWidth = false,
      outdoorMode = false,
      fleetSync = false,
      elevation,
      children,
      onClick,
      className,
      ...rest
    } = props;

    const theme = useTheme();

    // Debounced click handler for fleet sync operations
    const handleClick = React.useCallback(
      (event: React.MouseEvent<HTMLButtonElement>) => {
        if (disabled || !onClick) return;
        onClick(event);
      },
      [disabled, onClick]
    );

    return (
      <StyledButton
        ref={ref}
        variant={variant}
        size={size}
        color={color}
        disabled={disabled}
        fullWidth={fullWidth}
        outdoorMode={outdoorMode}
        fleetSync={fleetSync}
        elevation={elevation}
        onClick={handleClick}
        className={className}
        disableElevation={false}
        disableRipple={false}
        {...rest}
      >
        {children}
      </StyledButton>
    );
  })
);

TaldButton.displayName = 'TaldButton';

export default TaldButton;