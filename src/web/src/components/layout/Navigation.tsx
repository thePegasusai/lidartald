import React, { useEffect, useState, useMemo } from 'react';
import { BottomNavigation, BottomNavigationAction } from '@mui/material'; // ^5.13.0
import styled from '@emotion/styled'; // ^11.11.0
import { useLocation, useNavigate } from 'react-router-dom'; // ^6.11.0
import { Radar, Games, Group, Map } from '@mui/icons-material'; // ^5.13.0
import { THEME_COLORS } from '../../styles/theme';

// Types and Interfaces
interface NavigationProps {
  orientation: 'portrait' | 'landscape';
  highContrast: boolean;
  fleetSyncEnabled: boolean;
  hapticFeedback: boolean;
  onThemeChange: (theme: string) => void;
}

interface NavigationItem {
  path: string;
  label: string;
  icon: React.ReactNode;
  ariaLabel: string;
  requiresFleet: boolean;
}

// Styled Components
const NavigationContainer = styled.nav<{ orientation: 'portrait' | 'landscape'; highContrast: boolean }>`
  position: fixed;
  z-index: ${({ theme }) => theme.zIndex.appBar};
  transition: all 0.3s ease;
  touch-action: manipulation;
  user-select: none;

  ${({ orientation }) =>
    orientation === 'portrait'
      ? `
    bottom: 0;
    left: 0;
    right: 0;
    height: 64px;
    padding-bottom: env(safe-area-inset-bottom);
  `
      : `
    top: 64px;
    right: 0;
    bottom: 0;
    width: 80px;
    padding-right: env(safe-area-inset-right);
  `}
`;

const StyledBottomNavigation = styled(BottomNavigation)<{
  orientation: 'portrait' | 'landscape';
  highContrast: boolean;
}>`
  height: 100%;
  background-color: ${({ highContrast }) =>
    highContrast ? THEME_COLORS.highContrast.background : THEME_COLORS.primary.main};
  transition: background-color 0.3s ease;
  min-height: 64px;
  gap: 8px;

  ${({ orientation }) =>
    orientation === 'portrait'
      ? `
    flex-direction: row;
    justify-content: space-around;
  `
      : `
    flex-direction: column;
    align-items: center;
  `}

  ${({ highContrast }) =>
    highContrast &&
    `
    border: 2px solid ${THEME_COLORS.highContrast.border};
  `}
`;

const Navigation: React.FC<NavigationProps> = ({
  orientation,
  highContrast,
  fleetSyncEnabled,
  hapticFeedback,
  onThemeChange,
}) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [value, setValue] = useState(location.pathname);

  // Memoized navigation items configuration
  const navigationItems: NavigationItem[] = useMemo(
    () => [
      {
        path: '/social',
        label: 'Social',
        icon: <Radar />,
        ariaLabel: 'Social mode - User discovery and interactions',
        requiresFleet: true,
      },
      {
        path: '/gaming',
        label: 'Gaming',
        icon: <Games />,
        ariaLabel: 'Gaming mode - Reality-based games',
        requiresFleet: false,
      },
      {
        path: '/fleet',
        label: 'Fleet',
        icon: <Group />,
        ariaLabel: 'Fleet mode - Device mesh network management',
        requiresFleet: true,
      },
      {
        path: '/environment',
        label: 'Environment',
        icon: <Map />,
        ariaLabel: 'Environment mode - LiDAR scanning and mapping',
        requiresFleet: false,
      },
    ],
    []
  );

  // Handle navigation changes with haptic feedback and fleet sync
  const handleNavigationChange = async (path: string) => {
    const navItem = navigationItems.find((item) => item.path === path);
    
    if (!navItem) return;

    // Check fleet requirements
    if (navItem.requiresFleet && !fleetSyncEnabled) {
      // Fleet sync required but not available
      return;
    }

    // Trigger haptic feedback if enabled
    if (hapticFeedback) {
      try {
        navigator.vibrate?.(50);
      } catch (error) {
        console.warn('Haptic feedback not supported');
      }
    }

    // Update navigation state
    setValue(path);
    navigate(path);

    // Update theme based on selected mode
    const mode = path.substring(1);
    onThemeChange(mode);
  };

  // Sync with current route on mount and route changes
  useEffect(() => {
    setValue(location.pathname);
  }, [location]);

  return (
    <NavigationContainer orientation={orientation} highContrast={highContrast}>
      <StyledBottomNavigation
        value={value}
        onChange={(_, newValue) => handleNavigationChange(newValue)}
        orientation={orientation}
        highContrast={highContrast}
        showLabels
      >
        {navigationItems.map((item) => (
          <BottomNavigationAction
            key={item.path}
            label={item.label}
            icon={item.icon}
            value={item.path}
            aria-label={item.ariaLabel}
            disabled={item.requiresFleet && !fleetSyncEnabled}
            sx={{
              color: highContrast ? THEME_COLORS.highContrast.text : 'inherit',
              '&.Mui-selected': {
                color: highContrast ? THEME_COLORS.highContrast.active : THEME_COLORS.primary.contrastText,
              },
            }}
          />
        ))}
      </StyledBottomNavigation>
    </NavigationContainer>
  );
};

export default Navigation;