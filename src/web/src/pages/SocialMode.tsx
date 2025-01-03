import React, { useState, useCallback, useEffect, useMemo } from 'react';
import styled from '@emotion/styled';
import { useVirtualizer } from '@tanstack/react-virtual';
import { MainLayout } from '../../components/layout/MainLayout';
import UserRadar from '../../components/social/UserRadar';
import { useFleetConnection } from '../../hooks/useFleetConnection';
import { FLEET_CONSTANTS, UI_CONSTANTS } from '../../config/constants';

// Constants based on technical specifications
const RADAR_SIZE = 400; // Radar display size in pixels
const UPDATE_INTERVAL = 33; // 30Hz update frequency
const MAX_RANGE = 5.0; // 5-meter maximum range
const MATCH_LIST_HEIGHT = 400; // Virtual list height

// Styled components with GPU acceleration and outdoor visibility optimization
const SocialModeContainer = styled.div<{ isPortrait: boolean; highContrast: boolean }>`
  display: grid;
  grid-template-columns: ${({ isPortrait }) => isPortrait ? '1fr' : '1fr 1fr'};
  gap: 24px;
  padding: 24px;
  height: 100%;
  transform: translateZ(0);
  backface-visibility: hidden;
  will-change: transform;

  ${({ highContrast, theme }) => highContrast && `
    background-color: ${theme.palette.background.paper};
    color: ${theme.palette.text.primary};
    box-shadow: 0 0 8px rgba(0, 0, 0, 0.3);
  `}
`;

const RadarSection = styled.section`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
`;

const ProfileSection = styled.section<{ highContrast: boolean }>`
  display: flex;
  flex-direction: column;
  gap: 16px;
  background: ${({ theme }) => theme.palette.background.paper};
  border-radius: 8px;
  padding: 16px;
  box-shadow: ${({ theme }) => theme.shadows[4]};

  ${({ highContrast, theme }) => highContrast && `
    border: 2px solid ${theme.palette.divider};
    background: ${theme.palette.background.default};
  `}
`;

const UserList = styled.div`
  height: ${MATCH_LIST_HEIGHT}px;
  overflow: auto;
  border-radius: 4px;
  background: ${({ theme }) => theme.palette.background.default};
`;

const UserCard = styled.div<{ isSelected: boolean; highContrast: boolean }>`
  padding: 16px;
  border-bottom: 1px solid ${({ theme }) => theme.palette.divider};
  cursor: pointer;
  transition: background-color 0.2s ease;
  display: flex;
  align-items: center;
  gap: 12px;

  ${({ isSelected, theme }) => isSelected && `
    background-color: ${theme.palette.primary.light};
    color: ${theme.palette.primary.contrastText};
  `}

  ${({ highContrast, theme }) => highContrast && `
    border: 1px solid ${theme.palette.divider};
    margin-bottom: -1px;
  `}
`;

const FleetControls = styled.div`
  display: flex;
  gap: 12px;
  padding: 16px;
  justify-content: center;
`;

const SocialMode: React.FC = () => {
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null);
  const [isPortrait, setIsPortrait] = useState(window.innerWidth < window.innerHeight);
  const [highContrast, setHighContrast] = useState(false);
  
  const { 
    fleet, 
    isConnected, 
    createFleet, 
    joinFleet, 
    metrics 
  } = useFleetConnection();

  // Virtual list for performance optimization
  const rowVirtualizer = useVirtualizer({
    count: fleet?.participants?.length || 0,
    getScrollElement: () => document.querySelector('.user-list'),
    estimateSize: () => 72,
    overscan: 5
  });

  // Monitor ambient light for contrast adjustment
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-contrast: more)');
    setHighContrast(mediaQuery.matches);
    
    const handler = (e: MediaQueryListEvent) => setHighContrast(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  // Handle orientation changes
  useEffect(() => {
    const handleResize = () => {
      setIsPortrait(window.innerWidth < window.innerHeight);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Optimized user selection handler with debouncing
  const handleUserSelect = useCallback((userId: string) => {
    if (performance.now() - lastSelectTime < UPDATE_INTERVAL) return;
    setSelectedUserId(userId);
    lastSelectTime = performance.now();
  }, []);

  // Enhanced fleet formation handler with error recovery
  const handleAddToFleet = useCallback(async (userId: string) => {
    try {
      if (!fleet) {
        await createFleet('New Fleet');
      } else if (fleet.participants.length < FLEET_CONSTANTS.MAX_DEVICES) {
        await joinFleet(fleet.id);
      }
    } catch (error) {
      console.error('Fleet operation failed:', error);
    }
  }, [fleet, createFleet, joinFleet]);

  // Memoized user list for performance optimization
  const virtualUserList = useMemo(() => {
    if (!fleet?.participants) return null;

    return (
      <UserList className="user-list">
        <div
          style={{
            height: `${rowVirtualizer.getTotalSize()}px`,
            width: '100%',
            position: 'relative'
          }}
        >
          {rowVirtualizer.getVirtualItems().map((virtualRow) => {
            const user = fleet.participants[virtualRow.index];
            return (
              <UserCard
                key={user.participantId}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  transform: `translateY(${virtualRow.start}px)`
                }}
                isSelected={selectedUserId === user.participantId}
                highContrast={highContrast}
                onClick={() => handleUserSelect(user.participantId)}
              >
                <span>{user.displayName}</span>
                <span>{user.proximityData.distance.toFixed(1)}m</span>
              </UserCard>
            );
          })}
        </div>
      </UserList>
    );
  }, [fleet?.participants, selectedUserId, highContrast, rowVirtualizer]);

  return (
    <MainLayout>
      <SocialModeContainer isPortrait={isPortrait} highContrast={highContrast}>
        <RadarSection>
          <UserRadar
            size={RADAR_SIZE}
            onUserSelect={handleUserSelect}
            maxRange={MAX_RANGE}
            highContrast={highContrast}
          />
          <FleetControls>
            {selectedUserId && (
              <button
                onClick={() => handleAddToFleet(selectedUserId)}
                disabled={fleet?.participants.length === FLEET_CONSTANTS.MAX_DEVICES}
              >
                {fleet ? 'Add to Fleet' : 'Create Fleet'}
              </button>
            )}
          </FleetControls>
        </RadarSection>

        <ProfileSection highContrast={highContrast}>
          <h2>Nearby Users ({fleet?.participants.length || 0})</h2>
          {virtualUserList}
          {metrics.averageLatency > 50 && (
            <div role="alert">High network latency detected</div>
          )}
        </ProfileSection>
      </SocialModeContainer>
    </MainLayout>
  );
};

export default SocialMode;