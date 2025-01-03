import React, { useCallback, useMemo, useEffect } from 'react';
import styled from '@emotion/styled';
import { useVirtualizer } from '@tanstack/react-virtual';
import { ErrorBoundary } from 'react-error-boundary';

import ProfileCard from './ProfileCard';
import { useFleetConnection } from '../../hooks/useFleetConnection';
import { UI_CONSTANTS } from '../../config/constants';
import { UserProfile, UserLocation } from '../../types/user.types';
import { FleetStatus } from '../../types/fleet.types';

// Styled components with GPU acceleration
const MatchListContainer = styled.div<{ highContrastMode: boolean }>`
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
  max-height: calc(100vh - 200px);
  overflow-y: auto;
  transform: translateZ(0);
  will-change: transform;
  backface-visibility: hidden;
  
  ${({ highContrastMode }) => highContrastMode && `
    background-color: var(--background-high-contrast);
    border: 2px solid var(--border-high-contrast);
  `}
`;

const VirtualListWrapper = styled.div`
  width: 100%;
  height: 100%;
  position: relative;
  contain: strict;
`;

const EmptyState = styled.div<{ highContrastMode: boolean }>`
  text-align: center;
  padding: 32px;
  color: ${({ highContrastMode }) => 
    highContrastMode ? 'var(--text-high-contrast)' : 'var(--text-secondary)'};
  transition: color 0.2s ease-out;
  user-select: none;
`;

const ErrorFallback = styled.div`
  padding: 16px;
  color: var(--error);
  text-align: center;
`;

interface MatchListProps {
  matches: UserProfile[];
  loading?: boolean;
  onViewProfile: (userId: string) => void;
  onAddToFleet: (userId: string) => void;
  highContrastMode?: boolean;
  refreshRate?: number;
  errorBoundary?: boolean;
}

export const MatchList: React.FC<MatchListProps> = React.memo(({
  matches,
  loading = false,
  onViewProfile,
  onAddToFleet,
  highContrastMode = false,
  refreshRate = UI_CONSTANTS.TARGET_FPS,
  errorBoundary = true
}) => {
  // Fleet connection hook for real-time status
  const { 
    status: fleetStatus, 
    isConnected: fleetConnected,
    joinFleet 
  } = useFleetConnection({
    autoReconnect: true,
    syncInterval: 1000 / refreshRate
  });

  // Virtual list setup for performance
  const parentRef = React.useRef<HTMLDivElement>(null);
  const rowVirtualizer = useVirtualizer({
    count: matches.length,
    getScrollElement: () => parentRef.current,
    estimateSize: useCallback(() => 200, []), // ProfileCard height
    overscan: 5
  });

  // Sort matches by proximity and update frequency
  const sortedMatches = useMemo(() => {
    return [...matches].sort((a, b) => {
      const distanceA = (a as unknown as { location: UserLocation }).location?.distance || Infinity;
      const distanceB = (b as unknown as { location: UserLocation }).location?.distance || Infinity;
      return distanceA - distanceB;
    });
  }, [matches]);

  // Handle fleet interactions
  const handleAddToFleet = useCallback(async (userId: string) => {
    try {
      if (!fleetConnected) {
        await joinFleet(userId);
      }
      onAddToFleet(userId);
    } catch (error) {
      console.error('[MatchList] Fleet join error:', error);
    }
  }, [fleetConnected, joinFleet, onAddToFleet]);

  // Performance optimization for animations
  useEffect(() => {
    if (parentRef.current) {
      parentRef.current.style.transform = 'translateZ(0)';
    }
  }, []);

  // Render empty state
  if (!loading && matches.length === 0) {
    return (
      <EmptyState highContrastMode={highContrastMode}>
        No matches found nearby. Keep scanning!
      </EmptyState>
    );
  }

  const content = (
    <MatchListContainer 
      ref={parentRef} 
      highContrastMode={highContrastMode}
    >
      <VirtualListWrapper
        style={{
          height: `${rowVirtualizer.getTotalSize()}px`,
        }}
      >
        {rowVirtualizer.getVirtualItems().map((virtualRow) => {
          const match = sortedMatches[virtualRow.index];
          return (
            <div
              key={match.userId}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              <ProfileCard
                profile={match}
                location={(match as unknown as { location: UserLocation }).location}
                onMatch={() => onViewProfile(match.userId)}
                onAddToFleet={() => handleAddToFleet(match.userId)}
                highContrastMode={highContrastMode}
                fleetSyncStatus={fleetStatus === FleetStatus.SYNCING ? 'syncing' : 'synced'}
              />
            </div>
          );
        })}
      </VirtualListWrapper>
    </MatchListContainer>
  );

  // Wrap with error boundary if enabled
  return errorBoundary ? (
    <ErrorBoundary
      FallbackComponent={({ error }) => (
        <ErrorFallback>
          Error loading matches: {error.message}
        </ErrorFallback>
      )}
    >
      {content}
    </ErrorBoundary>
  ) : content;
});

MatchList.displayName = 'MatchList';

export default MatchList;