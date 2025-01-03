import React, { useCallback, useMemo } from 'react';
import { Card, CardContent, Typography, Box, IconButton, Tooltip } from '@mui/material'; // ^5.13.0
import styled from '@emotion/styled'; // ^11.11.0
import { debounce } from 'lodash'; // ^4.17.21
import { UserProfile, UserLocation } from '../../types/user.types';
import TaldButton from '../common/Button';

// Constants for experience level thresholds
const EXPERIENCE_LEVELS = {
  1: 0,
  2: 100,
  3: 250,
  4: 500,
  5: 1000
};

// Styled components for enhanced outdoor visibility
const StyledCard = styled(Card)<{ highContrastMode: boolean }>(({ theme, highContrastMode }) => ({
  width: '300px',
  minHeight: '200px',
  backgroundColor: highContrastMode ? theme.palette.background.paper : theme.palette.background.default,
  border: highContrastMode ? `2px solid ${theme.palette.primary.main}` : 'none',
  boxShadow: highContrastMode ? theme.shadows[8] : theme.shadows[2],
  position: 'relative',
  transition: theme.transitions.create(['background-color', 'box-shadow', 'border'], {
    duration: theme.transitions.duration.short
  })
}));

const DistanceIndicator = styled(Typography)<{ highContrastMode: boolean }>(({ theme, highContrastMode }) => ({
  position: 'absolute',
  top: theme.spacing(2),
  right: theme.spacing(2),
  padding: theme.spacing(0.5, 1),
  borderRadius: theme.shape.borderRadius,
  backgroundColor: highContrastMode ? theme.palette.primary.main : theme.palette.primary.light,
  color: theme.palette.primary.contrastText,
  fontWeight: highContrastMode ? 600 : 500
}));

interface ProfileCardProps {
  profile: UserProfile;
  location: UserLocation;
  onMatch?: (userId: string) => void;
  onMessage?: (userId: string) => void;
  onAddToFleet?: (userId: string) => void;
  highContrastMode?: boolean;
  fleetSyncStatus?: 'syncing' | 'synced' | 'error';
}

export const ProfileCard: React.FC<ProfileCardProps> = React.memo(({
  profile,
  location,
  onMatch,
  onMessage,
  onAddToFleet,
  highContrastMode = false,
  fleetSyncStatus
}) => {
  // Format distance with accuracy indicator
  const formattedDistance = useMemo(() => {
    if (!profile.privacySettings.shareLocation) return 'Distance hidden';
    
    const distance = Math.round(location.distance * 10) / 10;
    const accuracy = Math.round(location.accuracy * 100) / 100;
    
    if (distance < 1000) {
      return `${distance}m ±${accuracy}m`;
    }
    return `${(distance / 1000).toFixed(1)}km ±${accuracy}m`;
  }, [location.distance, location.accuracy, profile.privacySettings.shareLocation]);

  // Debounced fleet action handler
  const handleFleetAction = useCallback(
    debounce((action: 'match' | 'message' | 'fleet') => {
      switch (action) {
        case 'match':
          onMatch?.(profile.userId);
          break;
        case 'message':
          onMessage?.(profile.userId);
          break;
        case 'fleet':
          onAddToFleet?.(profile.userId);
          break;
      }
    }, 100),
    [profile.userId, onMatch, onMessage, onAddToFleet]
  );

  return (
    <StyledCard highContrastMode={highContrastMode}>
      <CardContent>
        <Box display="flex" flexDirection="column" gap={2}>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography
              variant="h6"
              fontWeight={highContrastMode ? 600 : 500}
              color={highContrastMode ? 'primary.main' : 'text.primary'}
            >
              {profile.displayName}
            </Typography>
            <DistanceIndicator
              variant="body2"
              highContrastMode={highContrastMode}
            >
              {formattedDistance}
            </DistanceIndicator>
          </Box>

          {profile.privacySettings.shareActivity && (
            <Box>
              <Typography
                variant="body1"
                color={highContrastMode ? 'text.primary' : 'text.secondary'}
              >
                Level {profile.level}
              </Typography>
              <Typography
                variant="body2"
                color={highContrastMode ? 'text.primary' : 'text.secondary'}
              >
                {profile.experience} XP
              </Typography>
            </Box>
          )}

          <Box display="flex" gap={1} mt={2}>
            <TaldButton
              variant="contained"
              size="small"
              color="primary"
              outdoorMode={highContrastMode}
              onClick={() => handleFleetAction('match')}
            >
              Match
            </TaldButton>
            
            <TaldButton
              variant="outlined"
              size="small"
              color="primary"
              outdoorMode={highContrastMode}
              onClick={() => handleFleetAction('message')}
            >
              Message
            </TaldButton>

            {profile.privacySettings.shareFleetHistory && (
              <TaldButton
                variant="outlined"
                size="small"
                color="lidar"
                outdoorMode={highContrastMode}
                fleetSync={fleetSyncStatus === 'syncing'}
                onClick={() => handleFleetAction('fleet')}
                disabled={fleetSyncStatus === 'error'}
              >
                Add to Fleet
              </TaldButton>
            )}
          </Box>
        </Box>
      </CardContent>
    </StyledCard>
  );
});

ProfileCard.displayName = 'ProfileCard';

export default ProfileCard;