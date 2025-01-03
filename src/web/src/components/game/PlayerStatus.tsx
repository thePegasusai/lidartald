import React, { useMemo, useCallback, useRef, useEffect } from 'react';
import { useSelector } from 'react-redux';
import { styled } from '@mui/material/styles';
import { Box, Typography, CircularProgress } from '@mui/material';
import { Virtuoso } from 'react-virtuoso'; // v1.0.0-alpha.21

import { PlayerState, GameStatus } from '../../types/game.types';
import { useGameSession } from '../../hooks/useGameSession';
import { selectPlayerStates } from '../../store/slices/gameSlice';

// Constants for performance optimization
const UPDATE_INTERVAL_MS = 16; // 60 FPS
const MAX_PLAYERS = 32;
const ANIMATION_DURATION_MS = 150;

// Styled components for optimized rendering
const PlayerStatusContainer = styled(Box)(({ theme }) => ({
    width: '100%',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: theme.palette.background.paper,
    borderRadius: theme.shape.borderRadius,
    overflow: 'hidden'
}));

const PlayerListContainer = styled(Box)({
    flex: 1,
    overflow: 'hidden'
});

const PlayerItem = styled(Box, {
    shouldForwardProp: prop => prop !== 'isActive'
})<{ isActive: boolean }>(({ theme, isActive }) => ({
    display: 'flex',
    alignItems: 'center',
    padding: theme.spacing(1, 2),
    borderBottom: `1px solid ${theme.palette.divider}`,
    backgroundColor: isActive ? theme.palette.action.selected : 'transparent',
    transition: `background-color ${ANIMATION_DURATION_MS}ms ease-in-out`,
    '&:hover': {
        backgroundColor: theme.palette.action.hover
    }
}));

const PlayerScore = styled(Typography)(({ theme }) => ({
    marginLeft: 'auto',
    fontWeight: 'bold',
    color: theme.palette.primary.main
}));

const PlayerPosition = styled(Typography)(({ theme }) => ({
    color: theme.palette.text.secondary,
    fontSize: '0.875rem',
    marginLeft: theme.spacing(2)
}));

/**
 * Memoized utility function for sorting players by score
 */
const sortPlayersByScore = (players: PlayerState[]): PlayerState[] => {
    return [...players].sort((a, b) => b.score - a.score);
};

/**
 * Custom hook for managing player status updates
 */
const usePlayerUpdates = (players: PlayerState[]) => {
    const frameRef = useRef<number>();
    const lastUpdateRef = useRef<number>(0);

    useEffect(() => {
        const updateFrame = (timestamp: number) => {
            if (timestamp - lastUpdateRef.current >= UPDATE_INTERVAL_MS) {
                lastUpdateRef.current = timestamp;
                // Additional frame-based updates can be implemented here
            }
            frameRef.current = requestAnimationFrame(updateFrame);
        };

        frameRef.current = requestAnimationFrame(updateFrame);

        return () => {
            if (frameRef.current) {
                cancelAnimationFrame(frameRef.current);
            }
        };
    }, [players]);
};

/**
 * PlayerStatus component for displaying real-time player information
 * Optimized for 60 FPS performance with up to 32 players
 */
const PlayerStatus: React.FC = React.memo(() => {
    // Access game session data with optimized hook
    const { status: gameStatus, updateInterval } = useGameSession();
    
    // Get player states through memoized selector
    const players = useSelector(selectPlayerStates);

    // Setup performance monitoring
    usePlayerUpdates(players);

    // Memoize sorted players for optimal rendering
    const sortedPlayers = useMemo(() => 
        sortPlayersByScore(players),
        [players]
    );

    // Memoized render function for virtualized list items
    const renderPlayer = useCallback((index: number) => {
        const player = sortedPlayers[index];
        const isActive = player.status === 'active';

        return (
            <PlayerItem isActive={isActive}>
                <Typography variant="body1">
                    {`#${index + 1} ${player.playerId}`}
                </Typography>
                <PlayerPosition variant="body2">
                    {`(${player.position.x.toFixed(2)}, ${player.position.y.toFixed(2)}, ${player.position.z.toFixed(2)})`}
                </PlayerPosition>
                <PlayerScore variant="body1">
                    {player.score.toLocaleString()}
                </PlayerScore>
            </PlayerItem>
        );
    }, [sortedPlayers]);

    if (!players.length) {
        return (
            <PlayerStatusContainer>
                <Box display="flex" justifyContent="center" alignItems="center" p={3}>
                    <CircularProgress size={24} />
                    <Typography variant="body1" ml={2}>
                        Waiting for players...
                    </Typography>
                </Box>
            </PlayerStatusContainer>
        );
    }

    return (
        <PlayerStatusContainer>
            <Box p={2} borderBottom={1} borderColor="divider">
                <Typography variant="h6">
                    Players ({players.length}/{MAX_PLAYERS})
                </Typography>
            </Box>
            <PlayerListContainer>
                <Virtuoso
                    style={{ height: '100%' }}
                    totalCount={sortedPlayers.length}
                    itemContent={renderPlayer}
                    overscan={5}
                    computeItemKey={index => sortedPlayers[index].playerId}
                />
            </PlayerListContainer>
        </PlayerStatusContainer>
    );
});

PlayerStatus.displayName = 'PlayerStatus';

export default PlayerStatus;