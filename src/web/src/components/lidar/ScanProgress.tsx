import React, { useCallback, useEffect, useMemo } from 'react';
import { LinearProgress, Typography, Box, useTheme } from '@mui/material';
import styled from '@emotion/styled';
import { usePerformanceMonitor } from '@react-performance-hooks/core';
import { useLidarScanner } from '../../hooks/useLidarScanner';
import { Loading } from '../common/Loading';
import { scanProgress, ANIMATION_DURATIONS, TIMING_FUNCTIONS } from '../../styles/animations';

// GPU-accelerated container with Material Design 3.0 elevation
const ProgressContainer = styled(Box)<{ elevation: number }>`
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 16px;
  transform: translateZ(0);
  will-change: transform;
  background-color: ${({ theme }) => theme.palette.background.paper};
  border-radius: ${({ theme }) => theme.shape.borderRadius}px;
  box-shadow: ${({ theme, elevation }) => theme.shadows[elevation]};
  transition: box-shadow ${ANIMATION_DURATIONS.normal} ${TIMING_FUNCTIONS.easeInOut};
`;

const StyledLinearProgress = styled(LinearProgress)`
  height: 8px;
  border-radius: 4px;
  transform: translateZ(0);
  will-change: transform;
  
  .MuiLinearProgress-bar {
    transition: transform ${ANIMATION_DURATIONS.fast} ${TIMING_FUNCTIONS.emphasized};
  }
`;

const MetricsContainer = styled(Box)`
  display: flex;
  justify-content: space-between;
  align-items: center;
  transform: translateZ(0);
  will-change: transform;
`;

interface ScanProgressProps {
  showDetails?: boolean;
  onComplete?: () => void;
  onError?: (error: Error) => void;
  highContrastMode?: boolean;
}

const calculateProgress = (scanResult: any): number => {
  if (!scanResult) return 0;
  const { pointCloud } = scanResult;
  const totalPoints = pointCloud?.points?.length ?? 0;
  const maxPoints = 1000; // Based on technical specifications
  return Math.min(Math.round((totalPoints / maxPoints) * 100), 100);
};

const ScanProgress: React.FC<ScanProgressProps> = React.memo(({
  showDetails = false,
  onComplete,
  onError,
  highContrastMode = false
}) => {
  const theme = useTheme();
  const { isScanning, scanResult, scanRate, resolution } = useLidarScanner();
  const { measurePerformance } = usePerformanceMonitor();

  // Calculate progress with performance monitoring
  const progress = useMemo(() => {
    return measurePerformance('calculateProgress', () => 
      calculateProgress(scanResult)
    );
  }, [scanResult, measurePerformance]);

  // Handle scan completion
  useEffect(() => {
    if (progress === 100 && onComplete) {
      onComplete();
    }
  }, [progress, onComplete]);

  // Handle errors
  useEffect(() => {
    if (!isScanning && scanResult?.error && onError) {
      onError(new Error(scanResult.error));
    }
  }, [isScanning, scanResult, onError]);

  // Format metrics for display
  const formatMetric = useCallback((value: number, unit: string): string => {
    return `${value.toFixed(2)}${unit}`;
  }, []);

  const elevation = useMemo(() => {
    if (!isScanning) return 1;
    return progress < 100 ? 4 : 2;
  }, [isScanning, progress]);

  return (
    <ProgressContainer elevation={elevation}>
      {isScanning ? (
        <>
          <StyledLinearProgress
            variant="determinate"
            value={progress}
            sx={{
              backgroundColor: theme.palette.lidar.background,
              '& .MuiLinearProgress-bar': {
                backgroundColor: highContrastMode ? 
                  theme.palette.lidar.highlight : 
                  theme.palette.lidar.point
              }
            }}
          />
          
          <MetricsContainer>
            <Typography
              variant="body2"
              color="textSecondary"
              sx={{ fontWeight: 500 }}
            >
              {progress}% Complete
            </Typography>
            
            {showDetails && (
              <Box sx={{ display: 'flex', gap: 2 }}>
                <Typography variant="body2" color="textSecondary">
                  Rate: {formatMetric(scanRate, 'Hz')}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Resolution: {formatMetric(resolution, 'cm')}
                </Typography>
              </Box>
            )}
          </MetricsContainer>
        </>
      ) : (
        <Loading 
          size="small"
          color={theme.palette.lidar.point}
          disableRipple
          thickness={4}
        />
      )}
    </ProgressContainer>
  );
});

ScanProgress.displayName = 'ScanProgress';

export default ScanProgress;
export type { ScanProgressProps };