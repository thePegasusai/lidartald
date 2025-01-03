import React, { useEffect, useMemo, useCallback } from 'react';
import styled from '@emotion/styled';
import { Alert as MuiAlert, AlertTitle } from '@mui/material'; // @version ^5.13.0
import {
  ErrorOutline as ErrorIcon,
  WarningAmber as WarningIcon,
  InfoOutlined as InfoIcon,
  CheckCircleOutline as CheckCircleIcon
} from '@mui/icons-material'; // @version ^5.13.0
import { fadeIn } from '../../styles/animations';

interface AlertProps {
  severity: 'error' | 'warning' | 'info' | 'success';
  title: string;
  message: string;
  autoHideDuration?: number;
  onClose?: () => void;
  className?: string;
  highContrast?: boolean;
  disableAnimation?: boolean;
}

const StyledAlert = styled(MuiAlert)<{ 'data-high-contrast'?: boolean }>`
  position: fixed;
  bottom: 24px;
  right: 24px;
  max-width: 400px;
  min-width: 300px;
  z-index: 1400;
  transform: translateZ(0);
  will-change: transform, opacity;
  animation: ${fadeIn} 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: var(--md-sys-elevation-3);

  @media (prefers-reduced-motion: reduce) {
    animation: none;
  }

  &[data-high-contrast='true'] {
    border: 2px solid currentColor;
    font-weight: 500;
    
    .MuiAlert-icon {
      opacity: 1;
    }
    
    .MuiAlert-message {
      color: inherit;
    }
  }

  .MuiAlert-icon {
    align-self: flex-start;
    margin-top: 6px;
  }

  .MuiAlert-message {
    padding: 8px 0;
  }

  .MuiAlertTitle-root {
    margin-top: 0;
    font-weight: 500;
    line-height: 1.5;
  }
`;

const Alert = React.memo<AlertProps>(({
  severity,
  title,
  message,
  autoHideDuration = 6000,
  onClose,
  className,
  highContrast = false,
  disableAnimation = false
}) => {
  useEffect(() => {
    if (autoHideDuration && onClose) {
      const timer = setTimeout(() => {
        onClose();
      }, autoHideDuration);

      return () => {
        clearTimeout(timer);
      };
    }
  }, [autoHideDuration, onClose]);

  const handleClose = useCallback((event?: React.SyntheticEvent) => {
    if (event) {
      event.preventDefault();
    }
    onClose?.();
  }, [onClose]);

  const icon = useMemo(() => {
    const iconProps = { fontSize: 'small' as const };
    switch (severity) {
      case 'error':
        return <ErrorIcon {...iconProps} />;
      case 'warning':
        return <WarningIcon {...iconProps} />;
      case 'info':
        return <InfoIcon {...iconProps} />;
      case 'success':
        return <CheckCircleIcon {...iconProps} />;
    }
  }, [severity]);

  const alertProps = {
    severity,
    icon,
    onClose: onClose ? handleClose : undefined,
    className,
    'data-high-contrast': highContrast,
    'aria-live': severity === 'error' ? 'assertive' : 'polite',
    'aria-atomic': true,
    role: severity === 'error' ? 'alert' : 'status',
    style: disableAnimation ? { animation: 'none' } : undefined,
  };

  return (
    <StyledAlert {...alertProps}>
      <AlertTitle>{title}</AlertTitle>
      {message}
    </StyledAlert>
  );
});

Alert.displayName = 'Alert';

export default Alert;