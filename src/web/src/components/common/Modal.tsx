import React, { useEffect, useCallback, useRef, memo } from 'react';
import styled from '@emotion/styled';
import { Portal, Paper } from '@mui/material';
import { fadeIn, fadeOut } from '../../styles/animations';

// Animation duration constant matching design system
const ANIMATION_DURATION = 300;

// Z-index constants for proper layering
const Z_INDEX = {
  backdrop: 1000,
  modal: 1001,
} as const;

// Interface for Modal component props
interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  closeOnBackdropClick?: boolean;
  closeOnEscape?: boolean;
  className?: string;
  'aria-labelledby'?: string;
  'aria-describedby'?: string;
}

// Styled backdrop with GPU-accelerated animations
const Backdrop = styled.div<{ isOpen: boolean }>`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: ${({ theme }) => 
    theme.palette.mode === 'dark' 
      ? 'rgba(0, 0, 0, 0.7)' 
      : 'rgba(0, 0, 0, 0.5)'
  };
  backdrop-filter: blur(20px);
  z-index: ${Z_INDEX.backdrop};
  animation: ${({ isOpen }) => isOpen ? fadeIn : fadeOut} ${ANIMATION_DURATION}ms cubic-bezier(0.4, 0, 0.2, 1);
  will-change: opacity;
  transform: translateZ(0);
`;

// Styled modal container with Material Design elevation
const ModalContainer = styled(Paper)<{ isOpen: boolean }>`
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%) translateZ(0);
  z-index: ${Z_INDEX.modal};
  max-width: 90vw;
  max-height: 90vh;
  overflow: auto;
  animation: ${({ isOpen }) => isOpen ? fadeIn : fadeOut} ${ANIMATION_DURATION}ms cubic-bezier(0.4, 0, 0.2, 1);
  will-change: opacity, transform;
  elevation: ${({ theme }) => theme.shadows[24]};
  border-radius: ${({ theme }) => theme.shape.borderRadius}px;
  background-color: ${({ theme }) => 
    theme.palette.mode === 'dark' 
      ? theme.palette.grey[900] 
      : theme.palette.background.paper
  };
`;

const Modal: React.FC<ModalProps> = memo(({
  isOpen,
  onClose,
  children,
  closeOnBackdropClick = true,
  closeOnEscape = true,
  className,
  'aria-labelledby': ariaLabelledBy,
  'aria-describedby': ariaDescribedBy,
}) => {
  const modalRef = useRef<HTMLDivElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);

  // Handle backdrop clicks
  const handleBackdropClick = useCallback((event: React.MouseEvent) => {
    if (
      closeOnBackdropClick &&
      event.target === event.currentTarget
    ) {
      event.stopPropagation();
      onClose();
    }
  }, [closeOnBackdropClick, onClose]);

  // Handle escape key press
  const handleEscapeKey = useCallback((event: KeyboardEvent) => {
    if (closeOnEscape && event.key === 'Escape') {
      event.preventDefault();
      onClose();
    }
  }, [closeOnEscape, onClose]);

  // Manage focus trap and event listeners
  useEffect(() => {
    if (isOpen) {
      // Store previous focus
      previousFocus.current = document.activeElement as HTMLElement;
      
      // Add event listeners
      document.addEventListener('keydown', handleEscapeKey);
      
      // Focus first focusable element
      if (modalRef.current) {
        const focusableElements = modalRef.current.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (focusableElements.length) {
          (focusableElements[0] as HTMLElement).focus();
        }
      }

      // Prevent body scroll
      document.body.style.overflow = 'hidden';
    }

    return () => {
      if (isOpen) {
        // Remove event listeners
        document.removeEventListener('keydown', handleEscapeKey);
        
        // Restore previous focus
        if (previousFocus.current) {
          previousFocus.current.focus();
        }

        // Restore body scroll
        document.body.style.overflow = '';
      }
    };
  }, [isOpen, handleEscapeKey]);

  if (!isOpen) {
    return null;
  }

  return (
    <Portal>
      <Backdrop
        isOpen={isOpen}
        onClick={handleBackdropClick}
        data-testid="modal-backdrop"
      />
      <ModalContainer
        ref={modalRef}
        isOpen={isOpen}
        className={className}
        role="dialog"
        aria-modal="true"
        aria-labelledby={ariaLabelledBy}
        aria-describedby={ariaDescribedBy}
        data-testid="modal-container"
      >
        {children}
      </ModalContainer>
    </Portal>
  );
});

Modal.displayName = 'Modal';

export default Modal;