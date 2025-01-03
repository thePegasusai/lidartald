import React, { useCallback, useState, useRef, useEffect } from 'react';
import { TextField } from '@mui/material'; // v5.13.0
import styled from '@emotion/styled'; // v11.11.0
import { validateInput } from '../../utils/validation';
import { COMPONENT_SIZES } from '../../styles/components';
import { fadeIn } from '../../styles/animations';

// Enhanced TextField with high-contrast support and outdoor visibility
const StyledTextField = styled(TextField)`
  width: 100%;
  min-height: ${COMPONENT_SIZES.medium};
  margin: ${({ theme }) => theme.spacing(1)} 0;
  
  // Base styles
  & .MuiInputBase-root {
    background-color: ${({ theme }) => theme.palette.background.paper};
    transition: background-color 0.3s ${({ theme }) => theme.transitions.easing.easeInOut};
  }

  // High contrast mode styles
  &.high-contrast {
    & .MuiInputBase-root {
      background-color: ${({ theme }) => theme.palette.mode === 'dark' ? '#000000' : '#ffffff'};
      border: 2px solid ${({ theme }) => theme.palette.mode === 'dark' ? '#ffffff' : '#000000'};
    }
    
    & .MuiInputLabel-root {
      color: ${({ theme }) => theme.palette.mode === 'dark' ? '#ffffff' : '#000000'};
    }

    & .MuiOutlinedInput-notchedOutline {
      border-width: 2px;
      border-color: ${({ theme }) => theme.palette.mode === 'dark' ? '#ffffff' : '#000000'};
    }
  }

  // Touch target optimization
  & .MuiInputBase-input {
    min-height: ${COMPONENT_SIZES.touchTarget};
    padding: ${({ theme }) => theme.spacing(1, 1.5)};
  }

  // Error state with enhanced visibility
  &.Mui-error {
    & .MuiOutlinedInput-notchedOutline {
      border-width: ${({ theme }) => theme.palette.mode === 'dark' ? '2px' : '1px'};
    }
  }

  // Animation
  animation: ${fadeIn} 0.3s ease-out;
`;

export interface InputProps {
  id: string;
  name: string;
  label: string;
  value: string;
  type?: 'text' | 'number' | 'email' | 'password';
  placeholder?: string;
  size?: 'small' | 'medium' | 'large';
  required?: boolean;
  disabled?: boolean;
  error?: boolean;
  helperText?: string;
  highContrast?: boolean;
  fleetSync?: boolean;
  onChange: (value: string) => void;
  onBlur?: (event: React.FocusEvent<HTMLInputElement>) => void;
  validator?: (value: string) => Promise<boolean>;
  onError?: (error: string) => void;
  onValidationStart?: () => void;
  onValidationComplete?: () => void;
}

const Input: React.FC<InputProps> = React.memo(({
  id,
  name,
  label,
  value,
  type = 'text',
  placeholder,
  size = 'medium',
  required = false,
  disabled = false,
  error = false,
  helperText,
  highContrast = false,
  fleetSync = false,
  onChange,
  onBlur,
  validator,
  onError,
  onValidationStart,
  onValidationComplete
}) => {
  const [isValidating, setIsValidating] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const validationTimeout = useRef<NodeJS.Timeout>();

  // Debounced validation handler
  const handleValidation = useCallback(async (inputValue: string) => {
    if (!validator) return true;

    try {
      setIsValidating(true);
      onValidationStart?.();

      const isValid = await validator(inputValue);
      
      if (!isValid) {
        const error = `Invalid ${name.toLowerCase()} format`;
        setLocalError(error);
        onError?.(error);
        return false;
      }

      setLocalError(null);
      return true;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Validation error';
      setLocalError(errorMessage);
      onError?.(errorMessage);
      return false;
    } finally {
      setIsValidating(false);
      onValidationComplete?.();
    }
  }, [validator, name, onError, onValidationStart, onValidationComplete]);

  // Handle input change with debounced validation
  const handleChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = event.target.value;
    
    // Update input value immediately
    onChange(newValue);

    // Clear previous validation timeout
    if (validationTimeout.current) {
      clearTimeout(validationTimeout.current);
    }

    // Set new validation timeout (300ms debounce)
    validationTimeout.current = setTimeout(() => {
      handleValidation(newValue);
    }, 300);
  }, [onChange, handleValidation]);

  // Handle input blur with final validation
  const handleBlur = useCallback((event: React.FocusEvent<HTMLInputElement>) => {
    handleValidation(event.target.value);
    onBlur?.(event);
  }, [handleValidation, onBlur]);

  // Cleanup validation timeout on unmount
  useEffect(() => {
    return () => {
      if (validationTimeout.current) {
        clearTimeout(validationTimeout.current);
      }
    };
  }, []);

  // Input size mapping
  const inputSize = size === 'small' ? 'small' : 'medium';

  return (
    <StyledTextField
      id={id}
      name={name}
      label={label}
      value={value}
      type={type}
      placeholder={placeholder}
      size={inputSize}
      required={required}
      disabled={disabled}
      error={error || !!localError}
      helperText={localError || helperText}
      className={highContrast ? 'high-contrast' : ''}
      onChange={handleChange}
      onBlur={handleBlur}
      inputProps={{
        'aria-label': label,
        'aria-required': required,
        'aria-invalid': error || !!localError,
        'data-fleet-sync': fleetSync,
        'data-testid': `input-${id}`,
      }}
      FormHelperTextProps={{
        'aria-live': 'polite',
      }}
      variant="outlined"
      fullWidth
    />
  );
});

Input.displayName = 'Input';

export default Input;