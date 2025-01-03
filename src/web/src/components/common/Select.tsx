import React, { useMemo } from 'react';
import {
  Select as MuiSelect,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent,
  useTheme,
  styled,
  alpha
} from '@mui/material'; // ^5.13.0
import { THEME_COLORS } from '../../styles/theme';

// Interface for Select component props
interface SelectProps {
  name: string;
  label: string;
  value: string | number;
  onChange: (event: SelectChangeEvent<string | number>) => void;
  options: Array<{ value: string | number; label: string }>;
  disabled?: boolean;
  error?: boolean;
  helperText?: string;
  fullWidth?: boolean;
  size?: 'small' | 'medium';
  outdoorMode?: boolean;
  contrastLevel?: 'normal' | 'high' | 'ultra';
}

// Styled components with outdoor visibility optimizations
const StyledFormControl = styled(FormControl, {
  shouldForwardProp: (prop) => 
    !['outdoorMode', 'contrastLevel'].includes(prop as string),
})<{
  outdoorMode?: boolean;
  contrastLevel?: string;
}>(({ theme, outdoorMode, contrastLevel }) => ({
  '& .MuiInputLabel-root': {
    color: outdoorMode
      ? contrastLevel === 'ultra'
        ? theme.palette.common.black
        : theme.palette.text.primary
      : theme.palette.text.primary,
    fontWeight: outdoorMode ? 500 : 400,
    fontSize: outdoorMode ? '1.1rem' : '1rem',
    textShadow: outdoorMode 
      ? '0 0 2px rgba(255, 255, 255, 0.5)'
      : 'none',
  },
  '& .MuiOutlinedInput-root': {
    backgroundColor: outdoorMode
      ? alpha(theme.palette.background.paper, 0.95)
      : 'transparent',
    '&:hover': {
      '& .MuiOutlinedInput-notchedOutline': {
        borderColor: outdoorMode
          ? theme.palette.primary.main
          : theme.palette.action.hover,
        borderWidth: outdoorMode ? 2 : 1,
      },
    },
    '&.Mui-focused': {
      '& .MuiOutlinedInput-notchedOutline': {
        borderColor: theme.palette.primary.main,
        borderWidth: outdoorMode ? 3 : 2,
      },
    },
  },
  '& .MuiOutlinedInput-notchedOutline': {
    borderColor: outdoorMode
      ? alpha(theme.palette.primary.main, 0.5)
      : theme.palette.action.disabled,
    borderWidth: outdoorMode ? 2 : 1,
  },
}));

const StyledSelect = styled(MuiSelect, {
  shouldForwardProp: (prop) => 
    !['outdoorMode', 'contrastLevel'].includes(prop as string),
})<{
  outdoorMode?: boolean;
  contrastLevel?: string;
}>(({ theme, outdoorMode, contrastLevel }) => ({
  '& .MuiSelect-select': {
    fontWeight: outdoorMode ? 500 : 400,
    color: outdoorMode && contrastLevel === 'ultra'
      ? theme.palette.common.black
      : theme.palette.text.primary,
  },
  '& .MuiSelect-icon': {
    color: outdoorMode
      ? theme.palette.primary.main
      : theme.palette.action.active,
  },
}));

const StyledMenuItem = styled(MenuItem, {
  shouldForwardProp: (prop) => 
    !['outdoorMode', 'contrastLevel'].includes(prop as string),
})<{
  outdoorMode?: boolean;
  contrastLevel?: string;
}>(({ theme, outdoorMode, contrastLevel }) => ({
  fontWeight: outdoorMode ? 500 : 400,
  color: outdoorMode && contrastLevel === 'ultra'
    ? theme.palette.common.black
    : theme.palette.text.primary,
  '&.Mui-selected': {
    backgroundColor: outdoorMode
      ? alpha(theme.palette.primary.main, 0.2)
      : alpha(theme.palette.primary.main, 0.1),
    '&:hover': {
      backgroundColor: outdoorMode
        ? alpha(theme.palette.primary.main, 0.3)
        : alpha(theme.palette.primary.main, 0.2),
    },
  },
  '&:hover': {
    backgroundColor: outdoorMode
      ? alpha(theme.palette.action.hover, 0.2)
      : alpha(theme.palette.action.hover, 0.1),
  },
}));

// Select component with outdoor visibility and fleet sync support
export const Select: React.FC<SelectProps> = ({
  name,
  label,
  value,
  onChange,
  options,
  disabled = false,
  error = false,
  helperText,
  fullWidth = false,
  size = 'medium',
  outdoorMode = false,
  contrastLevel = 'normal',
}) => {
  const theme = useTheme();

  // Memoize contrast-related styles for performance
  const contrastStyles = useMemo(() => ({
    labelId: `${name}-label`,
    menuProps: {
      PaperProps: {
        sx: {
          backgroundColor: outdoorMode
            ? alpha(theme.palette.background.paper, 0.95)
            : theme.palette.background.paper,
          boxShadow: outdoorMode
            ? '0 4px 20px rgba(0, 0, 0, 0.5)'
            : theme.shadows[8],
        },
      },
    },
  }), [name, outdoorMode, theme]);

  return (
    <StyledFormControl
      fullWidth={fullWidth}
      error={error}
      disabled={disabled}
      size={size}
      outdoorMode={outdoorMode}
      contrastLevel={contrastLevel}
    >
      <InputLabel id={contrastStyles.labelId}>
        {label}
      </InputLabel>
      <StyledSelect
        labelId={contrastStyles.labelId}
        id={name}
        value={value}
        label={label}
        onChange={onChange}
        outdoorMode={outdoorMode}
        contrastLevel={contrastLevel}
        MenuProps={contrastStyles.menuProps}
      >
        {options.map((option) => (
          <StyledMenuItem
            key={option.value}
            value={option.value}
            outdoorMode={outdoorMode}
            contrastLevel={contrastLevel}
          >
            {option.label}
          </StyledMenuItem>
        ))}
      </StyledSelect>
    </StyledFormControl>
  );
};

export default Select;