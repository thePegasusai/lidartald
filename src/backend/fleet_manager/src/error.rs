use std::error::Error as StdError;
use thiserror::Error;
use tokio::error::Error as TokioError;
use webrtc::error::Error as WebRtcError;

/// Custom error type for Fleet Manager operations with error codes in range 3000-3999
#[derive(Debug, Error)]
pub enum FleetError {
    /// Device discovery and advertisement errors (3000-3099)
    #[error("Discovery error (code: 3{:03}): {message}")]
    DiscoveryError {
        code: u16,
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },

    /// P2P connection establishment errors (3100-3199)
    #[error("Connection error (code: 3{:03}): {message}")]
    ConnectionError {
        code: u16,
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },

    /// Fleet session management errors (3200-3299)
    #[error("Session error (code: 3{:03}): {message}")]
    SessionError {
        code: u16,
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },

    /// State synchronization errors (3300-3399)
    #[error("Sync error (code: 3{:03}): {message}")]
    SyncError {
        code: u16,
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },

    /// Mesh network topology errors (3400-3499)
    #[error("Mesh error (code: 3{:03}): {message}")]
    MeshError {
        code: u16,
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },

    /// Invalid state and operation errors (3500-3599)
    #[error("Invalid state (code: 3{:03}): {message}")]
    InvalidState {
        code: u16,
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },
}

impl FleetError {
    /// Returns the numeric error code for the error variant
    pub fn error_code(&self) -> u16 {
        match self {
            FleetError::DiscoveryError { code, .. } => *code,
            FleetError::ConnectionError { code, .. } => *code,
            FleetError::SessionError { code, .. } => *code,
            FleetError::SyncError { code, .. } => *code,
            FleetError::MeshError { code, .. } => *code,
            FleetError::InvalidState { code, .. } => *code,
        }
    }

    /// Creates an error instance from a numeric code and message
    pub fn from_code(code: u16, message: String) -> Result<Self, &'static str> {
        match code {
            3000..=3099 => Ok(FleetError::DiscoveryError {
                code,
                message,
                source: None,
            }),
            3100..=3199 => Ok(FleetError::ConnectionError {
                code,
                message,
                source: None,
            }),
            3200..=3299 => Ok(FleetError::SessionError {
                code,
                message,
                source: None,
            }),
            3300..=3399 => Ok(FleetError::SyncError {
                code,
                message,
                source: None,
            }),
            3400..=3499 => Ok(FleetError::MeshError {
                code,
                message,
                source: None,
            }),
            3500..=3599 => Ok(FleetError::InvalidState {
                code,
                message,
                source: None,
            }),
            _ => Err("Invalid error code range"),
        }
    }
}

// Implement From traits for common error types
impl From<TokioError> for FleetError {
    fn from(err: TokioError) -> Self {
        FleetError::ConnectionError {
            code: 3100,
            message: err.to_string(),
            source: Some(Box::new(err)),
        }
    }
}

impl From<WebRtcError> for FleetError {
    fn from(err: WebRtcError) -> Self {
        FleetError::ConnectionError {
            code: 3101,
            message: err.to_string(),
            source: Some(Box::new(err)),
        }
    }
}

/// Result type alias for fleet operations using FleetError
pub type FleetResult<T> = Result<T, FleetError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_ranges() {
        let error = FleetError::from_code(3000, "test".to_string()).unwrap();
        assert_eq!(error.error_code(), 3000);

        let error = FleetError::from_code(3599, "test".to_string()).unwrap();
        assert_eq!(error.error_code(), 3599);

        assert!(FleetError::from_code(2999, "test".to_string()).is_err());
        assert!(FleetError::from_code(3600, "test".to_string()).is_err());
    }

    #[test]
    fn test_error_messages() {
        let error = FleetError::DiscoveryError {
            code: 3001,
            message: "Device not found".to_string(),
            source: None,
        };
        assert!(error.to_string().contains("Device not found"));
        assert!(error.to_string().contains("3001"));
    }
}