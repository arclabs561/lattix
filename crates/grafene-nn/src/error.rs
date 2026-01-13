//! Error types for propago.

use thiserror::Error;

/// Propago error type.
#[derive(Debug, Error)]
pub enum Error {
    /// Candle tensor error.
    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),

    /// Dimension mismatch.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Invalid configuration.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Training error.
    #[error("training error: {0}")]
    Training(String),
}

/// Result type alias.
pub type Result<T> = std::result::Result<T, Error>;
