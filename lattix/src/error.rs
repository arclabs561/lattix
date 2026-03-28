//! Error types for lattix.

use thiserror::Error;

/// Error type for lattix operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Triple parsing error.
    #[error("Failed to parse triple: {0}")]
    ParseTriple(String),

    /// N-Triples format error.
    #[error("Invalid N-Triples format: {0}")]
    InvalidNTriples(String),

    /// Entity not found.
    #[error("Entity not found: {0}")]
    EntityNotFound(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid data format.
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// A required file is missing.
    #[error("Missing file: {0}")]
    MissingFile(String),

    /// JSON error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Serialization/deserialization error (bincode, etc.).
    #[error("serialization error: {0}")]
    Serialization(Box<dyn std::error::Error + Send + Sync>),
}

/// Result type for lattix operations.
pub type Result<T> = std::result::Result<T, Error>;
