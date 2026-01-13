//! Error types for nexus-core.

use thiserror::Error;

/// Error type for lattice operations.
#[derive(Error, Debug)]
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

    /// JSON error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type for lattice operations.
pub type Result<T> = std::result::Result<T, Error>;
