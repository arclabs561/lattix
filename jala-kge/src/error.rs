use thiserror::Error;

/// Errors that can occur in lattix-kge.
#[derive(Error, Debug)]
pub enum Error {
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    /// ONNX Runtime error.
    #[cfg(feature = "onnx")]
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    /// Entity ID not found in mapping.
    #[error("Entity not found: {0}")]
    EntityNotFound(String),
    /// Relation ID not found in mapping.
    #[error("Relation not found: {0}")]
    RelationNotFound(String),
    /// Operation not supported by the model or configuration.
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    /// Generic not found error.
    #[error("Not found: {0}")]
    NotFound(String),
    /// Validation error.
    #[error("Validation error: {0}")]
    Validation(String),

    /// Box embedding error (from subsume crate).
    #[cfg(feature = "boxe")]
    #[error("Box embedding error: {0}")]
    BoxEmbedding(String),
}

/// Result type alias for lattix-kge.
pub type Result<T> = std::result::Result<T, Error>;
