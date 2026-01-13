use thiserror::Error;

/// Errors that can occur in grafene-embed.
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
}

/// Result type alias for grafene-embed.
pub type Result<T> = std::result::Result<T, Error>;
