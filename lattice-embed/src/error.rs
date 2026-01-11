//! Error types for lattice-embed.

use thiserror::Error;

/// Error type for embedding operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Entity not found in vocabulary.
    #[error("Entity not found: {0}")]
    EntityNotFound(String),

    /// Relation not found in vocabulary.
    #[error("Relation not found: {0}")]
    RelationNotFound(String),

    /// Model loading error.
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// Inference error.
    #[error("Inference error: {0}")]
    Inference(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// ONNX runtime error.
    #[cfg(feature = "onnx")]
    #[error("ONNX error: {0}")]
    Onnx(String),
}

/// Result type for embedding operations.
pub type Result<T> = std::result::Result<T, Error>;
