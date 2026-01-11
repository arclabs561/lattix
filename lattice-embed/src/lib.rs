//! Knowledge Graph Embedding inference.
//!
//! This crate provides inference for pre-trained KG embedding models:
//!
//! - TransE
//! - RotatE
//! - ComplEx
//! - DistMult
//!
//! Models are trained in Python (PyKEEN, PyG) and exported to ONNX format
//! for inference in Rust.
//!
//! # Example
//!
//! ```rust,ignore
//! use lattice_embed::TransE;
//!
//! // Load pre-trained model
//! let model = TransE::from_onnx("transE.onnx")?;
//!
//! // Score a triple
//! let score = model.score("Apple", "founded_by", "Steve Jobs")?;
//!
//! // Link prediction: find likely objects for (Apple, founded_by, ?)
//! let candidates = model.predict_object("Apple", "founded_by", 10)?;
//! ```
//!
//! # Training
//!
//! Models should be trained using PyKEEN and exported:
//!
//! ```python
//! from pykeen.pipeline import pipeline
//! from pykeen.models import TransE
//! import torch
//!
//! # Train
//! result = pipeline(model='TransE', dataset='FB15k-237')
//! model = result.model
//!
//! # Export to ONNX
//! torch.onnx.export(model, ..., "transE.onnx")
//! ```

mod error;
mod scoring;

#[cfg(feature = "onnx")]
mod onnx;

pub use error::{Error, Result};
pub use scoring::{LinkPredictionResult, ScoringFunction, TripleScore};

#[cfg(feature = "onnx")]
pub use onnx::{OnnxKGE, TransEOnnx};

/// Trait for KG embedding models.
pub trait KGEmbedding {
    /// Score a triple (h, r, t).
    ///
    /// Higher scores indicate more plausible triples.
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32>;

    /// Predict likely tail entities for (head, relation, ?).
    fn predict_tail(
        &self,
        head: &str,
        relation: &str,
        k: usize,
    ) -> Result<Vec<LinkPredictionResult>>;

    /// Predict likely head entities for (?, relation, tail).
    fn predict_head(
        &self,
        relation: &str,
        tail: &str,
        k: usize,
    ) -> Result<Vec<LinkPredictionResult>>;

    /// Get entity embedding by ID or label.
    fn entity_embedding(&self, entity: &str) -> Result<Vec<f32>>;

    /// Get relation embedding by ID or label.
    fn relation_embedding(&self, relation: &str) -> Result<Vec<f32>>;

    /// Embedding dimension.
    fn embedding_dim(&self) -> usize;

    /// Number of entities.
    fn num_entities(&self) -> usize;

    /// Number of relations.
    fn num_relations(&self) -> usize;
}
