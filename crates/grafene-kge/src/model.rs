//! Unified KGE model trait following anno's backend-agnostic pattern.
//!
//! This module defines the `KGEModel` trait that all KGE implementations must satisfy,
//! regardless of backend (ndarray, burn, onnx). The pattern is inspired by anno's `Model`
//! trait: abstract at the model level, not the tensor level.
//!
//! # Design Philosophy
//!
//! Instead of:
//! ```ignore
//! // DON'T: Generic tensor abstraction (complex, leaky)
//! trait Embedding { fn add(&self, other: &Self) -> Self; }
//! struct TransE<E: Embedding> { ... }
//! ```
//!
//! We do:
//! ```ignore
//! // DO: Model-level abstraction (simple, clean)
//! trait KGEModel { fn score(...); fn train(...); }
//! struct TransE { /* ndarray internals */ }
//! struct TransEBurn<B: Backend> { /* burn internals */ }
//! ```
//!
//! Each backend implementation handles its own tensor type internally.
//! This is cleaner because:
//! - Training loops are backend-specific anyway (autograd differs)
//! - No generic bounds pollution
//! - Easy feature gating (`#[cfg(feature = "burn")]`)
//! - Add backends incrementally without trait changes
//!
//! # Example
//!
//! ```rust,ignore
//! use grafene_kge::{KGEModel, TransE, TrainingConfig};
//!
//! // Create and train model
//! let mut model = TransE::new(128);  // 128-dim embeddings
//! model.train(&triples, &TrainingConfig::default())?;
//!
//! // Score and predict
//! let score = model.score("Einstein", "won", "NobelPrize")?;
//! let predictions = model.predict_tail("Einstein", "won", 10)?;
//! ```
//!
//! # Backend Implementations
//!
//! | Model | Backend | Feature | Status |
//! |-------|---------|---------|--------|
//! | `TransE` | ndarray | always | Working |
//! | `BoxE` | ndarray | always | Working |
//! | `MuRP` | ndarray + hyperball | `hyperbolic` | Planned |
//! | `TransEBurn` | burn | `burn` | Planned |

use crate::error::Result;
use crate::training::TrainingConfig;
use grafene_core::Triple;
use std::collections::HashMap;

/// A triple for KGE training/inference.
///
/// Generic over string type to support both owned and borrowed data.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fact<S = String> {
    /// Head entity (subject in RDF terminology).
    pub head: S,
    /// Relation (predicate in RDF terminology).
    pub relation: S,
    /// Tail entity (object in RDF terminology).
    pub tail: S,
}

impl<S> Fact<S> {
    /// Create a new fact from head, relation, tail.
    pub fn new(head: S, relation: S, tail: S) -> Self {
        Self { head, relation, tail }
    }
}

impl Fact<String> {
    /// Create from string slices (cloning into owned strings).
    pub fn from_strs(head: &str, relation: &str, tail: &str) -> Self {
        Self {
            head: head.to_string(),
            relation: relation.to_string(),
            tail: tail.to_string(),
        }
    }
}

impl<'a> Fact<&'a str> {
    /// Convert to owned Fact.
    pub fn to_owned(&self) -> Fact<String> {
        Fact {
            head: self.head.to_string(),
            relation: self.relation.to_string(),
            tail: self.tail.to_string(),
        }
    }
}

// ============================================================================
// Conversions between Fact and grafene_core::Triple
// ============================================================================

impl From<Triple> for Fact<String> {
    /// Convert from `grafene_core::Triple` (discarding metadata).
    fn from(t: Triple) -> Self {
        Self {
            head: t.subject.as_str().to_string(),
            relation: t.predicate.as_str().to_string(),
            tail: t.object.as_str().to_string(),
        }
    }
}

impl From<&Triple> for Fact<String> {
    fn from(t: &Triple) -> Self {
        Self {
            head: t.subject.as_str().to_string(),
            relation: t.predicate.as_str().to_string(),
            tail: t.object.as_str().to_string(),
        }
    }
}

impl From<Fact<String>> for Triple {
    /// Convert to `grafene_core::Triple` (with default confidence).
    fn from(f: Fact<String>) -> Self {
        Triple::new(f.head, f.relation, f.tail)
    }
}

/// Link prediction result with entity and score.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Predicted entity.
    pub entity: String,
    /// Plausibility score (higher = more plausible).
    pub score: f32,
}

/// Training metrics from one epoch.
#[derive(Debug, Clone, Default)]
pub struct EpochMetrics {
    /// Average loss for this epoch.
    pub loss: f32,
    /// MRR on validation set (if provided).
    pub val_mrr: Option<f32>,
    /// Hits@10 on validation set (if provided).
    pub val_hits_at_10: Option<f32>,
}

/// Callback for training progress.
pub type ProgressCallback = Box<dyn Fn(usize, &EpochMetrics) + Send + Sync>;

/// Unified trait for Knowledge Graph Embedding models.
///
/// All KGE models (TransE, BoxE, MuRP, etc.) implement this trait regardless
/// of their internal backend (ndarray, burn, onnx).
///
/// # Design
///
/// This follows anno's pattern of abstracting at the model level:
/// - Each implementation uses its own tensor type internally
/// - Training loops are backend-specific (different autograd)
/// - Inference returns standard Rust types (f32, Vec, HashMap)
///
/// # Required Methods
///
/// Models must implement `score`, `train`, and metadata methods.
/// Prediction methods have default implementations based on `score`.
pub trait KGEModel: Send + Sync {
    // =========================================================================
    // Core Operations
    // =========================================================================

    /// Score a triple (head, relation, tail).
    ///
    /// Returns a plausibility score where higher = more likely true.
    /// The exact scale depends on the model (TransE uses negative distance).
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32>;

    /// Train the model on a set of triples.
    ///
    /// # Arguments
    /// * `triples` - Training data as (head, relation, tail) facts
    /// * `config` - Training hyperparameters
    ///
    /// # Returns
    /// Average loss over the final epoch.
    fn train(&mut self, triples: &[Fact<String>], config: &TrainingConfig) -> Result<f32>;

    /// Train with progress callback.
    ///
    /// Default implementation ignores the callback and calls `train`.
    fn train_with_callback(
        &mut self,
        triples: &[Fact<String>],
        config: &TrainingConfig,
        _callback: ProgressCallback,
    ) -> Result<f32> {
        self.train(triples, config)
    }

    // =========================================================================
    // Embeddings Access
    // =========================================================================

    /// Get entity embedding as a flat f32 vector.
    ///
    /// Returns `None` if entity is unknown.
    fn entity_embedding(&self, entity: &str) -> Option<Vec<f32>>;

    /// Get relation embedding as a flat f32 vector.
    ///
    /// Returns `None` if relation is unknown.
    fn relation_embedding(&self, relation: &str) -> Option<Vec<f32>>;

    /// Get all entity embeddings.
    fn entity_embeddings(&self) -> &HashMap<String, Vec<f32>>;

    /// Get all relation embeddings.
    fn relation_embeddings(&self) -> &HashMap<String, Vec<f32>>;

    // =========================================================================
    // Metadata
    // =========================================================================

    /// Embedding dimension.
    fn embedding_dim(&self) -> usize;

    /// Number of entities.
    fn num_entities(&self) -> usize;

    /// Number of relations.
    fn num_relations(&self) -> usize;

    /// Model name (e.g., "TransE", "BoxE", "MuRP").
    fn name(&self) -> &'static str;

    /// Whether the model is trained and ready for inference.
    fn is_trained(&self) -> bool;

    // =========================================================================
    // Link Prediction (with default implementations)
    // =========================================================================

    /// Predict likely tail entities for (head, relation, ?).
    ///
    /// Returns top-k predictions sorted by score (descending).
    fn predict_tail(&self, head: &str, relation: &str, k: usize) -> Result<Vec<Prediction>> {
        let entities: Vec<String> = self.entity_embeddings().keys().cloned().collect();
        let mut scores: Vec<(String, f32)> = entities
            .into_iter()
            .filter_map(|e| {
                self.score(head, relation, &e)
                    .ok()
                    .map(|s| (e, s))
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores
            .into_iter()
            .map(|(entity, score)| Prediction { entity, score })
            .collect())
    }

    /// Predict likely head entities for (?, relation, tail).
    ///
    /// Returns top-k predictions sorted by score (descending).
    fn predict_head(&self, relation: &str, tail: &str, k: usize) -> Result<Vec<Prediction>> {
        let entities: Vec<String> = self.entity_embeddings().keys().cloned().collect();
        let mut scores: Vec<(String, f32)> = entities
            .into_iter()
            .filter_map(|e| {
                self.score(&e, relation, tail)
                    .ok()
                    .map(|s| (e, s))
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores
            .into_iter()
            .map(|(entity, score)| Prediction { entity, score })
            .collect())
    }
}

/// Marker trait for models that support GPU acceleration.
pub trait GpuCapable: KGEModel {
    /// Whether GPU is currently active.
    fn is_gpu_active(&self) -> bool;

    /// Device name (e.g., "cuda:0", "metal", "cpu").
    fn device(&self) -> &str;
}

/// Marker trait for models that support batch operations.
pub trait BatchCapable: KGEModel {
    /// Score multiple triples in a batch (more efficient than individual calls).
    fn score_batch(&self, triples: &[Fact<&str>]) -> Result<Vec<f32>>;

    /// Optimal batch size for this backend.
    fn optimal_batch_size(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fact_creation() {
        let f = Fact::new("Einstein", "won", "NobelPrize");
        assert_eq!(f.head, "Einstein");
        assert_eq!(f.relation, "won");
        assert_eq!(f.tail, "NobelPrize");
    }

    #[test]
    fn test_fact_from_strs() {
        let f = Fact::from_strs("Paris", "capitalOf", "France");
        assert_eq!(f.head, "Paris");
        assert_eq!(f.tail, "France");
    }

    #[test]
    fn test_borrowed_fact_to_owned() {
        let f: Fact<&str> = Fact::new("a", "b", "c");
        let owned = f.to_owned();
        assert_eq!(owned.head, "a");
    }

    #[test]
    fn test_fact_from_triple() {
        let triple = Triple::new("Einstein", "won", "NobelPrize");
        let fact: Fact<String> = Fact::from(&triple);
        assert_eq!(fact.head, "Einstein");
        assert_eq!(fact.relation, "won");
        assert_eq!(fact.tail, "NobelPrize");
    }

    #[test]
    fn test_triple_from_fact() {
        let fact = Fact::from_strs("Paris", "capitalOf", "France");
        let triple: Triple = fact.into();
        assert_eq!(triple.subject.as_str(), "Paris");
        assert_eq!(triple.predicate.as_str(), "capitalOf");
        assert_eq!(triple.object.as_str(), "France");
    }
}
