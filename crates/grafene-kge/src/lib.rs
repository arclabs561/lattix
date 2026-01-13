//! Knowledge Graph Embedding inference.
//!
//! Knowledge graphs store facts as (head, relation, tail) triples:
//! `(Einstein, won, NobelPrize)`, `(Paris, capitalOf, France)`.
//! KGE models learn low-dimensional vector representations where
//! **geometric operations on embeddings predict missing links**.
//!
//! ## The Core Intuition
//!
//! Each model encodes a different geometric hypothesis about how
//! relations transform entities:
//!
//! | Model | Hypothesis | Geometric Operation |
//! |-------|------------|---------------------|
//! | TransE | Relations are translations | h + r ≈ t |
//! | RotatE | Relations are rotations | h ∘ r ≈ t |
//! | DistMult | Relations are scalings | <h, r, t> |
//! | ComplEx | Asymmetric relations | Re(<h, r, conj(t)>) |
//!
//! ## TransE: Relations as Translations
//!
//! The simplest and most influential model ([Bordes et al. 2013](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)).
//!
//! **Idea**: If (h, r, t) is true, then h + r ≈ t in embedding space.
//!
//! ```text
//!   h ----r----> t
//!   |            |
//!   v            v
//!  [0.2, 0.5] + [0.3, 0.1] ≈ [0.5, 0.6]
//! ```
//!
//! **Scoring**: -||h + r - t||₂ (lower distance = more plausible)
//!
//! **Limitation**: Can't model symmetric relations (husband/wife) or
//! composition (grandparent = parent ∘ parent).
//!
//! ## RotatE: Relations as Rotations
//!
//! [Sun et al. 2019](https://arxiv.org/abs/1902.10197) models relations
//! as rotations in complex space.
//!
//! **Idea**: Each relation rotates entity embeddings by an angle θ.
//!
//! ```text
//! h ∘ r = t    where r = e^(iθ)
//! ```
//!
//! In complex multiplication: (a + bi)(cos θ + i sin θ) rotates by θ.
//!
//! **Why rotation?** It can model:
//! - **Symmetric relations**: 180° rotation (r ∘ r = identity)
//! - **Inverse relations**: -θ rotation
//! - **Composition**: Angles add (parent ∘ parent = grandparent)
//!
//! ## DistMult: Bilinear Diagonal
//!
//! [Yang et al. 2015](https://arxiv.org/abs/1412.6575) uses element-wise
//! product (Hadamard product).
//!
//! **Scoring**: Σᵢ hᵢ × rᵢ × tᵢ
//!
//! **Limitation**: Symmetric by construction—predicts (h, r, t) = (t, r, h).
//! Can't distinguish "parent_of" from "child_of".
//!
//! ## ComplEx: Complex Embeddings
//!
//! [Trouillon et al. 2016](https://arxiv.org/abs/1606.06357) extends
//! DistMult to complex space to handle asymmetry.
//!
//! **Scoring**: Re(<h, r, conj(t)>) where conj is complex conjugate.
//!
//! The conjugate breaks symmetry: (h, r, t) ≠ (t, r, h) in general.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use grafene_kge::{ScoringFunction, KGEmbedding};
//!
//! // Score using raw embeddings
//! let h = vec![1.0, 0.0, 0.5];
//! let r = vec![0.0, 1.0, 0.0];
//! let t = vec![1.0, 1.0, 0.5];
//!
//! let score = ScoringFunction::TransE.score(&h, &r, &t);
//! // score ≈ 0 means h + r ≈ t (plausible triple)
//! ```
//!
//! ## Training
//!
//! Models are trained in Python (PyKEEN, PyG) and exported to ONNX:
//!
//! ```python,ignore
//! from pykeen.pipeline import pipeline
//!
//! result = pipeline(model='TransE', dataset='FB15k-237')
//! # Export to ONNX for Rust inference
//! ```
//!
//! ## References
//!
//! - Bordes et al. (2013). "Translating Embeddings for Modeling
//!   Multi-relational Data." NIPS.
//! - Sun et al. (2019). "RotatE: Knowledge Graph Embedding by
//!   Relational Rotation in Complex Space." ICLR.
//! - Trouillon et al. (2016). "Complex Embeddings for Simple Link
//!   Prediction." ICML.

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
