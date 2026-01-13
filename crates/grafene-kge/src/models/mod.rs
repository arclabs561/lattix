//! KGE model implementations.
//!
//! Each model implements the [`KGEModel`](crate::KGEModel) trait, providing a unified
//! interface regardless of backend.
//!
//! # Design Philosophy
//!
//! Following anno's pattern: abstract at the **model level**, not the tensor level.
//! Each model handles its own backend internally. This keeps code simple and
//! allows feature-gated backends without trait complexity.
//!
//! # Available Models
//!
//! | Model | Description | Backend |
//! |-------|-------------|---------|
//! | [`TransE`] | Relations as translations | ndarray |
//! | [`BoxE`] | Relations as boxes | ndarray |
//! | [`HypE`] | N-ary hyperedge embeddings (Fatemi 2019) | ndarray |
//!
//! # Feature-Gated Models
//!
//! | Model | Description | Feature |
//! |-------|-------------|---------|
//! | `MuRP` | Hyperbolic translations (Balazevic 2019) | `hyperbolic` |
//! | `RotH` | Hyperbolic rotations (Chami 2020) | `hyperbolic` |
//! | `TransEBurn` | TransE on Burn | `burn` |
//!
//! # Example
//!
//! ```rust,ignore
//! use grafene_kge::{KGEModel, TransE, BoxE, Fact, TrainingConfig};
//!
//! // All models share the same interface
//! let mut transe = TransE::new(128);
//! let mut boxe = BoxE::new(128);
//!
//! let triples = vec![Fact::from_strs("A", "r", "B")];
//! let config = TrainingConfig::default();
//!
//! transe.train(&triples, &config)?;
//! boxe.train(&triples, &config)?;
//!
//! // Same scoring interface
//! let s1 = transe.score("A", "r", "B")?;
//! let s2 = boxe.score("A", "r", "B")?;
//! ```

mod boxe;
mod transe;

pub use boxe::BoxE;
pub use transe::TransE;

// Hyperedge models (n-ary relations)
mod hype;
pub use hype::{HypE, HyperFact};

// Hyperbolic models (Poincare ball)
#[cfg(feature = "hyperbolic")]
mod murp;
#[cfg(feature = "hyperbolic")]
pub use murp::MuRP;

#[cfg(feature = "hyperbolic")]
mod roth;
#[cfg(feature = "hyperbolic")]
pub use roth::RotH;

// ONNX inference (pre-trained models)
// NOTE: Disabled until ndarray version is aligned with ort crate requirements
// #[cfg(feature = "onnx")]
// mod onnx_wrapper;
// #[cfg(feature = "onnx")]
// pub use onnx_wrapper::KGEOnnx;

// GPU-accelerated models via Burn
#[cfg(feature = "burn")]
mod burn_backend;
#[cfg(feature = "burn")]
pub use burn_backend::TransEBurn;
