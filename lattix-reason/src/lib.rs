//! # lattix-reason
//!
//! Multi-hop logical reasoning engine for Knowledge Graphs.
//!
//! This crate provides a high-level reasoning engine that can execute complex
//! logical queries using various backends (Neural KGE, Symbolic Logic, etc.).

pub mod query;
pub mod algo;

pub use query::LogicalQuery;
pub use algo::SymbolicReasoner;

use lattix_kge::Prediction;
use thiserror::Error;

/// Errors that can occur during reasoning.
#[derive(Error, Debug)]
pub enum ReasonError {
    #[error("KG error: {0}")]
    Kg(String),
    #[error("Logic error: {0}")]
    Logic(String),
    #[error("Timeout: {0}")]
    Timeout(String),
}

pub type Result<T> = std::result::Result<T, ReasonError>;

/// Unified interface for reasoning engines.
pub trait Reasoner {
    /// Score an entity as an answer to a logical query.
    fn score(&self, query: &LogicalQuery, entity: &str) -> Result<f32>;

    /// Predict the top-k entities that satisfy a logical query.
    fn predict(&self, query: &LogicalQuery, k: usize) -> Result<Vec<Prediction>>;
}

    /// Inductive reasoning engine that works on unseen graphs.
    ///
    /// Based on ULTRA (Ren et al. 2024).
    fn reason_with_context(
        &self,
        query: &LogicalQuery,
        context: &lattix_core::KnowledgeGraph,
    ) -> Result<Vec<Prediction>>;
}
