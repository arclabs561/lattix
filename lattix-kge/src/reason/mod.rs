//! Multi-hop logical reasoning on Knowledge Graphs.
//!
//! This module provides tools for answering complex queries involving multiple
//! relations and logical operators (AND, OR, NOT).

pub mod query;

pub use query::LogicalQuery;

use crate::error::Result;
use crate::model::Prediction;

/// Trait for KGE models that support multi-hop logical reasoning.
///
/// Models like Query2box and BetaE implement this trait to answer
/// complex queries beyond simple triples.
pub trait ReasoningModel {
    /// Score an entity as an answer to a logical query.
    ///
    /// Returns a plausibility score (higher = more likely).
    fn score_query(&self, query: &LogicalQuery, entity: &str) -> Result<f32>;

    /// Predict the top-k entities that satisfy a logical query.
    fn predict_query(&self, query: &LogicalQuery, k: usize) -> Result<Vec<Prediction>>;
}
