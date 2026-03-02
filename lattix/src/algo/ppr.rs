//! Personalized PageRank (PPR).
//!
//! Computes node importance relative to a specific seed entity,
//! measuring proximity in the graph's link structure.
//!
//! The core power-iteration algorithm is provided by [`graphops`];
//! this module wraps it with a `KnowledgeGraph`-specific API that maps
//! petgraph node indices back to entity IDs.

use crate::KnowledgeGraph;
use std::collections::HashMap;

/// PPR configuration.
#[derive(Debug, Clone, Copy)]
pub struct PprConfig {
    /// Damping factor (probability of following a link vs teleporting back to seed).
    /// Typically 0.85.
    pub damping: f64,
    /// Maximum iterations before stopping.
    pub max_iterations: usize,
    /// Convergence tolerance (L1 norm of score changes).
    pub tolerance: f64,
}

impl Default for PprConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

/// Compute personalized PageRank from a seed entity.
///
/// Returns scores keyed by entity ID string. Higher scores indicate
/// entities closer/more connected to the seed in the graph's link structure.
///
/// Returns an empty map if the graph is empty or the seed entity is not found.
///
/// # Example
///
/// ```
/// use lattix::{KnowledgeGraph, Triple};
/// use lattix::algo::ppr::{personalized_pagerank, PprConfig};
///
/// let mut kg = KnowledgeGraph::new();
/// kg.add_triple(Triple::new("Alice", "knows", "Bob"));
/// kg.add_triple(Triple::new("Bob", "knows", "Carol"));
///
/// let scores = personalized_pagerank(&kg, "Alice", PprConfig::default());
/// assert!(scores.contains_key("Alice"));
/// assert!(scores.contains_key("Bob"));
/// ```
#[must_use]
pub fn personalized_pagerank(
    kg: &KnowledgeGraph,
    seed: &str,
    config: PprConfig,
) -> HashMap<String, f64> {
    let graph = kg.as_petgraph();
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    // Find the seed node index
    let seed_id = crate::EntityId::from(seed);
    let seed_idx = match kg.get_node_index(&seed_id) {
        Some(idx) => idx.index(),
        None => return HashMap::new(),
    };

    // Build personalization vector: 1.0 for seed, 0.0 elsewhere
    let mut personalization = vec![0.0; n];
    personalization[seed_idx] = 1.0;

    let gp_config = graphops::pagerank::PageRankConfig {
        damping: config.damping,
        max_iterations: config.max_iterations,
        tolerance: config.tolerance,
    };

    let scores = graphops::ppr::personalized_pagerank(graph, gp_config, &personalization);

    // Map indices back to entity IDs
    let mut result = HashMap::with_capacity(n);
    for (idx, score) in scores.into_iter().enumerate() {
        let entity = &graph[petgraph::graph::NodeIndex::new(idx)];
        result.insert(entity.id.0.clone(), score);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn ppr_seed_scores_highest() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));

        let scores = personalized_pagerank(&kg, "A", PprConfig::default());

        let a = *scores.get("A").unwrap();
        let b = *scores.get("B").unwrap();
        let c = *scores.get("C").unwrap();

        // Seed should have the highest score in PPR
        assert!(a > b, "Seed A ({a}) should score higher than B ({b})");
        assert!(a > c, "Seed A ({a}) should score higher than C ({c})");
    }

    #[test]
    fn ppr_missing_seed_returns_empty() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));

        let scores = personalized_pagerank(&kg, "Z", PprConfig::default());
        assert!(scores.is_empty());
    }

    #[test]
    fn ppr_empty_graph() {
        let kg = KnowledgeGraph::new();
        let scores = personalized_pagerank(&kg, "A", PprConfig::default());
        assert!(scores.is_empty());
    }

    #[test]
    fn ppr_scores_sum_to_one() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));
        kg.add_triple(Triple::new("A", "rel", "D"));

        let scores = personalized_pagerank(&kg, "A", PprConfig::default());
        let total: f64 = scores.values().sum();

        assert!(
            (total - 1.0).abs() < 1e-6,
            "Scores should sum to 1.0, got {total}",
        );
    }
}
