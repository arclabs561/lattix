//! `PageRank` centrality algorithm.
//!
//! Computes the importance of nodes based on link structure.
//! Higher scores indicate more "important" nodes.
//!
//! The power iteration runs over unique neighbor nodes, so parallel triples
//! do not act as implicit edge weights.

use crate::{EntityId, KnowledgeGraph};
use std::collections::HashMap;

/// `PageRank` configuration.
#[derive(Debug, Clone, Copy)]
pub struct PageRankConfig {
    /// Damping factor (probability of following a link vs teleporting).
    /// Typically 0.85.
    pub damping_factor: f64,
    /// Maximum iterations before stopping.
    pub max_iterations: usize,
    /// Convergence tolerance (L1 norm of score changes).
    pub tolerance: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

/// Compute `PageRank` for all entities.
///
/// Returns a map of `EntityId` -> Score, where scores sum to 1.0.
///
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn pagerank(kg: &KnowledgeGraph, config: PageRankConfig) -> HashMap<EntityId, f64> {
    let graph = kg.as_petgraph();
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    let adjacency: Vec<Vec<_>> = graph
        .node_indices()
        .map(|idx| {
            crate::algo::unique_neighbors_directed(graph, idx, petgraph::Direction::Outgoing)
                .into_iter()
                .map(|n| n.index())
                .collect()
        })
        .collect();

    let n_f = n as f64;
    let damping = config.damping_factor;
    let teleport = (1.0 - damping) / n_f;
    let mut scores = vec![1.0 / n_f; n];
    let mut next = vec![0.0; n];

    for _ in 0..config.max_iterations {
        next.fill(teleport);

        let dangling: f64 = adjacency
            .iter()
            .enumerate()
            .filter(|(_, neighbors)| neighbors.is_empty())
            .map(|(idx, _)| scores[idx])
            .sum();
        let dangling_share = damping * dangling / n_f;
        for score in &mut next {
            *score += dangling_share;
        }

        for (src, neighbors) in adjacency.iter().enumerate() {
            if neighbors.is_empty() {
                continue;
            }
            let share = damping * scores[src] / neighbors.len() as f64;
            for &dst in neighbors {
                next[dst] += share;
            }
        }

        let diff: f64 = scores
            .iter()
            .zip(next.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();
        std::mem::swap(&mut scores, &mut next);
        if diff < config.tolerance {
            break;
        }
    }

    let mut result = HashMap::with_capacity(n);
    for (idx, score) in scores.into_iter().enumerate() {
        let entity = &graph[petgraph::graph::NodeIndex::new(idx)];
        result.insert(entity.id.clone(), score);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn test_pagerank_cycle() {
        let mut kg = KnowledgeGraph::new();
        // A -> B -> C -> A (cycle)
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));

        let scores = pagerank(&kg, PageRankConfig::default());

        // Symmetric cycle: all scores should be equal
        let a = scores.get("A").unwrap();
        let b = scores.get("B").unwrap();
        let c = scores.get("C").unwrap();

        assert!((a - b).abs() < 1e-4, "A={a} B={b}");
        assert!((b - c).abs() < 1e-4, "B={b} C={c}");
        assert!((a - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_pagerank_star() {
        let mut kg = KnowledgeGraph::new();
        // Hub -> A, Hub -> B, Hub -> C (star topology)
        kg.add_triple(Triple::new("Hub", "rel", "A"));
        kg.add_triple(Triple::new("Hub", "rel", "B"));
        kg.add_triple(Triple::new("Hub", "rel", "C"));

        let scores = pagerank(&kg, PageRankConfig::default());

        // Hub has outlinks but no inlinks from the graph
        // A, B, C are dangling (no outlinks)
        let hub = scores.get("Hub").unwrap();
        let a = scores.get("A").unwrap();

        // Dangling nodes receive mass from Hub + teleport
        // Hub only gets teleport (no inlinks)
        assert!(a > hub, "Leaf A ({a}) should rank higher than Hub ({hub})",);
    }

    #[test]
    fn test_pagerank_sums_to_one() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));
        kg.add_triple(Triple::new("A", "rel", "D"));

        let scores = pagerank(&kg, PageRankConfig::default());
        let total: f64 = scores.values().sum();

        assert!(
            (total - 1.0).abs() < 1e-6,
            "Scores should sum to 1.0, got {total}",
        );
    }
}
