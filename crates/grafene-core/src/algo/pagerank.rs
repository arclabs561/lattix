//! `PageRank` centrality algorithm.
//!
//! Computes the importance of nodes based on link structure.
//! Higher scores indicate more "important" nodes.

use crate::KnowledgeGraph;
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
/// # Algorithm
/// Uses the power iteration method with proper handling of dangling nodes
/// (nodes with no outgoing edges). Dangling mass is redistributed uniformly.
#[must_use]
#[allow(clippy::cast_precision_loss)] // node counts won't exceed f64 precision
pub fn pagerank(kg: &KnowledgeGraph, config: PageRankConfig) -> HashMap<String, f64> {
    let graph = kg.as_petgraph();
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    let n_f64 = n as f64;
    let d = config.damping_factor;
    let teleport = (1.0 - d) / n_f64;

    // Initialize scores uniformly
    let mut scores = vec![1.0 / n_f64; n];
    let mut new_scores = vec![0.0; n];

    // Pre-compute out-degrees and identify dangling nodes
    let out_degrees: Vec<usize> = graph
        .node_indices()
        .map(|idx| graph.neighbors(idx).count())
        .collect();

    for _iter in 0..config.max_iterations {
        // Step 1: Compute dangling mass (sum of scores of nodes with no outlinks)
        let dangling_sum: f64 = out_degrees
            .iter()
            .enumerate()
            .filter(|(_, &deg)| deg == 0)
            .map(|(i, _)| scores[i])
            .sum();

        // Dangling contribution spread uniformly
        let dangling_contrib = d * dangling_sum / n_f64;

        // Step 2: Initialize new scores with teleport + dangling
        new_scores.fill(teleport + dangling_contrib);

        // Step 3: Distribute link mass
        for u_idx in graph.node_indices() {
            let u = u_idx.index();
            let deg = out_degrees[u];
            if deg > 0 {
                let share = d * scores[u] / deg as f64;
                for v_idx in graph.neighbors(u_idx) {
                    new_scores[v_idx.index()] += share;
                }
            }
        }

        // Step 4: Check convergence (L1 norm)
        let diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();

        std::mem::swap(&mut scores, &mut new_scores);

        if diff < config.tolerance {
            break;
        }
    }

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
