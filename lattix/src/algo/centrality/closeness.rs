//! Closeness centrality: measuring proximity to all other nodes.
//!
//! # Intuition
//!
//! Closeness measures how quickly information can spread from a node.
//! High closeness = short average distance to all others.
//!
//! In a social network: someone who can reach anyone in few hops.
//! In a transit network: a well-connected hub station.
//!
//! # Definition
//!
//! Classic closeness (Bavelas 1950):
//!
//! ```text
//! C_C(v) = (n - 1) / Σ_{u≠v} d(v, u)
//! ```
//!
//! Where d(v, u) is the shortest path distance from v to u.
//!
//! # Handling Disconnected Graphs
//!
//! If some nodes are unreachable, classic closeness breaks (infinite distance).
//! Two common fixes:
//!
//! | Variant | Formula | Behavior |
//! |---------|---------|----------|
//! | **Harmonic** | Σ_{u≠v} 1/d(v,u) | Ignore unreachable (d=∞ → 0) |
//! | **Wasserman-Faust** | (reachable - 1) / Σ d(v,u) | Only count reachable |
//!
//! This implementation uses **harmonic centrality** for robustness.
//!
//! # Normalization
//!
//! Harmonic centrality is normalized by dividing by (n-1):
//!
//! ```text
//! C_H_norm(v) = C_H(v) / (n - 1)
//! ```
//!
//! # References
//!
//! - Bavelas (1950). "Communication patterns in task-oriented groups"
//! - Rochat (2009). "Closeness centrality extended to unconnected graphs"

use crate::KnowledgeGraph;
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, VecDeque};

/// Configuration for closeness centrality.
#[derive(Debug, Clone, Copy)]
pub struct ClosenessConfig {
    /// Normalize scores to [0, 1] range.
    pub normalized: bool,
    /// Treat graph as undirected.
    pub undirected: bool,
    /// Use harmonic mean (recommended for disconnected graphs).
    pub harmonic: bool,
}

impl Default for ClosenessConfig {
    fn default() -> Self {
        Self {
            normalized: true,
            undirected: false,
            harmonic: true, // robust to disconnected components
        }
    }
}

/// Compute closeness centrality for all nodes.
///
/// Uses harmonic centrality by default, which handles disconnected graphs.
///
/// # Complexity
///
/// - Time: O(VE) (BFS from each node)
/// - Space: O(V)
///
/// # Example
///
/// ```
/// use lattix::{KnowledgeGraph, Triple};
/// use lattix::algo::centrality::{closeness_centrality, ClosenessConfig};
///
/// let mut kg = KnowledgeGraph::new();
/// // Star: Hub -> A, Hub -> B, Hub -> C
/// kg.add_triple(Triple::new("Hub", "rel", "A"));
/// kg.add_triple(Triple::new("Hub", "rel", "B"));
/// kg.add_triple(Triple::new("Hub", "rel", "C"));
///
/// let scores = closeness_centrality(&kg, ClosenessConfig::default());
/// // Hub reaches everyone in 1 hop
/// // A, B, C can't reach anyone (directed graph)
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn closeness_centrality(kg: &KnowledgeGraph, config: ClosenessConfig) -> HashMap<String, f64> {
    let graph = kg.as_petgraph();
    let n = graph.node_count();
    if n < 2 {
        return graph
            .node_indices()
            .map(|idx| (graph[idx].id.0.clone(), 0.0))
            .collect();
    }

    let mut result = HashMap::with_capacity(n);

    for source in graph.node_indices() {
        let distances = bfs_distances(graph, source, config.undirected);

        let closeness = if config.harmonic {
            // Harmonic: Σ 1/d(v,u) for all reachable u
            let sum: f64 = distances
                .iter()
                .enumerate()
                .filter(|(i, &d)| *i != source.index() && d > 0)
                .map(|(_, &d)| 1.0 / d as f64)
                .sum();
            sum
        } else {
            // Classic: (n-1) / Σ d(v,u)
            let reachable: Vec<_> = distances
                .iter()
                .enumerate()
                .filter(|(i, &d)| *i != source.index() && d > 0)
                .collect();

            if reachable.is_empty() {
                0.0
            } else {
                let total_dist: i32 = reachable.iter().map(|(_, &d)| d).sum();
                (reachable.len() as f64) / (total_dist as f64)
            }
        };

        let normalized_closeness = if config.normalized {
            closeness / (n - 1) as f64
        } else {
            closeness
        };

        let entity = &graph[source];
        result.insert(entity.id.0.clone(), normalized_closeness);
    }

    result
}

/// BFS to find distances from source.
///
/// Returns distance array. -1 means unreachable, 0 means self.
fn bfs_distances(
    graph: &petgraph::Graph<crate::Entity, crate::Relation>,
    source: NodeIndex,
    undirected: bool,
) -> Vec<i32> {
    let n = graph.node_count();
    let mut dist = vec![-1_i32; n];
    dist[source.index()] = 0;

    let mut queue = VecDeque::new();
    queue.push_back(source);

    while let Some(v) = queue.pop_front() {
        let v_dist = dist[v.index()];

        let neighbors: Vec<NodeIndex> = if undirected {
            graph.neighbors_undirected(v).collect()
        } else {
            graph
                .neighbors_directed(v, petgraph::Direction::Outgoing)
                .collect()
        };

        for w in neighbors {
            if dist[w.index()] < 0 {
                dist[w.index()] = v_dist + 1;
                queue.push_back(w);
            }
        }
    }

    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn test_closeness_star_directed() {
        let mut kg = KnowledgeGraph::new();
        // Star: Hub -> A, Hub -> B, Hub -> C
        kg.add_triple(Triple::new("Hub", "rel", "A"));
        kg.add_triple(Triple::new("Hub", "rel", "B"));
        kg.add_triple(Triple::new("Hub", "rel", "C"));

        let config = ClosenessConfig::default();
        let scores = closeness_centrality(&kg, config);

        // Hub can reach A, B, C in 1 hop each
        let hub = *scores.get("Hub").unwrap();
        let a = *scores.get("A").unwrap();

        // Hub has positive closeness, leaves have zero (can't reach anyone)
        assert!(hub > 0.0, "Hub should have positive closeness: {hub}");
        assert_eq!(a, 0.0, "A can't reach anyone: {a}");
    }

    #[test]
    fn test_closeness_line() {
        let mut kg = KnowledgeGraph::new();
        // Line: A -> B -> C
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));

        let config = ClosenessConfig {
            normalized: false,
            undirected: true, // treat as undirected
            harmonic: true,
        };
        let scores = closeness_centrality(&kg, config);

        let a = *scores.get("A").unwrap();
        let b = *scores.get("B").unwrap();
        let c = *scores.get("C").unwrap();

        // B is central (dist 1 to A and C), A and C are endpoints
        // Harmonic: B = 1/1 + 1/1 = 2, A = 1/1 + 1/2 = 1.5, C = 1/2 + 1/1 = 1.5
        assert!(b > a, "B should be more central than A: B={b}, A={a}");
        assert!(
            (a - c).abs() < 1e-6,
            "A and C should be symmetric: A={a}, C={c}"
        );
    }

    #[test]
    fn test_closeness_disconnected() {
        let mut kg = KnowledgeGraph::new();
        // Two disconnected edges: A -> B, C -> D
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("C", "rel", "D"));

        let config = ClosenessConfig::default();
        let scores = closeness_centrality(&kg, config);

        // Harmonic centrality handles this gracefully
        // A can only reach B (score > 0 but low)
        let a = *scores.get("A").unwrap();
        assert!(a > 0.0, "A should have some closeness: {a}");
    }

    #[test]
    fn test_closeness_normalized() {
        let mut kg = KnowledgeGraph::new();
        // Complete triangle
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "A"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "B"));
        kg.add_triple(Triple::new("A", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));

        let config = ClosenessConfig {
            normalized: true,
            undirected: false,
            harmonic: true,
        };
        let scores = closeness_centrality(&kg, config);

        // All nodes reach each other in 1 hop
        // Normalized harmonic = (1/1 + 1/1) / 2 = 1.0
        for (name, score) in scores {
            assert!(
                (score - 1.0).abs() < 1e-6,
                "Complete graph should have max closeness: {name}={score}"
            );
        }
    }
}
