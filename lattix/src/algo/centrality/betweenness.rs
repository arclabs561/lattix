//! Betweenness centrality: measuring brokerage and bridging.
//!
//! # Intuition
//!
//! Betweenness measures how often a node lies on shortest paths between
//! other nodes. High betweenness = broker, bridge, gatekeeper.
//!
//! In a social network: the person who connects different friend groups.
//! In a citation network: papers that bridge different research areas.
//!
//! # Definition
//!
//! ```text
//! C_B(v) = Σ_{s≠v≠t} σ_st(v) / σ_st
//! ```
//!
//! Where:
//! - σ_st = number of shortest paths from s to t
//! - σ_st(v) = number of those paths passing through v
//!
//! # Brandes' Algorithm (2001)
//!
//! Naive computation is O(V³). Brandes showed O(VE) is possible by:
//!
//! 1. Run BFS from each source s
//! 2. Track σ_sv (shortest path counts) during forward pass
//! 3. Accumulate dependencies δ_s(v) during backward pass
//!
//! Key insight: dependencies can be computed recursively:
//!
//! ```text
//! δ_s(v) = Σ_{w: v∈P_s(w)} (σ_sv/σ_sw) × (1 + δ_s(w))
//! ```
//!
//! Where P_s(w) is the set of predecessors of w on shortest paths from s.
//!
//! # Normalization
//!
//! For directed graphs:
//! ```text
//! C_B_norm(v) = C_B(v) / [(n-1)(n-2)]
//! ```
//!
//! For undirected graphs (each path counted twice):
//! ```text
//! C_B_norm(v) = 2 × C_B(v) / [(n-1)(n-2)]
//! ```
//!
//! # References
//!
//! - Brandes (2001). "A faster algorithm for betweenness centrality"
//! - Freeman (1977). "A set of measures of centrality based on betweenness"

use crate::KnowledgeGraph;
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, VecDeque};

/// Configuration for betweenness centrality.
#[derive(Debug, Clone, Copy)]
pub struct BetweennessConfig {
    /// Normalize scores to [0, 1] range.
    pub normalized: bool,
    /// Treat graph as undirected (follow edges both ways).
    pub undirected: bool,
}

impl Default for BetweennessConfig {
    fn default() -> Self {
        Self {
            normalized: true,
            undirected: false,
        }
    }
}

/// Compute betweenness centrality using Brandes' algorithm.
///
/// # Complexity
///
/// - Time: O(VE) for unweighted graphs
/// - Space: O(V + E)
///
/// # Example
///
/// ```
/// use lattix::{KnowledgeGraph, Triple};
/// use lattix::algo::centrality::{betweenness_centrality, BetweennessConfig};
///
/// let mut kg = KnowledgeGraph::new();
/// // Path: A -> B -> C
/// kg.add_triple(Triple::new("A", "rel", "B"));
/// kg.add_triple(Triple::new("B", "rel", "C"));
///
/// let config = BetweennessConfig { normalized: false, undirected: false };
/// let scores = betweenness_centrality(&kg, config);
///
/// // B is on the only path from A to C
/// let b = scores.get("B").unwrap();
/// assert!(*b > 0.0);
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn betweenness_centrality(
    kg: &KnowledgeGraph,
    config: BetweennessConfig,
) -> HashMap<String, f64> {
    let graph = kg.as_petgraph();
    let n = graph.node_count();
    if n < 2 {
        return graph
            .node_indices()
            .map(|idx| (graph[idx].id.0.clone(), 0.0))
            .collect();
    }

    // Initialize betweenness scores
    let mut betweenness = vec![0.0_f64; n];

    // Run BFS from each source
    for s in graph.node_indices() {
        let (sigma, predecessors, order) = bfs_shortest_paths(graph, s, config.undirected);

        // Backward pass: accumulate dependencies
        let mut delta = vec![0.0_f64; n];

        // Process nodes in reverse BFS order (farthest first)
        for &w in order.iter().rev() {
            let w_idx = w.index();
            for &v in &predecessors[w_idx] {
                let v_idx = v.index();
                // δ_s(v) += (σ_sv / σ_sw) × (1 + δ_s(w))
                let coeff = sigma[v_idx] / sigma[w_idx];
                delta[v_idx] += coeff * (1.0 + delta[w_idx]);
            }
            // Accumulate (skip source)
            if w != s {
                betweenness[w_idx] += delta[w_idx];
            }
        }
    }

    // For undirected graphs, each path is counted twice
    if config.undirected {
        for b in &mut betweenness {
            *b /= 2.0;
        }
    }

    // Normalize if requested
    if config.normalized && n > 2 {
        let norm = ((n - 1) * (n - 2)) as f64;
        for b in &mut betweenness {
            *b /= norm;
        }
    }

    // Map back to entity IDs
    graph
        .node_indices()
        .map(|idx| (graph[idx].id.0.clone(), betweenness[idx.index()]))
        .collect()
}

/// BFS to find shortest paths from source.
///
/// Returns:
/// - sigma: σ_sv = number of shortest paths from s to v
/// - predecessors: P_s(v) = predecessors on shortest paths
/// - order: nodes in BFS order (for backward pass)
#[allow(clippy::cast_precision_loss)]
fn bfs_shortest_paths(
    graph: &petgraph::Graph<crate::Entity, crate::Relation>,
    source: NodeIndex,
    undirected: bool,
) -> (Vec<f64>, Vec<Vec<NodeIndex>>, Vec<NodeIndex>) {
    let n = graph.node_count();
    let mut sigma = vec![0.0_f64; n]; // number of shortest paths
    let mut dist = vec![-1_i32; n]; // distance from source (-1 = unvisited)
    let mut predecessors: Vec<Vec<NodeIndex>> = vec![Vec::new(); n];
    let mut order = Vec::with_capacity(n);

    sigma[source.index()] = 1.0;
    dist[source.index()] = 0;

    let mut queue = VecDeque::new();
    queue.push_back(source);

    while let Some(v) = queue.pop_front() {
        order.push(v);
        let v_idx = v.index();
        let v_dist = dist[v_idx];

        // Get neighbors based on direction setting
        let neighbors: Vec<NodeIndex> = if undirected {
            graph.neighbors_undirected(v).collect()
        } else {
            graph
                .neighbors_directed(v, petgraph::Direction::Outgoing)
                .collect()
        };

        for w in neighbors {
            let w_idx = w.index();

            // First time seeing w?
            if dist[w_idx] < 0 {
                dist[w_idx] = v_dist + 1;
                queue.push_back(w);
            }

            // Is this a shortest path to w?
            if dist[w_idx] == v_dist + 1 {
                sigma[w_idx] += sigma[v_idx];
                predecessors[w_idx].push(v);
            }
        }
    }

    (sigma, predecessors, order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn test_betweenness_line() {
        let mut kg = KnowledgeGraph::new();
        // Line graph: A -> B -> C -> D
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "D"));

        let config = BetweennessConfig {
            normalized: false,
            undirected: false,
        };
        let scores = betweenness_centrality(&kg, config);

        // B is on paths: A->B->C, A->B->C->D (2 paths)
        // C is on paths: A->B->C->D, B->C->D (2 paths)
        // A and D are endpoints, score 0
        assert_eq!(*scores.get("A").unwrap(), 0.0);
        assert_eq!(*scores.get("D").unwrap(), 0.0);
        assert!(*scores.get("B").unwrap() > 0.0);
        assert!(*scores.get("C").unwrap() > 0.0);
    }

    #[test]
    fn test_betweenness_star() {
        let mut kg = KnowledgeGraph::new();
        // Star: Hub -> A, Hub -> B, Hub -> C
        kg.add_triple(Triple::new("Hub", "rel", "A"));
        kg.add_triple(Triple::new("Hub", "rel", "B"));
        kg.add_triple(Triple::new("Hub", "rel", "C"));

        let config = BetweennessConfig::default();
        let scores = betweenness_centrality(&kg, config);

        // In a directed star, Hub has no betweenness (it's the source, not bridge)
        // No paths go through Hub since all edges originate from Hub
        // A, B, C are leaf nodes with no betweenness either
        for score in scores.values() {
            assert_eq!(
                *score, 0.0,
                "Star nodes have no betweenness in directed graph"
            );
        }
    }

    #[test]
    fn test_betweenness_bridge() {
        let mut kg = KnowledgeGraph::new();
        // Two cliques connected by bridge
        // Clique 1: A <-> B
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "A"));
        // Bridge: B -> C
        kg.add_triple(Triple::new("B", "rel", "C"));
        // Clique 2: C <-> D
        kg.add_triple(Triple::new("C", "rel", "D"));
        kg.add_triple(Triple::new("D", "rel", "C"));

        let config = BetweennessConfig {
            normalized: false,
            undirected: false,
        };
        let scores = betweenness_centrality(&kg, config);

        // B and C should have high betweenness (bridge nodes)
        let b = *scores.get("B").unwrap();
        let c = *scores.get("C").unwrap();
        let a = *scores.get("A").unwrap();
        let d = *scores.get("D").unwrap();

        assert!(
            b > a && c > d,
            "Bridge nodes should have higher betweenness: B={b}, C={c}, A={a}, D={d}"
        );
    }

    #[test]
    fn test_betweenness_undirected() {
        let mut kg = KnowledgeGraph::new();
        // Path: A -- B -- C (undirected)
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));

        let config = BetweennessConfig {
            normalized: false,
            undirected: true,
        };
        let scores = betweenness_centrality(&kg, config);

        // B is on path from A to C and from C to A
        let b = *scores.get("B").unwrap();
        assert!(b > 0.0, "B should be on shortest paths: {b}");
    }
}
