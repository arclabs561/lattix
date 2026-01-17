//! Degree centrality: the simplest centrality measure.
//!
//! # Intuition
//!
//! Degree centrality simply counts connections. In a social network,
//! it measures "popularity" - how many friends someone has.
//!
//! # Variants
//!
//! For directed graphs, three variants exist:
//!
//! | Variant | Measures | Interpretation |
//! |---------|----------|----------------|
//! | In-degree | Incoming edges | Prestige, being cited |
//! | Out-degree | Outgoing edges | Activity, citing others |
//! | Total | Both directions | Overall connectivity |
//!
//! # Normalization
//!
//! Raw degree depends on graph size. Normalized degree:
//!
//! ```text
//! C_D(v) = deg(v) / (n - 1)
//! ```
//!
//! Where n is the number of nodes. This gives values in [0, 1].
//!
//! # Limitations
//!
//! - Ignores network structure beyond immediate neighbors
//! - A node with 10 low-degree neighbors ranks same as one with 10 hubs
//! - For structural importance, use eigenvector or betweenness centrality

use crate::KnowledgeGraph;
use std::collections::HashMap;

/// Degree centrality result for a node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DegreeCentrality {
    /// Number of incoming edges.
    pub in_degree: usize,
    /// Number of outgoing edges.
    pub out_degree: usize,
    /// Normalized in-degree (0 to 1).
    pub in_normalized: f64,
    /// Normalized out-degree (0 to 1).
    pub out_normalized: f64,
}

impl DegreeCentrality {
    /// Total degree (in + out).
    #[must_use]
    pub fn total(&self) -> usize {
        self.in_degree + self.out_degree
    }

    /// Normalized total degree.
    #[must_use]
    pub fn total_normalized(&self) -> f64 {
        self.in_normalized + self.out_normalized
    }
}

/// Compute degree centrality for all nodes.
///
/// Returns in-degree, out-degree, and normalized values.
///
/// # Example
///
/// ```
/// use lattix_core::{KnowledgeGraph, Triple};
/// use lattix_core::algo::centrality::degree_centrality;
///
/// let mut kg = KnowledgeGraph::new();
/// kg.add_triple(Triple::new("A", "rel", "B"));
/// kg.add_triple(Triple::new("A", "rel", "C"));
/// kg.add_triple(Triple::new("B", "rel", "C"));
///
/// let degrees = degree_centrality(&kg);
/// let a = degrees.get("A").unwrap();
/// assert_eq!(a.out_degree, 2);  // A -> B, A -> C
/// assert_eq!(a.in_degree, 0);   // nothing points to A
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn degree_centrality(kg: &KnowledgeGraph) -> HashMap<String, DegreeCentrality> {
    let graph = kg.as_petgraph();
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    let norm_factor = if n > 1 { (n - 1) as f64 } else { 1.0 };
    let mut result = HashMap::with_capacity(n);

    for idx in graph.node_indices() {
        let in_deg = graph
            .neighbors_directed(idx, petgraph::Direction::Incoming)
            .count();
        let out_deg = graph
            .neighbors_directed(idx, petgraph::Direction::Outgoing)
            .count();

        let entity = &graph[idx];
        result.insert(
            entity.id.0.clone(),
            DegreeCentrality {
                in_degree: in_deg,
                out_degree: out_deg,
                in_normalized: in_deg as f64 / norm_factor,
                out_normalized: out_deg as f64 / norm_factor,
            },
        );
    }

    result
}

/// Compute in-degree centrality only.
///
/// Returns normalized in-degree for each node.
#[must_use]
pub fn in_degree_centrality(kg: &KnowledgeGraph) -> HashMap<String, f64> {
    degree_centrality(kg)
        .into_iter()
        .map(|(k, v)| (k, v.in_normalized))
        .collect()
}

/// Compute out-degree centrality only.
///
/// Returns normalized out-degree for each node.
#[must_use]
pub fn out_degree_centrality(kg: &KnowledgeGraph) -> HashMap<String, f64> {
    degree_centrality(kg)
        .into_iter()
        .map(|(k, v)| (k, v.out_normalized))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn test_degree_star() {
        let mut kg = KnowledgeGraph::new();
        // Star: Hub -> A, Hub -> B, Hub -> C
        kg.add_triple(Triple::new("Hub", "rel", "A"));
        kg.add_triple(Triple::new("Hub", "rel", "B"));
        kg.add_triple(Triple::new("Hub", "rel", "C"));

        let degrees = degree_centrality(&kg);

        let hub = degrees.get("Hub").unwrap();
        assert_eq!(hub.out_degree, 3);
        assert_eq!(hub.in_degree, 0);

        let a = degrees.get("A").unwrap();
        assert_eq!(a.out_degree, 0);
        assert_eq!(a.in_degree, 1);
    }

    #[test]
    fn test_degree_normalization() {
        let mut kg = KnowledgeGraph::new();
        // Complete triangle: everyone connects to everyone
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "A"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "B"));
        kg.add_triple(Triple::new("A", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));

        let degrees = degree_centrality(&kg);

        // Each node has 2 in, 2 out. n-1 = 2, so normalized = 1.0
        for (_, deg) in degrees {
            assert!((deg.in_normalized - 1.0).abs() < 1e-6);
            assert!((deg.out_normalized - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_empty_graph() {
        let kg = KnowledgeGraph::new();
        let degrees = degree_centrality(&kg);
        assert!(degrees.is_empty());
    }
}
