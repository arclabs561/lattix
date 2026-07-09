//! Algorithms for graph analysis and embedding generation.
//!
//! This module contains implementations of graph algorithms:
//!
//! - **Centrality**: Measure node importance ([`centrality`])
//! - **Random walks**: Node2Vec-style biased walks ([`random_walk`])
//! - **Components**: Find connected components ([`components`])
//! - **Sampling**: Mini-batch sampling for GNNs ([`sampling`])
//! - **PPR**: Personalized PageRank from a seed entity ([`ppr`])
//! - **Label propagation**: Community detection ([`label_propagation`])
//!
//! # Centrality Overview
//!
//! | Algorithm | Question | Complexity |
//! |-----------|----------|------------|
//! These complexities include the cost of deduplicating parallel triples into
//! unique neighbor nodes, where `d_max` is the maximum stored degree.
//!
//! | Algorithm | Question | Complexity |
//! |-----------|----------|------------|
//! | Degree | How many connections? | O(V + E log d_max) |
//! | Betweenness | Bridge between communities? | O(VE log d_max) |
//! | Closeness | How close to everyone? | O(VE log d_max) |
//! | Eigenvector | Connected to important nodes? | O(E log d_max × iter) |
//! | Katz | Reachable via damped paths? | O(E log d_max × iter) |
//! | PageRank | Random walk equilibrium? | O(E log d_max + E × iter) |
//! | HITS | Hub or authority? | O(E log d_max × iter) |

/// Centrality algorithms for measuring node importance.
pub mod centrality;

/// Random walk algorithm (Node2Vec style).
pub mod random_walk;

/// PageRank centrality algorithm (also available via [`centrality`]).
pub mod pagerank;

/// Connected components algorithm.
pub mod components;

/// Graph sampling algorithms (e.g. for GNNs).
pub mod sampling;

/// Personalized PageRank from a seed entity.
pub mod ppr;

/// Label propagation community detection.
pub mod label_propagation;

use std::collections::HashMap;

use petgraph::graph::NodeIndex;

use crate::EntityId;

/// Return the top-n scored entities, sorted descending by score.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use lattix::EntityId;
/// use lattix::algo::top_n;
///
/// let scores: HashMap<EntityId, f64> = [
///     (EntityId::from("A"), 0.5),
///     (EntityId::from("B"), 0.3),
///     (EntityId::from("C"), 0.2),
/// ].into_iter().collect();
///
/// let top = top_n(&scores, 2);
/// assert_eq!(top.len(), 2);
/// assert_eq!(top[0].0.as_str(), "A");
/// assert_eq!(top[1].0.as_str(), "B");
/// ```
#[must_use]
pub fn top_n(scores: &HashMap<EntityId, f64>, n: usize) -> Vec<(EntityId, f64)> {
    let mut entries: Vec<(EntityId, f64)> = scores.iter().map(|(k, &v)| (k.clone(), v)).collect();
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    entries.truncate(n);
    entries
}

pub(crate) fn unique_neighbors_directed(
    graph: &petgraph::Graph<crate::Entity, crate::Relation>,
    node: NodeIndex,
    direction: petgraph::Direction,
) -> Vec<NodeIndex> {
    let mut neighbors: Vec<_> = graph.neighbors_directed(node, direction).collect();
    dedup_nodes(&mut neighbors);
    neighbors
}

pub(crate) fn unique_neighbors_undirected(
    graph: &petgraph::Graph<crate::Entity, crate::Relation>,
    node: NodeIndex,
) -> Vec<NodeIndex> {
    let mut neighbors: Vec<_> = graph.neighbors_undirected(node).collect();
    dedup_nodes(&mut neighbors);
    neighbors
}

fn dedup_nodes(nodes: &mut Vec<NodeIndex>) {
    nodes.sort_unstable_by_key(|idx| idx.index());
    nodes.dedup();
}
