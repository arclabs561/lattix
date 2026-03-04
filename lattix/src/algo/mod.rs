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
//! | Degree | How many connections? | O(V) |
//! | Betweenness | Bridge between communities? | O(VE) |
//! | Closeness | How close to everyone? | O(VE) |
//! | Eigenvector | Connected to important nodes? | O(E × iter) |
//! | Katz | Reachable via damped paths? | O(E × iter) |
//! | PageRank | Random walk equilibrium? | O(E × iter) |
//! | HITS | Hub or authority? | O(E × iter) |

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

/// Return the top-n scored entities, sorted descending by score.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use lattix::algo::top_n;
///
/// let scores: HashMap<String, f64> = [
///     ("A".to_string(), 0.5),
///     ("B".to_string(), 0.3),
///     ("C".to_string(), 0.2),
/// ].into_iter().collect();
///
/// let top = top_n(&scores, 2);
/// assert_eq!(top.len(), 2);
/// assert_eq!(top[0].0, "A");
/// assert_eq!(top[1].0, "B");
/// ```
#[must_use]
pub fn top_n(scores: &HashMap<String, f64>, n: usize) -> Vec<(String, f64)> {
    let mut entries: Vec<(String, f64)> = scores.iter().map(|(k, &v)| (k.clone(), v)).collect();
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    entries.truncate(n);
    entries
}
