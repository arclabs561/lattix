//! Algorithms for graph analysis and embedding generation.
//!
//! This module contains implementations of graph algorithms:
//!
//! - **Centrality**: Measure node importance ([`centrality`])
//! - **Random walks**: Node2Vec-style biased walks ([`random_walk`])
//! - **Components**: Find connected components ([`components`])
//! - **Sampling**: Mini-batch sampling for GNNs ([`sampling`])
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
