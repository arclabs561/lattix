//! Algorithms for graph analysis and embedding generation.
//!
//! This module contains implementations of graph algorithms such as random walks.

/// Random walk algorithm (Node2Vec style).
pub mod random_walk;

/// PageRank centrality algorithm.
pub mod pagerank;

/// Connected components algorithm.
pub mod components;

/// Graph sampling algorithms (e.g. for GNNs).
pub mod sampling;
