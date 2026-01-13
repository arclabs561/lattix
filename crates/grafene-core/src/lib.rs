// Allow minor clippy style warnings at crate level
// These are mostly style preferences, not bugs
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::match_single_binding)]
#![allow(clippy::manual_strip)]
#![allow(clippy::items_after_statements)]

//! Core types for knowledge graphs.
//!
//! This crate provides foundational types for working with knowledge graphs:
//!
//! - [`Triple`] - A (subject, predicate, object) triple
//! - [`Entity`] - A node in the knowledge graph
//! - [`Relation`] - An edge type in the knowledge graph
//! - [`KnowledgeGraph`] - A homogeneous graph structure built from triples
//! - [`HeteroGraph`] - A heterogeneous graph with typed nodes and edges
//!
//! # Serialization Formats
//!
//! Supports modern RDF 1.2 specifications (2024):
//! - N-Triples - Line-based, simple
//! - N-Quads - N-Triples with named graphs
//! - Turtle - Human-readable
//! - JSON-LD - Linked data
//!
//! # Algorithms
//!
//! - [`algo::random_walk`] - `Node2Vec` style random walks
//! - [`algo::pagerank`] - `PageRank` centrality
//! - [`algo::components`] - Connected components
//! - [`algo::sampling`] - Neighbor sampling for mini-batch GNN training
//!
//! # Example
//!
//! ```rust
//! use grafene_core::{Triple, KnowledgeGraph};
//!
//! let mut kg = KnowledgeGraph::new();
//!
//! // Add triples
//! kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));
//! kg.add_triple(Triple::new("Apple", "headquartered_in", "Cupertino"));
//! kg.add_triple(Triple::new("Steve Jobs", "born_in", "San Francisco"));
//!
//! // Query
//! let apple_relations = kg.relations_from("Apple");
//! assert_eq!(apple_relations.len(), 2);
//! ```

pub mod algo;
mod entity;
mod error;
pub mod formats;
mod graph;
pub mod hetero;
mod relation;
mod triple;

#[cfg(feature = "sophia")]
mod sophia_impl;

pub use entity::{Entity, EntityId};
pub use error::{Error, Result};
pub use graph::{KnowledgeGraph, KnowledgeGraphStats};
pub use hetero::{EdgeStore, EdgeType, HeteroGraph, HeteroGraphStats, NodeStore, NodeType};
pub use relation::{Relation, RelationType};
pub use triple::Triple;

// Re-export petgraph for advanced graph operations
pub use petgraph;
