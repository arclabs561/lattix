//! Core types for knowledge graphs.
//!
//! This crate provides foundational types for working with knowledge graphs:
//!
//! - [`Triple`] - A (subject, predicate, object) triple
//! - [`Entity`] - A node in the knowledge graph
//! - [`Relation`] - An edge type in the knowledge graph
//! - [`KnowledgeGraph`] - A graph structure built from triples
//!
//! # Example
//!
//! ```rust
//! use lattice_core::{Triple, KnowledgeGraph};
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
//!
//! # Integration with anno
//!
//! Import triples from anno's N-Triples export:
//!
//! ```rust,ignore
//! use lattice_core::KnowledgeGraph;
//!
//! let kg = KnowledgeGraph::from_ntriples_file("entities.nt")?;
//! ```

mod entity;
mod error;
mod graph;
mod relation;
mod triple;

pub use entity::{Entity, EntityId};
pub use error::{Error, Result};
pub use graph::{KnowledgeGraph, KnowledgeGraphStats};
pub use relation::{Relation, RelationType};
pub use triple::Triple;

// Re-export petgraph for advanced graph operations
pub use petgraph;
