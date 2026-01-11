//! Knowledge graph construction and embedding inference.
//!
//! `lattice` is a Rust library for working with knowledge graphs:
//!
//! - Parse and build knowledge graphs from triples
//! - Import from N-Triples format (from anno export)
//! - KG embedding inference (TransE, RotatE, etc.)
//! - Link prediction and entity similarity
//!
//! # Crate Structure
//!
//! - [`lattice_core`] - Core types: Triple, Entity, Relation, KnowledgeGraph
//! - [`lattice_embed`] - KGE inference from pre-trained models
//!
//! # Example
//!
//! ```rust
//! use lattice::{KnowledgeGraph, Triple};
//!
//! // Build a knowledge graph
//! let mut kg = KnowledgeGraph::new();
//! kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));
//! kg.add_triple(Triple::new("Apple", "headquartered_in", "Cupertino"));
//!
//! // Query
//! let founders = kg.relations_from("Apple");
//! println!("Apple has {} relations", founders.len());
//!
//! // Statistics
//! let stats = kg.stats();
//! println!("Entities: {}, Triples: {}", stats.entity_count, stats.triple_count);
//! ```
//!
//! # Integration with anno
//!
//! Import knowledge graphs from anno's N-Triples export:
//!
//! ```rust,ignore
//! // First, export from anno:
//! // anno export -i docs/ -o kg/ --format ntriples
//!
//! use lattice::KnowledgeGraph;
//!
//! let kg = KnowledgeGraph::from_ntriples_file("kg/document.nt")?;
//! ```
//!
//! # KG Embedding Inference
//!
//! Load pre-trained models for link prediction:
//!
//! ```rust,ignore
//! use lattice::embed::{TransEOnnx, KGEmbedding};
//!
//! // Load model trained in Python (PyKEEN)
//! let model = TransEOnnx::from_file("transE.onnx", "entities.json", "relations.json")?;
//!
//! // Score a triple
//! let score = model.score("Apple", "founded_by", "Steve Jobs")?;
//!
//! // Link prediction
//! let candidates = model.predict_tail("Apple", "founded_by", 10)?;
//! ```

// Re-export core types
pub use lattice_core::{
    Entity, EntityId, Error as CoreError, KnowledgeGraph, KnowledgeGraphStats, Relation,
    RelationType, Result as CoreResult, Triple,
};

// Re-export petgraph for advanced graph operations
pub use lattice_core::petgraph;

/// RDF serialization formats (RDF 1.2).
pub mod formats {
    pub use lattice_core::formats::*;
}

/// Embedding inference module.
pub mod embed {
    pub use lattice_embed::*;
}
