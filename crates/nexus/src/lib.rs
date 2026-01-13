//! Graph learning for Rust: GNN, knowledge graphs, temporal graphs.
//!
//! `nexus` is a Rust library for graph learning and knowledge graphs:
//!
//! - Parse and build knowledge graphs from triples
//! - Import from N-Triples format (from anno export)
//! - GNN layers (GCN, GAT, GraphSAGE)
//! - KG embedding training and inference
//! - Temporal graph operations
//!
//! # Crate Structure
//!
//! - [`nexus_core`] - Core types: Triple, Entity, Relation, KnowledgeGraph
//! - [`nexus_nn`] - Graph neural network layers, Node2Vec, hyperbolic GNN
//! - [`nexus_kge`] - Knowledge Graph Embeddings (TransE, RotatE, ComplEx)
//! - [`nexus_temporal`] - Temporal graph primitives
//!
//! # Example
//!
//! ```rust
//! use nexus::{KnowledgeGraph, Triple};
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
//! use nexus::KnowledgeGraph;
//!
//! // Load single file
//! let kg = KnowledgeGraph::from_ntriples_file("kg/document.nt")?;
//!
//! // Or load entire directory of .nt files
//! let kg = nexus::load_anno_exports("kg/")?;
//!
//! // Run PageRank on extracted entities
//! use nexus::algo::pagerank::{pagerank, PageRankConfig};
//! let scores = pagerank(&kg, PageRankConfig::default());
//!
//! // Generate random walks for node embedding training
//! use nexus::algo::random_walk::{generate_walks, RandomWalkConfig};
//! let walks = generate_walks(&kg, RandomWalkConfig::default());
//! ```

// Re-export core types
pub use nexus_core::{
    EdgeStore, EdgeType, Entity, EntityId, Error as CoreError, HeteroGraph, HeteroGraphStats,
    KnowledgeGraph, KnowledgeGraphStats, NodeStore, NodeType, Relation, RelationType,
    Result as CoreResult, Triple,
};

// Re-export petgraph for advanced graph operations
pub use nexus_core::petgraph;

/// RDF serialization formats (RDF 1.2).
pub mod formats {
    pub use nexus_core::formats::*;
}

/// Graph algorithms: PageRank, random walks, connected components.
pub mod algo {
    pub use nexus_core::algo::*;
}

/// Knowledge Graph Embedding inference (requires `kge` or `onnx` feature).
#[cfg(feature = "kge")]
pub mod kge {
    pub use nexus_kge::*;
}

/// GNN layers (requires `nn` or `gnn` feature).
#[cfg(feature = "nn")]
pub mod nn {
    pub use nexus_nn::*;
}

// Backwards compatibility aliases
#[cfg(feature = "kge")]
pub use kge as embed;

#[cfg(feature = "nn")]
pub use nn as gnn;

/// Temporal graph module (requires `temporal` feature).
#[cfg(feature = "temporal")]
pub mod temporal {
    pub use nexus_temporal::*;
}

/// Load all N-Triples files from a directory (e.g., anno export directory).
///
/// Merges all .nt files into a single KnowledgeGraph.
///
/// # Example
///
/// ```rust,ignore
/// // After: anno export -i docs/ -o kg/ --format ntriples
/// let kg = nexus::load_anno_exports("kg/")?;
/// ```
pub fn load_anno_exports(dir: impl AsRef<std::path::Path>) -> CoreResult<KnowledgeGraph> {
    use std::fs;

    let dir = dir.as_ref();
    let mut kg = KnowledgeGraph::new();

    let entries = fs::read_dir(dir)?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "nt") {
            let file_kg = KnowledgeGraph::from_ntriples_file(&path)?;
            // Merge triples
            for triple in file_kg.triples() {
                kg.add_triple(triple.clone());
            }
        }
    }

    Ok(kg)
}
