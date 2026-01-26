//! Knowledge graph substrate: core types + basic algorithms + formats.
//!
//! This crate is intentionally *minimal*: it re-exports `lattix-core` and nothing higher-level.
//! Higher-level KG systems (training, reasoning, temporal systems, CLI) live in `webs/*`.
//!
//! # Example
//!
//! ```rust
//! use lattix::{KnowledgeGraph, Triple};
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
//! use lattix::KnowledgeGraph;
//!
//! // Load single file
//! let kg = KnowledgeGraph::from_ntriples_file("kg/document.nt")?;
//!
//! // Or load entire directory of .nt files
//! let kg = lattix::load_anno_exports("kg/")?;
//!
//! // Run PageRank on extracted entities
//! use lattix::algo::pagerank::{pagerank, PageRankConfig};
//! let scores = pagerank(&kg, PageRankConfig::default());
//!
//! // Generate random walks for node embedding training
//! use lattix::algo::random_walk::{generate_walks, RandomWalkConfig};
//! let walks = generate_walks(&kg, RandomWalkConfig::default());
//! ```

// Re-export core types
pub use lattix_core::{
    EdgeStore, EdgeType, Entity, EntityId, Error, HeteroGraph, HeteroGraphStats, KnowledgeGraph,
    GraphDocument, GraphEdge, GraphExportFormat, GraphNode, KnowledgeGraphStats, NodeStore,
    NodeType, Relation, RelationType, Result, Triple,
};

/// Back-compat aliases (older code used these names).
pub use Error as CoreError;
pub use Result as CoreResult;

// Re-export petgraph for advanced graph operations
pub use lattix_core::petgraph;

/// RDF serialization formats (RDF 1.2).
pub mod formats {
    pub use lattix_core::formats::*;
}

/// Graph algorithms: PageRank, random walks, connected components.
pub mod algo {
    pub use lattix_core::algo::*;
}

// NOTE: Higher-level modules intentionally do not live here.

/// Load all N-Triples files from a directory (e.g., anno export directory).
///
/// Merges all .nt files into a single KnowledgeGraph.
///
/// # Example
///
/// ```rust,ignore
/// // After: anno export -i docs/ -o kg/ --format ntriples
/// let kg = lattix::load_anno_exports("kg/")?;
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
