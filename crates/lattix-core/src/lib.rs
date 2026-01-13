// Allow minor clippy style warnings at crate level
// These are mostly style preferences, not bugs
#![allow(unused_results)]
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
//! # Historical Context: From Databases to Graphs
//!
//! | Era | Representation | Example | Limitation |
//! |-----|----------------|---------|------------|
//! | 1970s | Relational | SQL tables | Fixed schema, join-heavy |
//! | 2001 | RDF | Semantic Web | XML verbosity, query complexity |
//! | 2012 | Property Graphs | Neo4j | No standard query language |
//! | 2019 | GNN-ready graphs | PyG, DGL | Framework-specific formats |
//!
//! Knowledge graphs became essential when search engines realized that
//! "things, not strings" (Google, 2012) captures real-world semantics
//! that keyword matching misses.
//!
//! # The Triple: Atomic Unit of Knowledge
//!
//! All knowledge graph formalisms reduce to the **triple**:
//!
//! ```text
//! (subject, predicate, object)
//! (Albert Einstein, born_in, Ulm)
//! (Ulm, located_in, Germany)
//! ```
//!
//! This simple structure enables:
//! - **Inference**: If A → B → C, deduce A → C
//! - **Composition**: Merge graphs by merging shared entities
//! - **Embedding**: Represent triples as vectors for ML
//!
//! # Beyond Triples: N-ary Relations and Hypergraphs
//!
//! Triples are limited - they can only express binary relations. Real-world
//! facts often involve more than two entities:
//!
//! ```text
//! (Einstein, won, Nobel Prize, Physics, 1921)  -- 4 entities!
//! (Alice, purchased, Book, $20, Amazon, 2024-01-15)  -- 6 entities!
//! ```
//!
//! Three approaches handle this:
//!
//! | Approach | Structure | Example | Trade-off |
//! |----------|-----------|---------|-----------|
//! | **Reification** | Break into multiple triples | Creates artificial nodes | Information loss |
//! | **Qualifiers** | Triple + key-value pairs | Wikidata model | Complex querying |
//! | **Hyperedges** | N-ary relation directly | Native structure | New embeddings needed |
//!
//! ## Reification (Workaround)
//!
//! Convert n-ary facts to binary by introducing intermediate nodes:
//!
//! ```text
//! Original: (Einstein, won, Nobel, Physics, 1921)
//! Reified:  (Award_1, recipient, Einstein)
//!           (Award_1, prize, Nobel)
//!           (Award_1, field, Physics)
//!           (Award_1, year, 1921)
//! ```
//!
//! **Problem**: Loses the atomic nature of the fact. Embedding models struggle
//! because Award_1 is artificial - it has no semantic meaning.
//!
//! ## Hyper-relational KGs (Wikidata Style)
//!
//! Attach qualifiers to triples:
//!
//! ```text
//! (Einstein, won, Nobel Prize)
//!   qualifiers: {field: Physics, year: 1921}
//! ```
//!
//! Implemented in [`hyper::HyperTriple`]. Embeddings: StarE, HINGE.
//!
//! ## Knowledge Hypergraphs (Native N-ary)
//!
//! Represent facts as hyperedges connecting multiple entities with roles:
//!
//! ```text
//! HyperEdge {
//!   relation: "award_ceremony",
//!   bindings: {
//!     recipient: Einstein,
//!     prize: Nobel,
//!     field: Physics,
//!     year: 1921
//!   }
//! }
//! ```
//!
//! Implemented in [`hyper::HyperEdge`]. Embeddings: HSimplE, HypE, HyCubE.
//!
//! **Key insight**: Position-aware or role-aware encoding is essential.
//! The relation "recipient" carries different semantics than "year".
//!
//! See [`hyper`] module for hypergraph types.
//!
//! # Homogeneous vs Heterogeneous Graphs
//!
//! | Type | Nodes | Edges | Use Case |
//! |------|-------|-------|----------|
//! | Homogeneous | One type | One type | Citation networks, social graphs |
//! | Heterogeneous | Multiple types | Multiple types | Knowledge graphs, biomedical |
//!
//! [`HeteroGraph`] supports typed nodes and edges, essential for:
//! - **RGCN**: Relation-specific weight matrices
//! - **HGT**: Heterogeneous graph transformers
//! - **Link prediction**: Typed edge prediction
//!
//! # Serialization Formats
//!
//! Supports modern RDF 1.2 specifications (2024):
//! - N-Triples - Line-based, simple (fastest parsing)
//! - N-Quads - N-Triples with named graphs
//! - Turtle - Human-readable (best for debugging)
//! - JSON-LD - Linked data (web integration)
//!
//! # Algorithms
//!
//! - [`algo::random_walk`] - Node2Vec style random walks (biased BFS/DFS)
//! - [`algo::pagerank`] - PageRank centrality (importance ranking)
//! - [`algo::components`] - Connected components (graph structure)
//! - [`algo::sampling`] - Neighbor sampling for mini-batch GNN training
//!
//! # When to Use Which Structure
//!
//! | Task | Structure | Why |
//! |------|-----------|-----|
//! | Node classification | KnowledgeGraph | Homogeneous GCN/GAT |
//! | Link prediction | HeteroGraph | Relation types matter |
//! | Knowledge completion | HeteroGraph + embeddings | TransE, RotatE, BoxE |
//! | Graph classification | KnowledgeGraph | Global pooling over nodes |
//!
//! # Example
//!
//! ```rust
//! use lattix_core::{Triple, KnowledgeGraph};
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
pub mod hyper;
mod relation;
mod triple;

#[cfg(feature = "sophia")]
mod sophia_impl;

pub use entity::{Entity, EntityId};
pub use error::{Error, Result};
pub use graph::{KnowledgeGraph, KnowledgeGraphStats};
pub use hetero::{EdgeStore, EdgeType, HeteroGraph, HeteroGraphStats, NodeStore, NodeType};
pub use hyper::{HyperEdge, HyperGraph, HyperTriple, RoleBinding};
pub use relation::{Relation, RelationType};
pub use triple::Triple;

// Re-export petgraph for advanced graph operations
pub use petgraph;
