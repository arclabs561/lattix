//! QA correctness probes for lattix v0.5.0.
//!
//! Covers changes since commit 7575936 (v0.3.0 QA baseline):
//! - v0.4.0: Triple/EntityId/RelationType/EdgeStore private fields + accessor API
//! - v0.4.0: generate_walk_corpus() returns Result
//! - v0.5.0: Error enum is #[non_exhaustive] with Serialization variant
//! - rand_distr removed from dependencies

use lattix::{
    EdgeStore, EdgeType, EntityId, Error, HeteroGraph, KnowledgeGraph, NodeType, RelationType,
    Triple,
};

// ============================================================
// A. Accessor API completeness
// ============================================================

mod accessor_api {
    use super::*;

    // -- Triple --

    #[test]
    fn triple_new_and_accessors() {
        let t = Triple::new("Alice", "knows", "Bob");
        assert_eq!(t.subject().as_str(), "Alice");
        assert_eq!(t.predicate().as_str(), "knows");
        assert_eq!(t.object().as_str(), "Bob");
        assert_eq!(t.confidence(), None, "default confidence should be None");
        assert_eq!(t.source(), None, "default source should be None");
    }

    #[test]
    fn triple_with_confidence_normal() {
        let t = Triple::new("A", "r", "B").with_confidence(0.5);
        assert_eq!(t.confidence(), Some(0.5));
    }

    #[test]
    fn triple_with_confidence_clamps_below_zero() {
        let t = Triple::new("A", "r", "B").with_confidence(-0.1);
        assert_eq!(
            t.confidence(),
            Some(0.0),
            "negative confidence must clamp to 0.0"
        );
    }

    #[test]
    fn triple_with_confidence_clamps_above_one() {
        let t = Triple::new("A", "r", "B").with_confidence(1.5);
        assert_eq!(
            t.confidence(),
            Some(1.0),
            "confidence > 1.0 must clamp to 1.0"
        );
    }

    #[test]
    fn triple_with_source() {
        let t = Triple::new("A", "r", "B").with_source("doc.txt");
        assert_eq!(t.source(), Some("doc.txt"));
    }

    // -- EntityId --

    #[test]
    fn entity_id_new_as_str_into_string() {
        let eid = EntityId::new("hello");
        assert_eq!(eid.as_str(), "hello");
        let s: String = eid.into_string();
        assert_eq!(s, "hello");
    }

    // -- RelationType --

    #[test]
    fn relation_type_new_as_str_into_string() {
        let rt = RelationType::new("knows");
        assert_eq!(rt.as_str(), "knows");
        let s: String = rt.into_string();
        assert_eq!(s, "knows");
    }

    // -- EdgeStore --

    #[test]
    fn edge_store_accessors_after_add_edge() {
        let mut store = EdgeStore::new();
        store.add_edge(0, 1);
        store.add_edge(0, 2);
        store.add_edge(1, 2);

        assert_eq!(store.src(), &[0, 0, 1]);
        assert_eq!(store.dst(), &[1, 2, 2]);

        let (src, dst) = store.edge_index();
        assert_eq!(src, &[0, 0, 1]);
        assert_eq!(dst, &[1, 2, 2]);

        assert_eq!(store.num_edges(), 3);
        assert!(!store.is_empty());

        // Verify adjacency works through accessors
        assert_eq!(store.neighbors(0), &[1, 2]);
        assert_eq!(store.neighbors(1), &[2]);
        assert_eq!(store.incoming(2), &[0, 1]);
    }
}

// ============================================================
// B. generate_walk_corpus error handling
// ============================================================

#[cfg(feature = "algo")]
mod walk_corpus {
    use super::*;
    use lattix::algo::random_walk::{generate_walk_corpus, RandomWalkConfig};

    #[test]
    fn walk_corpus_valid_graph_ok() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "r", "B"));
        kg.add_triple(Triple::new("B", "r", "C"));
        kg.add_triple(Triple::new("C", "r", "A"));

        let config = RandomWalkConfig {
            walk_length: 5,
            num_walks: 2,
            seed: 42,
            ..Default::default()
        };

        let result = generate_walk_corpus(&kg, config);
        assert!(result.is_ok(), "valid graph should return Ok");
        let corpus = result.unwrap();
        assert_eq!(corpus.node_ids.len(), 3, "3 entities -> 3 node IDs");
        assert!(!corpus.walks.is_empty(), "should have walks");

        // All dense indices in walks must be < node_ids.len()
        for walk in &corpus.walks {
            for &idx in walk {
                assert!(
                    (idx as usize) < corpus.node_ids.len(),
                    "dense index {} out of bounds (N={})",
                    idx,
                    corpus.node_ids.len()
                );
            }
        }
    }

    #[test]
    fn walk_corpus_empty_graph_ok() {
        let kg = KnowledgeGraph::new();
        let config = RandomWalkConfig {
            walk_length: 5,
            num_walks: 2,
            seed: 42,
            ..Default::default()
        };

        let result = generate_walk_corpus(&kg, config);
        assert!(
            result.is_ok(),
            "empty graph should return Ok, got {:?}",
            result.err()
        );
        let corpus = result.unwrap();
        assert!(corpus.node_ids.is_empty());
        assert!(corpus.walks.is_empty());
    }
}

// ============================================================
// C. Error::Serialization
// ============================================================

mod error_serialization {
    use super::*;

    #[test]
    fn error_serialization_from_custom_error() {
        let inner = std::io::Error::new(std::io::ErrorKind::Other, "boom");
        let err = Error::Serialization(Box::new(inner));
        let msg = format!("{}", err);
        assert!(
            msg.contains("boom"),
            "display should contain inner error text, got: {}",
            msg
        );
    }

    #[test]
    fn error_serialization_display_contains_inner() {
        let inner_msg = "custom codec failure XYZ-123";
        let inner = std::io::Error::new(std::io::ErrorKind::InvalidData, inner_msg);
        let err = Error::Serialization(Box::new(inner));
        let display = format!("{}", err);
        assert!(
            display.contains(inner_msg),
            "expected display to contain '{}', got: '{}'",
            inner_msg,
            display
        );
    }

    /// Compile-time check: Error must be Send + Sync.
    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }
}

// ============================================================
// D. Serde round-trip with private fields
// ============================================================

mod serde_roundtrip {
    use super::*;

    #[test]
    fn kg_json_roundtrip() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("Alice", "knows", "Bob"));
        kg.add_triple(Triple::new("Bob", "works_at", "Acme"));
        kg.add_triple(Triple::new("Alice", "likes", "Pizza").with_confidence(0.9));

        let json = serde_json::to_string(&kg).expect("serialize KG to JSON");
        let loaded: KnowledgeGraph = serde_json::from_str(&json).expect("deserialize KG from JSON");

        assert_eq!(loaded.triple_count(), 3);
        assert_eq!(loaded.entity_count(), kg.entity_count());

        // Verify indexes rebuilt: queryable
        let alice_rels = loaded.relations_from("Alice");
        assert_eq!(alice_rels.len(), 2, "Alice has 2 outgoing triples");

        let bob_incoming = loaded.relations_to("Bob");
        assert_eq!(bob_incoming.len(), 1);

        // Verify predicate index
        let knows_triples = loaded.triples_with_relation("knows");
        assert_eq!(knows_triples.len(), 1);

        // Verify accessor methods on deserialized triples
        let alice_triple = loaded
            .triples()
            .find(|t| t.predicate().as_str() == "likes")
            .expect("likes triple should exist");
        assert_eq!(alice_triple.subject().as_str(), "Alice");
        assert_eq!(alice_triple.object().as_str(), "Pizza");
        assert_eq!(alice_triple.confidence(), Some(0.9));
    }

    /// HeteroGraph JSON round-trip: edge_stores serialized as Vec<(EdgeType, EdgeStore)>
    /// to avoid serde_json's "key must be a string" limitation.
    #[test]
    fn heterograph_json_roundtrip() {
        let mut hg = HeteroGraph::new();
        let buys = EdgeType::new("user", "buys", "item");
        hg.add_edge(&buys, "alice", "book1");
        hg.add_edge(&buys, "bob", "book2");

        let json = serde_json::to_string(&hg).expect("HeteroGraph should serialize to JSON");
        let loaded: HeteroGraph =
            serde_json::from_str(&json).expect("HeteroGraph should deserialize from JSON");

        // Verify structure survived
        let store = loaded.edge_store(&buys).expect("buys edge store");
        assert_eq!(store.src().len(), 2);
        assert_eq!(store.dst().len(), 2);
        // Verify adjacency was rebuilt (neighbors works)
        assert!(!store.neighbors(0).is_empty() || !store.neighbors(1).is_empty());
    }

    /// HeteroGraph round-trips through bincode (binary format handles struct keys).
    #[cfg(feature = "binary")]
    #[test]
    fn heterograph_bincode_roundtrip() {
        let mut hg = HeteroGraph::new();
        let buys = EdgeType::new("user", "buys", "item");
        hg.add_edge(&buys, "alice", "book1");
        hg.add_edge(&buys, "alice", "book2");
        hg.add_edge(&buys, "bob", "book1");

        let bytes = bincode::serialize(&hg).expect("bincode serialize HeteroGraph");
        let loaded: HeteroGraph =
            bincode::deserialize(&bytes).expect("bincode deserialize HeteroGraph");

        // Must rebuild adjacency after bincode deser (fwd_adj/rev_adj are #[serde(skip)])
        // The custom Deserialize impl should handle this.
        assert_eq!(loaded.total_nodes(), 4);
        assert_eq!(loaded.total_edges(), 3);

        // Verify neighbors() works (adjacency indexes rebuilt on deser)
        let alice_idx = loaded
            .get_node_index(&NodeType::new("user"), "alice")
            .expect("alice should exist");
        let neighbors = loaded.neighbors(&buys, alice_idx);
        assert_eq!(
            neighbors.len(),
            2,
            "alice should have 2 neighbors after bincode deser"
        );

        // Verify incoming also works
        let book1_idx = loaded
            .get_node_index(&NodeType::new("item"), "book1")
            .expect("book1 should exist");
        let incoming = loaded.incoming_neighbors(&buys, book1_idx);
        assert_eq!(incoming.len(), 2, "book1 should have 2 incoming");
    }

    #[cfg(feature = "binary")]
    #[test]
    fn kg_binary_roundtrip_to_file() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("X", "r1", "Y"));
        kg.add_triple(Triple::new("Y", "r2", "Z"));
        kg.add_triple(
            Triple::new("X", "r3", "Z")
                .with_confidence(0.75)
                .with_source("test"),
        );

        let path = std::env::temp_dir().join("qa_v050_binary.bin");
        kg.to_binary_file(&path).expect("write binary");

        let loaded = KnowledgeGraph::from_binary_file(&path).expect("read binary");
        assert_eq!(loaded.triple_count(), 3);
        assert_eq!(loaded.entity_count(), 3);

        // Verify indexes rebuilt
        assert_eq!(loaded.relations_from("X").len(), 2);
        assert_eq!(loaded.relations_to("Z").len(), 2);
        assert_eq!(loaded.relation_type_count(), 3);

        // Verify accessor on deserialized triple
        let r3 = loaded
            .triples()
            .find(|t| t.predicate().as_str() == "r3")
            .expect("r3 triple");
        assert_eq!(r3.confidence(), Some(0.75));
        assert_eq!(r3.source(), Some("test"));

        std::fs::remove_file(&path).ok();
    }
}

// ============================================================
// E. sophia_api with private fields
// ============================================================

#[cfg(feature = "sophia")]
mod sophia_bridge {
    use super::*;
    use sophia_api::graph::{Graph, MutableGraph};
    use sophia_api::ns::Namespace;
    use sophia_api::term::matcher::Any;

    #[test]
    fn insert_via_mutable_graph_visible_in_graph() {
        let mut kg = KnowledgeGraph::new();
        let ex = Namespace::new("http://example.org/").unwrap();
        let alice = ex.get("Alice").unwrap();
        let knows = ex.get("knows").unwrap();
        let bob = ex.get("Bob").unwrap();

        let inserted = MutableGraph::insert(&mut kg, &alice, &knows, &bob).unwrap();
        assert!(inserted, "insert should return true");
        assert_eq!(kg.triple_count(), 1);

        // Verify via Graph iteration
        let count = Graph::triples(&kg).count();
        assert_eq!(count, 1);

        // Verify via triples_matching
        let matches: Vec<_> = kg
            .triples_matching([alice.clone()], Any, Any)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(matches.len(), 1);

        // Verify accessor on the stored triple
        let t = kg.triples().next().unwrap();
        assert_eq!(t.subject().as_str(), "http://example.org/Alice");
        assert_eq!(t.predicate().as_str(), "http://example.org/knows");
        assert_eq!(t.object().as_str(), "http://example.org/Bob");
    }

    #[test]
    fn remove_via_mutable_graph_disappears() {
        let mut kg = KnowledgeGraph::new();
        let ex = Namespace::new("http://example.org/").unwrap();
        let alice = ex.get("Alice").unwrap();
        let knows = ex.get("knows").unwrap();
        let bob = ex.get("Bob").unwrap();
        let carol = ex.get("Carol").unwrap();

        MutableGraph::insert(&mut kg, &alice, &knows, &bob).unwrap();
        MutableGraph::insert(&mut kg, &alice, &knows, &carol).unwrap();
        assert_eq!(kg.triple_count(), 2);

        let removed = MutableGraph::remove(&mut kg, &alice, &knows, &bob).unwrap();
        assert!(removed);
        assert_eq!(kg.triple_count(), 1);

        // The remaining triple should be Alice->Carol
        assert!(!kg.contains(&alice, &knows, &bob).unwrap());
        assert!(kg.contains(&alice, &knows, &carol).unwrap());

        // Verify accessor on remaining triple
        let t = kg.triples().next().unwrap();
        assert_eq!(t.object().as_str(), "http://example.org/Carol");
    }
}

// ============================================================
// F. Format round-trips with accessor API
// ============================================================

#[cfg(feature = "formats")]
mod format_roundtrips {
    use super::*;
    use lattix::formats::{JsonLd, NTriples};

    #[test]
    fn ntriples_roundtrip_with_blank_nodes_and_iris() {
        let mut kg = KnowledgeGraph::new();
        // IRI triple
        kg.add_triple(Triple::new(
            "http://example.org/Alice",
            "http://example.org/knows",
            "http://example.org/Bob",
        ));
        // Blank node subject
        kg.add_triple(Triple::new(
            "_:node1",
            "http://example.org/label",
            "http://example.org/Thing",
        ));

        let serialized = NTriples::to_string(&kg).expect("NTriples serialize");

        // Verify serialization format
        assert!(
            serialized.contains("<http://example.org/Alice>"),
            "IRI should be bracketed"
        );
        assert!(
            serialized.contains("_:node1 "),
            "blank node must not have angle brackets"
        );
        assert!(
            !serialized.contains("<_:node1>"),
            "blank node must NOT have angle brackets"
        );

        // Parse back
        let loaded = NTriples::parse(&serialized).expect("NTriples parse");
        assert_eq!(loaded.triple_count(), 2);

        // Verify accessor methods on parsed triples
        let alice_triple = loaded
            .triples()
            .find(|t| t.subject().as_str() == "http://example.org/Alice")
            .expect("Alice triple should parse back");
        assert_eq!(
            alice_triple.predicate().as_str(),
            "http://example.org/knows"
        );
        assert_eq!(alice_triple.object().as_str(), "http://example.org/Bob");

        let bnode_triple = loaded
            .triples()
            .find(|t| t.subject().as_str() == "_:node1")
            .expect("blank node triple should parse back");
        assert_eq!(
            bnode_triple.predicate().as_str(),
            "http://example.org/label"
        );
        assert_eq!(bnode_triple.object().as_str(), "http://example.org/Thing");
    }

    #[test]
    fn jsonld_roundtrip() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new(
            "http://example.org/Alice",
            "http://example.org/knows",
            "http://example.org/Bob",
        ));
        kg.add_triple(Triple::new(
            "http://example.org/Alice",
            "http://example.org/age",
            "30",
        ));

        let json_str = JsonLd::to_string(&kg).expect("JSON-LD serialize");
        assert!(json_str.contains("@graph"));

        let loaded = JsonLd::parse(&json_str).expect("JSON-LD parse");

        // JSON-LD round-trip: URI objects preserved, literal "30" also preserved
        assert_eq!(loaded.triple_count(), 2);

        // Verify accessor on loaded triples
        let knows = loaded
            .triples()
            .find(|t| t.predicate().as_str() == "http://example.org/knows")
            .expect("knows triple");
        assert_eq!(knows.subject().as_str(), "http://example.org/Alice");
        assert_eq!(knows.object().as_str(), "http://example.org/Bob");

        let age = loaded
            .triples()
            .find(|t| t.predicate().as_str() == "http://example.org/age")
            .expect("age triple");
        assert_eq!(age.object().as_str(), "30");
    }
}
