// QA exercise tests for lattix v0.3.0 (qa-2026-03-05)
// Covers: core types, algorithms, formats, HeteroGraph, HyperGraph, edge cases

use lattix::{KnowledgeGraph, Triple};

// ── 4a. Core types ──

#[test]
fn qa_triple_construction() {
    let t = Triple::new("Alice", "knows", "Bob");
    assert_eq!(t.subject().as_str(), "Alice");
    assert_eq!(t.predicate().as_str(), "knows");
    assert_eq!(t.object().as_str(), "Bob");
}

#[test]
fn qa_confidence_clamping() {
    let t = Triple::new("A", "r", "B").with_confidence(1.5);
    assert!(t.confidence().unwrap() <= 1.0);
    let t = Triple::new("A", "r", "B").with_confidence(-0.5);
    assert!(t.confidence().unwrap() >= 0.0);
    let t = Triple::new("A", "r", "B").with_confidence(0.75);
    assert!((t.confidence().unwrap() - 0.75).abs() < 1e-10);
}

#[test]
fn qa_kg_basics() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("Alice", "knows", "Bob"));
    kg.add_triple(Triple::new("Alice", "works_at", "Acme"));
    kg.add_triple(Triple::new("Bob", "works_at", "Acme"));
    assert_eq!(kg.entity_count(), 3);
    assert_eq!(kg.triple_count(), 3);

    let from_alice = kg.relations_from("Alice");
    assert_eq!(from_alice.len(), 2);
    let to_acme = kg.relations_to("Acme");
    assert_eq!(to_acme.len(), 2);
}

#[test]
fn qa_path_finding() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("A", "r", "B"));
    kg.add_triple(Triple::new("B", "r", "C"));
    kg.add_triple(Triple::new("C", "r", "D"));

    let path = kg.find_path("A", "D");
    assert!(path.is_some());
    let path = path.unwrap();
    assert_eq!(path.len(), 3); // 3 edges, not 4 nodes
    assert_eq!(path[0].subject().as_str(), "A");
    assert_eq!(path[2].object().as_str(), "D");

    // No path in reverse (directed graph)
    assert!(kg.find_path("D", "A").is_none());
}

#[test]
fn qa_remove_triple_index_consistency() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("Alice", "knows", "Bob"));
    kg.add_triple(Triple::new("Alice", "works_at", "Acme"));
    kg.add_triple(Triple::new("Bob", "works_at", "Acme"));

    let removed = kg.remove_triple(&"Alice".into(), &"knows".into(), &"Bob".into());
    assert!(removed);
    assert_eq!(kg.triple_count(), 2);
    assert_eq!(kg.relations_from("Alice").len(), 1);
    // Bob still exists via Bob->works_at->Acme
    assert_eq!(kg.relations_from("Bob").len(), 1);
}

#[test]
fn qa_clear_resets_all() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("A", "r", "B"));
    kg.add_triple(Triple::new("C", "r", "D"));
    kg.clear();
    assert_eq!(kg.entity_count(), 0);
    assert_eq!(kg.triple_count(), 0);
    assert!(kg.relations_from("A").is_empty());

    // Works after clear
    kg.add_triple(Triple::new("X", "r", "Y"));
    assert_eq!(kg.entity_count(), 2);
    assert_eq!(kg.triple_count(), 1);
}

// ── 4b. Algorithms ──

#[test]
fn qa_pagerank_star() {
    use lattix::algo::pagerank::*;

    let mut kg = KnowledgeGraph::new();
    for leaf in ["A", "B", "C", "D"] {
        kg.add_triple(Triple::new("Hub", "r", leaf));
        kg.add_triple(Triple::new(leaf, "r", "Hub"));
    }

    let pr = pagerank(&kg, PageRankConfig::default());
    let hub_pr = pr["Hub"];
    let leaf_pr = pr["A"];
    assert!(hub_pr > leaf_pr, "Hub should have higher PageRank");
    let sum: f64 = pr.values().sum();
    assert!((sum - 1.0).abs() < 1e-4, "PageRank sums to 1, got {sum}");
}

#[test]
fn qa_pagerank_complete() {
    use lattix::algo::pagerank::*;

    let mut kg = KnowledgeGraph::new();
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                kg.add_triple(Triple::new(format!("N{i}"), "r", format!("N{j}")));
            }
        }
    }
    let pr = pagerank(&kg, PageRankConfig::default());
    let values: Vec<f64> = pr.values().cloned().collect();
    for v in &values[1..] {
        assert!(
            (values[0] - v).abs() < 1e-4,
            "Equal in complete graph: {} vs {}",
            values[0],
            v
        );
    }
}

#[test]
fn qa_algorithms_on_empty() {
    use lattix::algo::centrality::degree_centrality;
    use lattix::algo::pagerank::*;

    let empty = KnowledgeGraph::new();
    assert!(degree_centrality(&empty).is_empty());
    assert!(pagerank(&empty, PageRankConfig::default()).is_empty());
}

#[test]
fn qa_disconnected_components() {
    use lattix::algo::components::weakly_connected_components;

    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("A", "r", "B"));
    kg.add_triple(Triple::new("C", "r", "D"));
    kg.add_triple(Triple::new("E", "r", "F"));

    let wcc = weakly_connected_components(&kg);
    assert_eq!(wcc.len(), 3, "Should have 3 components, got {}", wcc.len());
}

// ── 4c. Format round-trips ──

#[test]
fn qa_ntriples_roundtrip() {
    use lattix::formats::NTriples;

    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new(
        "http://ex.org/Alice",
        "http://ex.org/knows",
        "http://ex.org/Bob",
    ));
    kg.add_triple(Triple::new(
        "http://ex.org/Bob",
        "http://ex.org/lives_in",
        "http://ex.org/Paris",
    ));

    let serialized = NTriples::to_string(&kg).unwrap();
    let recovered = NTriples::parse(&serialized).unwrap();
    assert_eq!(kg.triple_count(), recovered.triple_count());
}

#[test]
fn qa_per_triple_ntriples_roundtrip() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new(
        "http://ex.org/A",
        "http://ex.org/r",
        "http://ex.org/B",
    ));
    for triple in kg.triples() {
        let nt = triple.to_ntriples();
        let parsed = Triple::from_ntriples(&nt).unwrap();
        assert_eq!(triple.subject(), parsed.subject());
        assert_eq!(triple.predicate(), parsed.predicate());
        assert_eq!(triple.object(), parsed.object());
    }
}

/// Verify Triple::from_ntriples handles blank node subjects and literal objects.
#[test]
fn qa_from_ntriples_blank_node_and_literal() {
    let t = Triple::new("_:b0", "http://ex.org/r", "\"hello\"@en");
    let nt = t.to_ntriples();
    let parsed = Triple::from_ntriples(&nt).unwrap();
    assert_eq!(parsed.subject().as_str(), "_:b0");
    assert_eq!(parsed.predicate().as_str(), "http://ex.org/r");
    assert_eq!(parsed.object().as_str(), "\"hello\"@en");
}

#[test]
fn qa_json_serde_roundtrip() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("Alice", "knows", "Bob"));
    kg.add_triple(Triple::new("Bob", "works_at", "Acme"));

    let json = serde_json::to_string(&kg).unwrap();
    let recovered: KnowledgeGraph = serde_json::from_str(&json).unwrap();
    assert_eq!(kg.triple_count(), recovered.triple_count());
    assert_eq!(kg.entity_count(), recovered.entity_count());
    // Verify indexes rebuilt after deserialization
    assert_eq!(recovered.relations_from("Alice").len(), 1);
    assert_eq!(recovered.relations_to("Acme").len(), 1);
}

#[test]
fn qa_turtle_roundtrip() {
    use lattix::formats::Turtle;
    use std::collections::HashMap;

    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new(
        "http://ex.org/A",
        "http://ex.org/knows",
        "http://ex.org/B",
    ));

    let prefixes: HashMap<String, String> = HashMap::new();
    let mut buf = Vec::new();
    Turtle::write(&kg, &mut buf, &prefixes).unwrap();
    let recovered = Turtle::read(std::io::Cursor::new(&buf), None).unwrap();
    assert_eq!(kg.triple_count(), recovered.triple_count());
}

// ── 4d. HeteroGraph ──

#[test]
fn qa_heterograph_from_knowledge_graph() {
    use lattix::hetero::HeteroGraph;

    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("Alice", "knows", "Bob"));
    kg.add_triple(Triple::new("Bob", "works_at", "Acme"));

    // Test the new convenience method
    let hg = HeteroGraph::from_knowledge_graph(&kg);
    assert_eq!(hg.num_edge_types(), 2); // knows, works_at

    // Also test From trait directly
    let hg2: HeteroGraph = (&kg).into();
    assert_eq!(hg.num_edge_types(), hg2.num_edge_types());

    // Round-trip back to KG
    let kg2 = hg.to_knowledge_graph();
    assert_eq!(kg2.triple_count(), 2);
}

#[test]
fn qa_heterograph_typed_ops() {
    use lattix::hetero::*;

    let mut hg = HeteroGraph::new();
    let user = NodeType::new("user");
    let item = NodeType::new("item");
    let edge = EdgeType::new(user.clone(), "bought", item.clone());

    hg.add_node(user.clone(), "alice");
    hg.add_node(user.clone(), "bob");
    hg.add_node(item.clone(), "book1");
    hg.add_edge(&edge, "alice", "book1");

    assert_eq!(hg.num_node_types(), 2);
    assert_eq!(hg.num_edge_types(), 1);
    assert_eq!(hg.num_nodes(&user), 2);
    assert_eq!(hg.num_nodes(&item), 1);
}

// ── 4d. HyperGraph ──

#[test]
fn qa_hypertriple_with_qualifiers() {
    use lattix::hyper::*;
    use lattix::RelationType;

    let ht = HyperTriple::from_parts("Einstein", "won", "Nobel Prize")
        .with_qualifier("year", "1921")
        .with_qualifier("field", "Physics");

    assert_eq!(ht.arity(), 4); // subject + object + 2 qualifiers
    assert_eq!(ht.qualifiers.len(), 2);
    assert_eq!(ht.qualifiers[&RelationType::new("year")].as_str(), "1921");
}

#[test]
fn qa_hyperedge_reification() {
    use lattix::hyper::*;

    let he = HyperEdge::new("marriage")
        .with_binding("spouse1", "Alice")
        .with_binding("spouse2", "Bob")
        .with_binding("year", "2020");

    assert_eq!(he.arity(), 3);
    assert_eq!(he.entity_by_role("spouse1").unwrap().as_str(), "Alice");

    let reified = he.reify("_:marriage1");
    // Should produce 1 rdf:type triple + 3 binding triples = 4
    assert_eq!(reified.len(), 4);
}

// ── 4e. Edge cases ──

#[test]
fn qa_empty_string_entities() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("", "r", ""));
    // Permissive: empty strings accepted
    assert_eq!(kg.entity_count(), 1); // "" is one entity used as both subject and object
    assert_eq!(kg.triple_count(), 1);
}

#[test]
fn qa_self_loops() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("A", "r", "A"));
    assert_eq!(kg.entity_count(), 1);
    assert!(kg.has_edge("A", "A"));
}

#[test]
fn qa_duplicate_triples_stored() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("A", "r", "B"));
    kg.add_triple(Triple::new("A", "r", "B"));
    // By design: duplicates stored, not deduplicated
    assert_eq!(kg.triple_count(), 2);
}

#[test]
fn qa_unicode_entities() {
    let t = Triple::new("東京", "located_in", "日本");
    let nt = t.to_ntriples();
    let recovered = Triple::from_ntriples(&nt).unwrap();
    assert_eq!(recovered.subject().as_str(), "東京");
    assert_eq!(recovered.object().as_str(), "日本");
}

#[test]
fn qa_very_long_entity_names() {
    let long = "A".repeat(2000);
    let t = Triple::new(long.as_str(), "r", "B");
    assert_eq!(t.subject().as_str(), long);

    let mut kg = KnowledgeGraph::new();
    kg.add_triple(t);
    assert_eq!(kg.entity_count(), 2);
}

#[test]
fn qa_large_graph() {
    let mut kg = KnowledgeGraph::new();
    for i in 0..10_000 {
        kg.add_triple(Triple::new(
            format!("N{}", i % 100),
            "r",
            format!("N{}", (i + 1) % 100),
        ));
    }
    assert_eq!(kg.triple_count(), 10_000);
    assert_eq!(kg.entity_count(), 100);
}

#[test]
fn qa_remove_nonexistent() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("A", "r", "B"));
    let removed = kg.remove_triple(&"X".into(), &"y".into(), &"Z".into());
    assert!(!removed);
    assert_eq!(kg.triple_count(), 1);
}

// ── 4f. GraphDocument / exchange ──

#[test]
fn qa_graph_document_roundtrip() {
    use lattix::exchange::*;

    let doc = GraphDocument {
        nodes: vec![
            GraphNode::new("A", "Person", "Alice"),
            GraphNode::new("B", "Person", "Bob"),
        ],
        edges: vec![GraphEdge::new("A", "B", "knows")],
        metadata: Default::default(),
    };

    let json = serde_json::to_string(&doc).unwrap();
    let recovered: GraphDocument = serde_json::from_str(&json).unwrap();
    assert_eq!(recovered.nodes.len(), 2);
    assert_eq!(recovered.edges.len(), 1);

    let kg = recovered.to_knowledge_graph();
    assert_eq!(kg.triple_count(), 1);
}

// ── 4g. TripleQuery builder ──

#[test]
fn qa_triple_query_builder() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("A", "knows", "B"));
    kg.add_triple(Triple::new("A", "knows", "C"));
    kg.add_triple(Triple::new("A", "works_at", "X"));
    kg.add_triple(Triple::new("B", "knows", "C"));

    // Subject filter
    assert_eq!(kg.query().subject("A").count(), 3);

    // Predicate filter
    assert_eq!(kg.query().predicate("knows").count(), 3);

    // All three filters
    assert_eq!(
        kg.query()
            .subject("A")
            .predicate("knows")
            .object("B")
            .count(),
        1
    );

    // exists()
    assert!(kg.query().subject("A").predicate("knows").exists());
    assert!(!kg.query().subject("Z").exists());
}

// ── 4h. BufWriter fix verification ──

#[test]
fn qa_ntriples_write_bufwriter() {
    use lattix::formats::NTriples;

    // Verify NTriples::write still works correctly after BufWriter wrapping
    let mut kg = KnowledgeGraph::new();
    for i in 0..100 {
        kg.add_triple(Triple::new(
            format!("http://ex.org/N{i}"),
            "http://ex.org/r",
            format!("http://ex.org/N{}", i + 1),
        ));
    }

    let serialized = NTriples::to_string(&kg).unwrap();
    let recovered = NTriples::parse(&serialized).unwrap();
    assert_eq!(kg.triple_count(), recovered.triple_count());

    // Also verify writing to a Vec<u8> writer
    let mut buf: Vec<u8> = Vec::new();
    NTriples::write(&kg, &mut buf).unwrap();
    let output = String::from_utf8(buf).unwrap();
    let lines: Vec<&str> = output.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 100);
}

/// Katz alpha validation: alpha >= 1.0 should panic.
#[test]
#[should_panic(expected = "Katz alpha must be in (0, 1)")]
fn qa_katz_alpha_validation() {
    use lattix::algo::centrality::{katz_centrality, KatzConfig};
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("A", "r", "B"));
    let _ = katz_centrality(
        &kg,
        KatzConfig {
            alpha: 1.5,
            ..Default::default()
        },
    );
}
