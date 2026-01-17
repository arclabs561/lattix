//! Integration tests for centrality algorithms.
//!
//! These tests verify that centrality algorithms work correctly on
//! realistic graph structures and produce consistent, mathematically
//! sound results.

use lattix_core::algo::centrality::{
    betweenness_centrality, closeness_centrality, degree_centrality, eigenvector_centrality, hits,
    katz_centrality, BetweennessConfig, ClosenessConfig, EigenvectorConfig, HitsConfig, KatzConfig,
};
use lattix_core::algo::pagerank::{pagerank, PageRankConfig};
use lattix_core::{KnowledgeGraph, Triple};

/// Create a small social network for testing.
///
/// Structure:
/// ```text
///        Alice
///       /  |  \
///      v   v   v
///    Bob  Carol Dave
///      \   |   /
///       v  v  v
///         Eve
/// ```
fn social_network() -> KnowledgeGraph {
    let mut kg = KnowledgeGraph::new();
    // Alice connects to everyone
    kg.add_triple(Triple::new("Alice", "friends", "Bob"));
    kg.add_triple(Triple::new("Alice", "friends", "Carol"));
    kg.add_triple(Triple::new("Alice", "friends", "Dave"));
    // Bob, Carol, Dave connect to Eve
    kg.add_triple(Triple::new("Bob", "friends", "Eve"));
    kg.add_triple(Triple::new("Carol", "friends", "Eve"));
    kg.add_triple(Triple::new("Dave", "friends", "Eve"));
    kg
}

/// Create a citation network with clear hubs and authorities.
///
/// Structure: Survey papers cite primary papers.
/// ```text
/// Survey1 ---> Paper1
/// Survey1 ---> Paper2
/// Survey2 ---> Paper1
/// Survey2 ---> Paper2
/// Survey2 ---> Paper3
/// ```
fn citation_network() -> KnowledgeGraph {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("Survey1", "cites", "Paper1"));
    kg.add_triple(Triple::new("Survey1", "cites", "Paper2"));
    kg.add_triple(Triple::new("Survey2", "cites", "Paper1"));
    kg.add_triple(Triple::new("Survey2", "cites", "Paper2"));
    kg.add_triple(Triple::new("Survey2", "cites", "Paper3"));
    kg
}

/// Create a long chain graph.
fn chain_graph(length: usize) -> KnowledgeGraph {
    let mut kg = KnowledgeGraph::new();
    for i in 0..length {
        let from = format!("N{i}");
        let to = format!("N{}", i + 1);
        kg.add_triple(Triple::new(from, "next", to));
    }
    kg
}

/// Create a complete graph where everyone connects to everyone.
fn complete_graph(n: usize) -> KnowledgeGraph {
    let mut kg = KnowledgeGraph::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let from = format!("N{i}");
                let to = format!("N{j}");
                kg.add_triple(Triple::new(from, "connected", to));
            }
        }
    }
    kg
}

// ============================================================================
// Degree Centrality Tests
// ============================================================================

#[test]
fn test_degree_social_network() {
    let kg = social_network();
    let degrees = degree_centrality(&kg);

    // Alice has highest out-degree (3 outgoing)
    let alice = degrees.get("Alice").unwrap();
    assert_eq!(alice.out_degree, 3);
    assert_eq!(alice.in_degree, 0);

    // Eve has highest in-degree (3 incoming)
    let eve = degrees.get("Eve").unwrap();
    assert_eq!(eve.in_degree, 3);
    assert_eq!(eve.out_degree, 0);
}

#[test]
fn test_degree_complete_graph() {
    let kg = complete_graph(4);
    let degrees = degree_centrality(&kg);

    // Everyone should have n-1 in and n-1 out
    for (name, deg) in degrees {
        assert_eq!(deg.in_degree, 3, "{name} in-degree");
        assert_eq!(deg.out_degree, 3, "{name} out-degree");
        // Normalized should be 1.0 (connected to all n-1 others)
        assert!((deg.in_normalized - 1.0).abs() < 1e-6);
    }
}

// ============================================================================
// Betweenness Centrality Tests
// ============================================================================

#[test]
fn test_betweenness_chain() {
    // In a chain A -> B -> C -> D -> E, middle nodes should have highest betweenness
    let kg = chain_graph(4); // N0 -> N1 -> N2 -> N3 -> N4

    let config = BetweennessConfig {
        normalized: false,
        undirected: false,
    };
    let scores = betweenness_centrality(&kg, config);

    // Endpoints have 0 betweenness
    assert_eq!(*scores.get("N0").unwrap(), 0.0);
    assert_eq!(*scores.get("N4").unwrap(), 0.0);

    // Middle nodes have positive betweenness
    let n1 = *scores.get("N1").unwrap();
    let n2 = *scores.get("N2").unwrap();
    let n3 = *scores.get("N3").unwrap();

    // N2 should have highest (most paths go through it)
    assert!(n2 >= n1, "N2={n2} should be >= N1={n1}");
    assert!(n2 >= n3, "N2={n2} should be >= N3={n3}");
}

#[test]
fn test_betweenness_social_network() {
    let kg = social_network();

    let config = BetweennessConfig::default();
    let scores = betweenness_centrality(&kg, config);

    // Bob, Carol, Dave are bridges between Alice and Eve
    let bob = *scores.get("Bob").unwrap();
    let alice = *scores.get("Alice").unwrap();
    let _eve = *scores.get("Eve").unwrap();

    // Alice and Eve are sources/sinks, should have lower betweenness
    assert!(bob >= alice, "Bob={bob} should be >= Alice={alice}");
}

// ============================================================================
// Closeness Centrality Tests
// ============================================================================

#[test]
fn test_closeness_star() {
    let mut kg = KnowledgeGraph::new();
    // Star with hub in center (bidirectional)
    for leaf in ["A", "B", "C", "D"] {
        kg.add_triple(Triple::new("Hub", "rel", leaf));
        kg.add_triple(Triple::new(leaf, "rel", "Hub"));
    }

    let config = ClosenessConfig::default();
    let scores = closeness_centrality(&kg, config);

    let hub = *scores.get("Hub").unwrap();
    let leaf = *scores.get("A").unwrap();

    // Hub reaches everyone in 1 hop, leaves need 2 hops to reach each other
    assert!(
        hub > leaf,
        "Hub={hub} should be more central than leaf={leaf}"
    );
}

#[test]
fn test_closeness_complete_graph() {
    let kg = complete_graph(4);

    let config = ClosenessConfig {
        normalized: true,
        undirected: false,
        harmonic: true,
    };
    let scores = closeness_centrality(&kg, config);

    // In complete graph, everyone has same closeness (all distances = 1)
    let values: Vec<_> = scores.values().cloned().collect();
    let first = values[0];
    for v in &values {
        assert!(
            (v - first).abs() < 1e-6,
            "All should be equal in complete graph"
        );
    }
}

// ============================================================================
// Eigenvector Centrality Tests
// ============================================================================

#[test]
fn test_eigenvector_convergence() {
    let kg = complete_graph(5);

    let config = EigenvectorConfig::default();
    let scores = eigenvector_centrality(&kg, config);

    // All nodes should have equal centrality in complete graph
    let values: Vec<_> = scores.values().cloned().collect();
    let first = values[0];
    for v in &values {
        assert!(
            (v - first).abs() < 0.01,
            "Complete graph should have equal eigenvector centrality"
        );
    }
}

#[test]
fn test_eigenvector_normalized() {
    let kg = social_network();

    let scores = eigenvector_centrality(&kg, EigenvectorConfig::default());

    // L2 norm should be 1
    let norm: f64 = scores.values().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "Should be L2 normalized: {norm}");
}

// ============================================================================
// Katz Centrality Tests
// ============================================================================

#[test]
fn test_katz_baseline() {
    // Katz with β > 0 ensures all nodes have positive score
    let kg = social_network();

    let config = KatzConfig::default();
    let scores = katz_centrality(&kg, config);

    for (name, score) in &scores {
        assert!(*score > 0.0, "{name} should have positive Katz score");
    }
}

#[test]
fn test_katz_chain_ordering() {
    // In a chain, nodes further along should have higher Katz (more paths reach them)
    let kg = chain_graph(3); // N0 -> N1 -> N2 -> N3

    let config = KatzConfig {
        normalized: false,
        ..Default::default()
    };
    let scores = katz_centrality(&kg, config);

    let n0 = *scores.get("N0").unwrap();
    let n1 = *scores.get("N1").unwrap();
    let n2 = *scores.get("N2").unwrap();
    let n3 = *scores.get("N3").unwrap();

    // More paths reach nodes further along
    assert!(n3 >= n2, "N3={n3} >= N2={n2}");
    assert!(n2 >= n1, "N2={n2} >= N1={n1}");
    assert!(n1 >= n0, "N1={n1} >= N0={n0}");
}

// ============================================================================
// PageRank Tests
// ============================================================================

#[test]
fn test_pagerank_sums_to_one() {
    let kg = social_network();

    let scores = pagerank(&kg, PageRankConfig::default());
    let total: f64 = scores.values().sum();

    assert!(
        (total - 1.0).abs() < 1e-4,
        "PageRank should sum to 1: {total}"
    );
}

#[test]
fn test_pagerank_dangling_nodes() {
    // Eve is a dangling node (no outgoing edges)
    let kg = social_network();

    let scores = pagerank(&kg, PageRankConfig::default());

    // Eve should still have positive PageRank (receives from Bob, Carol, Dave)
    let eve = *scores.get("Eve").unwrap();
    assert!(
        eve > 0.0,
        "Dangling node Eve should have positive PR: {eve}"
    );

    // Eve should have highest PageRank (all paths lead to Eve)
    let alice = *scores.get("Alice").unwrap();
    assert!(eve > alice, "Eve={eve} should be > Alice={alice}");
}

// ============================================================================
// HITS Tests
// ============================================================================

#[test]
fn test_hits_citation_network() {
    let kg = citation_network();

    let scores = hits(&kg, HitsConfig::default());

    // Surveys should be hubs (they cite papers)
    let survey1 = scores.get("Survey1").unwrap();
    let survey2 = scores.get("Survey2").unwrap();

    // Papers should be authorities (they are cited)
    let paper1 = scores.get("Paper1").unwrap();

    assert!(
        survey1.hub > survey1.authority,
        "Survey1 should be more hub than authority"
    );
    assert!(
        paper1.authority > paper1.hub,
        "Paper1 should be more authority than hub"
    );

    // Survey2 cites more papers, should have higher hub score
    assert!(
        survey2.hub > survey1.hub,
        "Survey2={} should be better hub than Survey1={}",
        survey2.hub,
        survey1.hub
    );
}

#[test]
fn test_hits_normalized() {
    let kg = citation_network();

    let scores = hits(
        &kg,
        HitsConfig {
            normalized: true,
            ..Default::default()
        },
    );

    let hub_sum: f64 = scores.values().map(|s| s.hub).sum();
    let auth_sum: f64 = scores.values().map(|s| s.authority).sum();

    assert!(
        (hub_sum - 1.0).abs() < 1e-4,
        "Hub sum should be 1: {hub_sum}"
    );
    assert!(
        (auth_sum - 1.0).abs() < 1e-4,
        "Auth sum should be 1: {auth_sum}"
    );
}

// ============================================================================
// Cross-Algorithm Consistency Tests
// ============================================================================

#[test]
fn test_algorithms_agree_on_obvious_cases() {
    // In a star graph with bidirectional edges, the hub should be most central
    // by ALL measures
    let mut kg = KnowledgeGraph::new();
    for leaf in ["A", "B", "C", "D", "E"] {
        kg.add_triple(Triple::new("Hub", "rel", leaf));
        kg.add_triple(Triple::new(leaf, "rel", "Hub"));
    }

    let degrees = degree_centrality(&kg);
    let closeness = closeness_centrality(&kg, ClosenessConfig::default());
    let eigenvector = eigenvector_centrality(&kg, EigenvectorConfig::default());
    let katz = katz_centrality(&kg, KatzConfig::default());
    let pr = pagerank(&kg, PageRankConfig::default());

    let hub_degree = degrees.get("Hub").unwrap().total();
    let leaf_degree = degrees.get("A").unwrap().total();
    assert!(hub_degree > leaf_degree, "Hub should have higher degree");

    let hub_close = *closeness.get("Hub").unwrap();
    let leaf_close = *closeness.get("A").unwrap();
    assert!(hub_close > leaf_close, "Hub should have higher closeness");

    let hub_eigen = *eigenvector.get("Hub").unwrap();
    let leaf_eigen = *eigenvector.get("A").unwrap();
    assert!(hub_eigen > leaf_eigen, "Hub should have higher eigenvector");

    let hub_katz = *katz.get("Hub").unwrap();
    let leaf_katz = *katz.get("A").unwrap();
    assert!(hub_katz >= leaf_katz, "Hub should have higher Katz");

    let hub_pr = *pr.get("Hub").unwrap();
    let leaf_pr = *pr.get("A").unwrap();
    assert!(hub_pr > leaf_pr, "Hub should have higher PageRank");
}

#[test]
fn test_empty_graph() {
    let kg = KnowledgeGraph::new();

    assert!(degree_centrality(&kg).is_empty());
    assert!(betweenness_centrality(&kg, BetweennessConfig::default()).is_empty());
    assert!(closeness_centrality(&kg, ClosenessConfig::default()).is_empty());
    assert!(eigenvector_centrality(&kg, EigenvectorConfig::default()).is_empty());
    assert!(katz_centrality(&kg, KatzConfig::default()).is_empty());
    assert!(pagerank(&kg, PageRankConfig::default()).is_empty());
    assert!(hits(&kg, HitsConfig::default()).is_empty());
}

#[test]
fn test_single_node() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("A", "rel", "A")); // self-loop creates entity

    let degrees = degree_centrality(&kg);
    assert_eq!(degrees.len(), 1);
}
