//! PageRank Demo
//!
//! Demonstrates computing PageRank centrality on a knowledge graph.
//!
//! ```bash
//! cargo run --example pagerank_demo --features algo
//! ```

use lattix::algo::pagerank::{pagerank, PageRankConfig};
use lattix::KnowledgeGraph;
use lattix::Triple;

fn main() {
    println!("PageRank Demo");
    println!("=============\n");

    // Build a small knowledge graph about scientists
    let triples = vec![
        Triple::new("Einstein", "influenced", "Feynman"),
        Triple::new("Einstein", "influenced", "Bohr"),
        Triple::new("Bohr", "influenced", "Heisenberg"),
        Triple::new("Bohr", "influenced", "Pauli"),
        Triple::new("Feynman", "influenced", "Weinberg"),
        Triple::new("Heisenberg", "influenced", "Pauli"),
        Triple::new("Pauli", "influenced", "Feynman"),
        Triple::new("Newton", "influenced", "Einstein"),
        Triple::new("Maxwell", "influenced", "Einstein"),
        Triple::new("Planck", "influenced", "Einstein"),
        Triple::new("Planck", "influenced", "Bohr"),
    ];

    println!("Knowledge Graph: Scientific influences");
    println!("Triples:");
    for t in &triples {
        println!("  ({}, {}, {})", t.subject, t.predicate, t.object);
    }

    let mut kg = KnowledgeGraph::new();
    for triple in triples {
        kg.add_triple(triple);
    }
    let entity_count = kg.entities().count();
    println!("\nGraph: {} entities", entity_count);

    // Compute PageRank
    let config = PageRankConfig {
        damping_factor: 0.85,
        max_iterations: 50,
        tolerance: 1e-6,
    };

    let scores = pagerank(&kg, config);

    // Sort by PageRank score
    let mut sorted_scores: Vec<_> = scores.iter().collect();
    sorted_scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("\nPageRank Results:");
    println!("Scientist        | Score");
    println!("-----------------|--------");
    for (name, score) in sorted_scores.iter().take(10) {
        println!("{:16} | {:.4}", name, score);
    }

    println!("\nInterpretation:");
    println!(
        "- Weinberg/Feynman have highest PageRank: they're sink nodes (receive but don't give)"
    );
    println!("- Einstein is mid-rank: receives from Newton/Maxwell/Planck, gives to Feynman/Bohr");
    println!("- Newton/Maxwell/Planck have low scores: they're source nodes (no incoming links)");

    println!("\nDone!");
}
