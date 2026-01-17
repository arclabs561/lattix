//! Random Walks Demo (Node2Vec style)
//!
//! Demonstrates generating random walks for graph embedding algorithms.
//!
//! ```bash
//! cargo run --example random_walks --features algo
//! ```

use lattix_core::algo::random_walk::{generate_walks, RandomWalkConfig};
use lattix_core::KnowledgeGraph;
use lattix_core::Triple;

fn main() {
    println!("Random Walks Demo (Node2Vec Style)");
    println!("==================================\n");

    // Build a knowledge graph about movie relationships
    let triples = vec![
        Triple::new("Spielberg", "directed", "Jurassic_Park"),
        Triple::new("Spielberg", "directed", "E.T."),
        Triple::new("Spielberg", "directed", "Schindlers_List"),
        Triple::new("Nolan", "directed", "Inception"),
        Triple::new("Nolan", "directed", "Interstellar"),
        Triple::new("Nolan", "directed", "Dunkirk"),
        Triple::new("DiCaprio", "acted_in", "Inception"),
        Triple::new("DiCaprio", "acted_in", "Shutter_Island"),
        Triple::new("DiCaprio", "acted_in", "Revenant"),
        Triple::new("Hanks", "acted_in", "Schindlers_List"),
        Triple::new("Hanks", "acted_in", "Saving_Private_Ryan"),
        Triple::new("Spielberg", "directed", "Saving_Private_Ryan"),
        Triple::new("Goldblum", "acted_in", "Jurassic_Park"),
        Triple::new("Goldblum", "acted_in", "Independence_Day"),
        Triple::new("Williams", "composed", "Jurassic_Park"),
        Triple::new("Williams", "composed", "E.T."),
        Triple::new("Williams", "composed", "Schindlers_List"),
        Triple::new("Zimmer", "composed", "Inception"),
        Triple::new("Zimmer", "composed", "Interstellar"),
        Triple::new("Zimmer", "composed", "Dunkirk"),
    ];

    let mut kg = KnowledgeGraph::new();
    for triple in triples {
        kg.add_triple(triple);
    }
    let entity_count = kg.entities().count();
    println!("Graph: {} entities", entity_count);

    // Generate random walks
    let config = RandomWalkConfig {
        walk_length: 10,
        num_walks: 3,
        p: 1.0, // Return parameter (Node2Vec)
        q: 1.0, // In-out parameter (Node2Vec)
        seed: 42,
    };

    println!("\nConfiguration:");
    println!("  Walk length: {}", config.walk_length);
    println!("  Walks per node: {}", config.num_walks);
    println!("  p (return): {:.1}", config.p);
    println!("  q (in-out): {:.1}", config.q);

    let walks = generate_walks(&kg, config);

    println!("\nGenerated {} walks", walks.len());
    println!("\nSample walks:");
    for (i, walk) in walks.iter().take(5).enumerate() {
        println!("  Walk {}: {}", i + 1, walk.join(" -> "));
    }

    println!("\nNode2Vec Parameters Explained:");
    println!("  p (return parameter):");
    println!("    p < 1: Encourages returning to visited nodes (BFS-like)");
    println!("    p > 1: Discourages returning (DFS-like)");
    println!("  q (in-out parameter):");
    println!("    q < 1: Encourages exploring outward (global structure)");
    println!("    q > 1: Encourages staying local (community structure)");

    println!("\nThese walks can be used with Word2Vec/Skip-gram to learn");
    println!("node embeddings that capture graph structure.");

    println!("\nDone!");
}
