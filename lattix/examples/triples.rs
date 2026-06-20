//! Basic triple graph queries.
//!
//! Run:
//! `cargo run --example triples`

use lattix::{KnowledgeGraph, Triple};

fn main() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));
    kg.add_triple(Triple::new("Apple", "headquartered_in", "Cupertino"));
    kg.add_triple(Triple::new("Steve Jobs", "born_in", "San Francisco"));
    kg.add_triple(Triple::new("Cupertino", "located_in", "California"));
    kg.add_triple(Triple::new("San Francisco", "located_in", "California"));

    println!("entities: {}", kg.entity_count());
    println!("triples:  {}", kg.triple_count());

    println!("\nrelations from Apple:");
    for triple in kg.relations_from("Apple") {
        println!(
            "  {} --{}--> {}",
            triple.subject(),
            triple.predicate(),
            triple.object()
        );
    }

    println!("\nentities connected to California:");
    for triple in kg.relations_to("California") {
        println!("  {} via {}", triple.subject(), triple.predicate());
    }

    let path = kg
        .find_path("Apple", "California")
        .expect("Apple should reach California");
    println!("\npath Apple -> California ({} hops):", path.len());
    for triple in &path {
        println!(
            "  {} --{}--> {}",
            triple.subject(),
            triple.predicate(),
            triple.object()
        );
    }

    assert_eq!(kg.relations_from("Apple").len(), 2);
    assert_eq!(kg.relations_to("California").len(), 2);
    assert_eq!(path.len(), 2);
}
