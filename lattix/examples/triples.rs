//! Basic triple and knowledge graph operations.

use lattix::{KnowledgeGraph, Triple};

fn main() {
    let mut kg = KnowledgeGraph::new();

    // Build a small knowledge graph about programming languages.
    let triples = [
        ("Rust", "designed_by", "Graydon Hoare"),
        ("Rust", "paradigm", "Systems"),
        ("Rust", "influenced_by", "ML"),
        ("Rust", "influenced_by", "C++"),
        ("OCaml", "paradigm", "Functional"),
        ("OCaml", "influenced_by", "ML"),
        ("Haskell", "paradigm", "Functional"),
        ("Haskell", "influenced_by", "ML"),
        ("C++", "paradigm", "Systems"),
        ("C++", "designed_by", "Bjarne Stroustrup"),
    ];
    for (s, p, o) in &triples {
        kg.add_triple(Triple::new(*s, *p, *o));
    }

    let stats = kg.stats();
    println!("=== Knowledge Graph Stats ===");
    println!("  entities: {}", stats.entity_count);
    println!("  triples:  {}", stats.triple_count);

    // Query: all relations from "Rust"
    println!("\n=== Relations from 'Rust' ===");
    for t in kg.relations_from("Rust") {
        println!("  {} --[{}]--> {}", t.subject, t.predicate, t.object);
    }

    // Query: what is influenced by "ML"?
    println!("\n=== Influenced by 'ML' ===");
    for t in kg.relations_to("ML") {
        println!("  {} --[{}]--> ML", t.subject, t.predicate);
    }

    // Query: all "paradigm" relations
    println!("\n=== All 'paradigm' relations ===");
    for t in kg.triples_with_relation("paradigm") {
        println!("  {} => {}", t.subject, t.object);
    }

    // Find path: Rust -> ML (should exist via influenced_by)
    if let Some(path) = kg.find_path("Rust", "ML") {
        println!("\n=== Path: Rust -> ML ===");
        println!("  hops: {}", path.len());
        for t in &path {
            println!("  {} --[{}]--> {}", t.subject, t.predicate, t.object);
        }
    }
}
