//! HyperGraph Demo
//!
//! Demonstrates n-ary relations using HyperTriple, HyperEdge, and HyperGraph.
//!
//! ```bash
//! cargo run --example hypergraph
//! ```

use lattix::{HyperEdge, HyperGraph, HyperTriple};

fn main() {
    println!("HyperGraph Demo");
    println!("===============\n");

    // ── 1. HyperTriples: triples with qualifiers ────────────────────────
    //
    // A HyperTriple is a standard (subject, predicate, object) triple
    // augmented with qualifier key-value pairs -- the Wikidata model.

    println!("1. HyperTriples with qualifiers");
    println!("-------------------------------\n");

    let einstein_nobel = HyperTriple::from_parts("Einstein", "won", "Nobel Prize")
        .with_qualifier("year", "1921")
        .with_qualifier("field", "Physics");

    let curie_nobel = HyperTriple::from_parts("Curie", "won", "Nobel Prize")
        .with_qualifier("year", "1911")
        .with_qualifier("field", "Chemistry");

    let bohr_nobel = HyperTriple::from_parts("Bohr", "won", "Nobel Prize")
        .with_qualifier("year", "1922")
        .with_qualifier("field", "Physics");

    println!(
        "  {} won {} (year={}, field={})",
        einstein_nobel.core.subject(),
        einstein_nobel.core.object(),
        einstein_nobel.qualifier("year").unwrap(),
        einstein_nobel.qualifier("field").unwrap(),
    );
    println!(
        "  {} won {} (year={}, field={})",
        curie_nobel.core.subject(),
        curie_nobel.core.object(),
        curie_nobel.qualifier("year").unwrap(),
        curie_nobel.qualifier("field").unwrap(),
    );
    println!(
        "  {} won {} (year={}, field={})",
        bohr_nobel.core.subject(),
        bohr_nobel.core.object(),
        bohr_nobel.qualifier("year").unwrap(),
        bohr_nobel.qualifier("field").unwrap(),
    );
    println!(
        "  Arity of each: {} (subject + object + 2 qualifiers)\n",
        einstein_nobel.arity()
    );

    // ── 2. HyperEdges: true n-ary relations with roles ──────────────────
    //
    // A HyperEdge connects multiple entities through explicit semantic roles.
    // Unlike triples, there is no fixed subject/object -- every participant
    // has a named role.

    println!("2. HyperEdges with role bindings");
    println!("--------------------------------\n");

    let transaction = HyperEdge::new("transaction")
        .with_binding("buyer", "Alice")
        .with_binding("seller", "Bob")
        .with_binding("item", "Book")
        .with_binding("price", "$20")
        .with_confidence(0.95);

    println!("  Relation: {}", transaction.relation);
    for binding in &transaction.bindings {
        println!("    {}: {}", binding.role, binding.entity);
    }
    println!("  Confidence: {:.2}", transaction.confidence.unwrap());
    println!("  Arity: {}", transaction.arity());
    println!(
        "  Buyer: {}\n",
        transaction.entity_by_role("buyer").unwrap()
    );

    // ── 3. HyperGraph: querying mixed facts ─────────────────────────────

    println!("3. HyperGraph queries");
    println!("---------------------\n");

    let mut hg = HyperGraph::new();
    hg.add_hyper_triple(einstein_nobel);
    hg.add_hyper_triple(curie_nobel);
    hg.add_hyper_triple(bohr_nobel);
    hg.add_hyperedge(transaction);

    println!("  Facts: {} total", hg.fact_count());
    println!("  Entities: {}\n", hg.entities().len());

    // Find by entity: all hyper-triples mentioning "Nobel Prize"
    let nobel_facts = hg.find_by_entity("Nobel Prize");
    println!(
        "  find_by_entity(\"Nobel Prize\"): {} results",
        nobel_facts.len()
    );
    for ht in &nobel_facts {
        println!(
            "    {} {} {}",
            ht.core.subject(),
            ht.core.predicate(),
            ht.core.object()
        );
    }

    // Find by qualifier: physics laureates
    let physics = hg.find_by_qualifier("field", "Physics");
    println!(
        "\n  find_by_qualifier(\"field\", \"Physics\"): {} results",
        physics.len()
    );
    for ht in &physics {
        println!(
            "    {} ({})",
            ht.core.subject(),
            ht.qualifier("year").unwrap()
        );
    }

    // ── 4. KnowledgeGraph interop ───────────────────────────────────────
    //
    // to_knowledge_graph() flattens all hyper-triples and hyperedges into
    // standard triples via reification: each hyper-triple becomes its core
    // triple plus qualifier triples, and each hyperedge becomes an rdf:type
    // triple plus one triple per role binding.

    println!("\n4. KnowledgeGraph interop");
    println!("------------------------\n");

    let kg = hg.to_knowledge_graph();
    println!(
        "  Converted to KnowledgeGraph: {} triples",
        kg.triple_count()
    );
    println!("  Reified triples:");
    for triple in kg.triples() {
        println!(
            "    ({}, {}, {})",
            triple.subject(),
            triple.predicate(),
            triple.object()
        );
    }

    println!("\nDone!");
}
