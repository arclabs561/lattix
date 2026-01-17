use lattix_core::{KnowledgeGraph, Triple};
use lattix_reason::{LogicalQuery, Reasoner, SparseReasoner, SymbolicReasoner};
use std::collections::HashSet;

fn setup_sample_kg() -> KnowledgeGraph {
    let mut kg = KnowledgeGraph::new();
    // Path 1: Einstein -> born_in -> Ulm -> located_in -> Germany
    kg.add_triple(Triple::new("Einstein", "born_in", "Ulm"));
    kg.add_triple(Triple::new("Ulm", "located_in", "Germany"));
    
    // Path 2: Einstein -> won -> Nobel_Prize
    kg.add_triple(Triple::new("Einstein", "won", "Nobel_Prize"));
    
    // Path 3: Planck -> born_in -> Kiel -> located_in -> Germany
    kg.add_triple(Triple::new("Planck", "born_in", "Kiel"));
    kg.add_triple(Triple::new("Kiel", "located_in", "Germany"));
    
    // Path 4: Planck -> won -> Nobel_Prize
    kg.add_triple(Triple::new("Planck", "won", "Nobel_Prize"));

    // Path 5: Hawking -> won -> Copley_Medal (not Nobel)
    kg.add_triple(Triple::new("Hawking", "won", "Copley_Medal"));
    
    kg
}

#[test]
fn test_dnf_conversion() {
    // (A OR B) AND C -> (A AND C) OR (B AND C)
    let query = LogicalQuery::and(vec![
        LogicalQuery::or(vec![
            LogicalQuery::entity("A"),
            LogicalQuery::entity("B"),
        ]),
        LogicalQuery::entity("C"),
    ]);

    let dnf = query.to_dnf();
    assert_eq!(dnf.len(), 2); // Two conjunctions
    
    // Check first conjunction: (A AND C)
    let c1 = &dnf[0];
    assert_eq!(c1.len(), 2);
    
    // Check second conjunction: (B AND C)
    let c2 = &dnf[1];
    assert_eq!(c2.len(), 2);
}

#[test]
fn test_sparse_vs_symbolic_reasoner() {
    let kg = setup_sample_kg();
    let sparse = SparseReasoner::from_kg(&kg);
    let symbolic = SymbolicReasoner::new(&kg);

    // Query: "Nobel Prize winners born in Germany"
    // Wait, we need reverse relations for "born in Germany" if we start from Germany.
    // Let's add them or use forward query: "Someone who won Nobel AND was born in Ulm"
    
    let query = LogicalQuery::and(vec![
        LogicalQuery::entity("Einstein").project("won"), // This results in {Nobel_Prize}
        LogicalQuery::entity("Einstein").project("born_in"), // This results in {Ulm}
    ]);
    // The intersection of {Nobel_Prize} and {Ulm} is empty.
    
    // Correct query: "Who was born in Ulm AND won Nobel Prize?"
    // (e, r) queries usually return tails.
    // So we need: query that returns entities.
    // Let's use reverse relations for structural reasoning.
    
    let mut kg_rev = kg.clone();
    kg_rev.add_triple(Triple::new("Ulm", "rev_born_in", "Einstein"));
    kg_rev.add_triple(Triple::new("Nobel_Prize", "rev_won", "Einstein"));
    
    let query = LogicalQuery::and(vec![
        LogicalQuery::entity("Ulm").project("rev_born_in"),
        LogicalQuery::entity("Nobel_Prize").project("rev_won"),
    ]);

    let sparse_reasoner = SparseReasoner::from_kg(&kg_rev);
    let symbolic_reasoner = SymbolicReasoner::new(&kg_rev);

    let sparse_results: HashSet<_> = sparse_reasoner.predict(&query, 10).unwrap()
        .into_iter().map(|p| p.entity).collect();
    let symbolic_results: HashSet<_> = symbolic_reasoner.predict(&query, 10).unwrap()
        .into_iter().map(|p| p.entity).collect();

    assert_eq!(sparse_results, symbolic_results);
    assert!(sparse_results.contains("Einstein"));
}

#[test]
fn test_sparse_negation() {
    let kg = setup_sample_kg();
    // Add some entities to make negation interesting
    let reasoner = SparseReasoner::from_kg(&kg);
    
    // Query: NOT (won Nobel_Prize)
    let won_nobel = LogicalQuery::entity("Planck").project("won"); // Returns {Nobel_Prize}
    let query = won_nobel.not();
    
    let results = reasoner.predict(&query, 100).unwrap();
    let entities: HashSet<_> = results.into_iter().map(|p| p.entity).collect();
    
    assert!(!entities.contains("Nobel_Prize"));
    assert!(entities.contains("Einstein"));
    assert!(entities.contains("Hawking"));
}
