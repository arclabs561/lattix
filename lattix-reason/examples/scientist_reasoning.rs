use lattix_core::{KnowledgeGraph, Triple};
use lattix_kge::models::Query2box;
use lattix_reason::{LogicalQuery, Reasoner, SymbolicReasoner, SparseReasoner};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup a small Knowledge Graph
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("Einstein", "born_in", "Ulm"));
    kg.add_triple(Triple::new("Ulm", "located_in", "Germany"));
    kg.add_triple(Triple::new("Einstein", "won", "Nobel_Prize"));
    kg.add_triple(Triple::new("Nobel_Prize", "category", "Physics"));
    kg.add_triple(Triple::new("Planck", "won", "Nobel_Prize"));
    kg.add_triple(Triple::new("Planck", "born_in", "Kiel"));
    kg.add_triple(Triple::new("Kiel", "located_in", "Germany"));

    // 2. Define a multi-hop logical query:
    // "Scientists born in Germany who won a Nobel Prize in Physics"
    // Query: (born_in -> located_in -> Germany) AND (won -> Nobel_Prize)
    let born_in_germany = LogicalQuery::entity("Germany")
        .project("rev_located_in") // Assuming reverse relations for simplicity
        .project("rev_born_in");
    
    let won_physics = LogicalQuery::entity("Physics")
        .project("rev_category")
        .project("rev_won");

    let query = LogicalQuery::and(vec![born_in_germany, won_physics]);

    println!("Query (DNF): {:?}", query.to_dnf());

    // 3. Symbolic Reasoning (Exact paths)
    let symbolic = SymbolicReasoner::new(&kg);
    let results = symbolic.predict(&query, 10)?;
    println!("Symbolic Results: {:?}", results);

    // 4. Sparse Matrix Reasoning (Fast vectorized)
    let sparse = SparseReasoner::from_kg(&kg);
    let results = sparse.predict(&query, 10)?;
    println!("Sparse Results: {:?}", results);

    Ok(())
}
