use lattix_kge::models::{Query2box, TransE};
use lattix_kge::reason::{LogicalQuery, ReasoningModel};
use lattix_kge::{Fact, TrainingConfig};

#[test]
fn test_logical_query_structure() {
    // ((e1, r1) AND (e2, r2)) -> r3
    let q1 = LogicalQuery::entity("A").project("r1");
    let q2 = LogicalQuery::entity("B").project("r2");
    let query = LogicalQuery::and(vec![q1, q2]).project("r3");

    if let LogicalQuery::Projection(sub, rel) = query {
        assert_eq!(rel, "r3");
        if let LogicalQuery::Intersection(vec) = *sub {
            assert_eq!(vec.len(), 2);
        } else {
            panic!("Expected Intersection");
        }
    } else {
        panic!("Expected Projection");
    }
}

#[test]
fn test_query2box_reasoning() {
    let mut model = Query2box::new(32);
    // Mocking some data since we don't have a real training loop for multi-hop yet
    // In a real scenario, this would be trained.
    
    // For now, let's just test that the reasoning methods exist and don't crash
    let query = LogicalQuery::entity("Einstein").project("won");
    let result = model.predict_query(&query, 5);
    // Should fail with EntityNotFound since we haven't trained/initialized
    assert!(result.is_err());
}
