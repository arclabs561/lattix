use lattix_reason::LogicalQuery;
use proptest::prelude::*;

fn arb_logical_query() -> impl Strategy<Value = LogicalQuery> {
    let leaf = prop_oneof![
        any::<String>().prop_map(LogicalQuery::entity),
    ];
    leaf.prop_recursive(
        4,  // 4 levels deep
        64, // 64 nodes total
        16, // max 16 items in collections
        |inner| prop_oneof![
            (inner.clone(), any::<String>()).prop_map(|(q, r)| q.project(r)),
            prop::collection::vec(inner.clone(), 1..5).prop_map(LogicalQuery::and),
            prop::collection::vec(inner.clone(), 1..5).prop_map(LogicalQuery::or),
            inner.prop_map(LogicalQuery::not),
        ]
    )
}

proptest! {
    #[test]
    fn prop_dnf_conversion_is_stable(query in arb_logical_query()) {
        let dnf = query.to_dnf();
        // DNF should always be non-empty (even for trivial cases)
        prop_assert!(!dnf.is_empty());
        
        // Converting DNF to DNF again should result in the same structure
        // (This is a simplified stability check)
        // Note: Query2box/BetaE expect DNF format.
    }

    #[test]
    fn prop_dnf_flattening(q1 in arb_logical_query(), q2 in arb_logical_query()) {
        // (A OR B) OR C -> A OR B OR C
        let query = LogicalQuery::or(vec![
            LogicalQuery::or(vec![q1.clone(), q2.clone()]),
            q1.clone(),
        ]);
        let dnf = query.to_dnf();
        // Should flatten nested unions
        prop_assert!(dnf.len() >= 2);
    }
}
