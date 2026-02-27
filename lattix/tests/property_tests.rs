//! Property-based tests for lattix knowledge graph types.
//!
//! These tests verify invariants that should hold for any knowledge graph:
//! - Graph structure consistency
//! - Triple properties
//! - Index integrity
//! - Serialization roundtrips

use proptest::prelude::*;

mod triple_props {
    use super::*;

    /// Generate arbitrary entity IDs (simple strings for testing)
    fn arb_entity_id() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9_]{0,20}".prop_map(|s| s)
    }

    /// Generate arbitrary relation types
    fn arb_relation() -> impl Strategy<Value = String> {
        "[a-z_]{1,15}".prop_map(|s| s)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn triple_to_ntriples_parseable(
            subject in arb_entity_id(),
            predicate in arb_relation(),
            object in arb_entity_id(),
        ) {
            use lattix::Triple;

            let original = Triple::new(subject.as_str(), predicate.as_str(), object.as_str());
            let ntriples = original.to_ntriples();

            // Should be parseable back
            let parsed = Triple::from_ntriples(&ntriples);
            prop_assert!(
                parsed.is_ok(),
                "Failed to parse: {} (from {:?})",
                ntriples, original
            );

            let parsed = parsed.unwrap();
            prop_assert_eq!(original.subject, parsed.subject);
            prop_assert_eq!(original.predicate, parsed.predicate);
            prop_assert_eq!(original.object, parsed.object);
        }

        #[test]
        fn confidence_always_clamped(
            subject in arb_entity_id(),
            predicate in arb_relation(),
            object in arb_entity_id(),
            confidence in -10.0f32..10.0f32,
        ) {
            use lattix::Triple;

            let triple = Triple::new(subject.as_str(), predicate.as_str(), object.as_str())
                .with_confidence(confidence);

            let actual = triple.confidence.unwrap();
            prop_assert!(
                (0.0..=1.0).contains(&actual),
                "Confidence {} not in [0,1], input was {}",
                actual, confidence
            );
        }

        #[test]
        fn triple_display_contains_components(
            subject in arb_entity_id(),
            predicate in arb_relation(),
            object in arb_entity_id(),
        ) {
            use lattix::Triple;

            let triple = Triple::new(subject.as_str(), predicate.as_str(), object.as_str());
            let display = format!("{}", triple);

            prop_assert!(
                display.contains(&subject),
                "Display '{}' should contain subject '{}'",
                display, subject
            );
            prop_assert!(
                display.contains(&predicate),
                "Display '{}' should contain predicate '{}'",
                display, predicate
            );
            prop_assert!(
                display.contains(&object),
                "Display '{}' should contain object '{}'",
                display, object
            );
        }
    }
}

mod graph_props {
    use super::*;
    use std::collections::HashSet;

    fn arb_entity_id() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9]{0,10}".prop_map(|s| s)
    }

    fn arb_relation() -> impl Strategy<Value = String> {
        "[a-z_]{1,8}".prop_map(|s| s)
    }

    prop_compose! {
        fn arb_triple()(
            subject in arb_entity_id(),
            predicate in arb_relation(),
            object in arb_entity_id(),
        ) -> (String, String, String) {
            (subject, predicate, object)
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn entity_count_correct(
            triples in prop::collection::vec(arb_triple(), 0..50),
        ) {
            use lattix::{KnowledgeGraph, Triple};

            let mut kg = KnowledgeGraph::new();
            let mut unique_entities: HashSet<String> = HashSet::new();

            for (subject, predicate, object) in &triples {
                kg.add_triple(Triple::new(subject.as_str(), predicate.as_str(), object.as_str()));
                unique_entities.insert(subject.clone());
                unique_entities.insert(object.clone());
            }

            prop_assert_eq!(
                kg.entity_count(),
                unique_entities.len(),
                "Entity count mismatch: graph has {}, expected {} unique",
                kg.entity_count(), unique_entities.len()
            );
        }

        #[test]
        fn triple_count_correct(
            triples in prop::collection::vec(arb_triple(), 0..50),
        ) {
            use lattix::{KnowledgeGraph, Triple};

            let mut kg = KnowledgeGraph::new();
            for (subject, predicate, object) in &triples {
                kg.add_triple(Triple::new(subject.as_str(), predicate.as_str(), object.as_str()));
            }

            prop_assert_eq!(
                kg.triple_count(),
                triples.len(),
                "Triple count mismatch"
            );
        }

        #[test]
        fn get_entity_after_add(
            subject in arb_entity_id(),
            predicate in arb_relation(),
            object in arb_entity_id(),
        ) {
            use lattix::{KnowledgeGraph, Triple};

            let mut kg = KnowledgeGraph::new();
            kg.add_triple(Triple::new(subject.as_str(), predicate.as_str(), object.as_str()));

            // Entities should be added - if subject == object, only 1 entity; otherwise 2
            let expected_entities = if subject == object { 1 } else { 2 };
            prop_assert_eq!(
                kg.entity_count(),
                expected_entities,
                "Expected {} entities for subject='{}', object='{}'",
                expected_entities, subject, object
            );
            // Test via relations_from which uses Into<EntityId>
            let rels = kg.relations_from(subject.as_str());
            prop_assert!(
                !rels.is_empty(),
                "Subject {} should have relations",
                subject
            );
        }

        #[test]
        fn relations_from_contains_added(
            subject in arb_entity_id(),
            predicate in arb_relation(),
            object in arb_entity_id(),
        ) {
            use lattix::{KnowledgeGraph, Triple};

            let mut kg = KnowledgeGraph::new();
            kg.add_triple(Triple::new(subject.as_str(), predicate.as_str(), object.as_str()));

            let relations = kg.relations_from(subject.as_str());

            prop_assert!(
                relations.iter().any(|t| t.object.as_str() == object && t.predicate.as_str() == predicate),
                "Added triple not found in relations_from"
            );
        }

        #[test]
        fn relations_to_contains_added(
            subject in arb_entity_id(),
            predicate in arb_relation(),
            object in arb_entity_id(),
        ) {
            use lattix::{KnowledgeGraph, Triple};

            let mut kg = KnowledgeGraph::new();
            kg.add_triple(Triple::new(subject.as_str(), predicate.as_str(), object.as_str()));

            let relations = kg.relations_to(object.as_str());

            prop_assert!(
                relations.iter().any(|t| t.subject.as_str() == subject && t.predicate.as_str() == predicate),
                "Added triple not found in relations_to"
            );
        }

        #[test]
        fn relation_types_tracked(
            triples in prop::collection::vec(arb_triple(), 1..30),
        ) {
            use lattix::{KnowledgeGraph, Triple};

            let mut kg = KnowledgeGraph::new();
            let mut expected_types: HashSet<String> = HashSet::new();

            for (subject, predicate, object) in &triples {
                kg.add_triple(Triple::new(subject.as_str(), predicate.as_str(), object.as_str()));
                expected_types.insert(predicate.clone());
            }

            let actual_types: HashSet<String> = kg.relation_types()
                .iter()
                .map(|r| r.as_str().to_string())
                .collect();

            prop_assert_eq!(
                actual_types,
                expected_types,
                "Relation types mismatch"
            );
        }
    }
}

mod serialization_props {
    use super::*;

    fn arb_entity_id() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9]{0,8}".prop_map(|s| s)
    }

    fn arb_relation() -> impl Strategy<Value = String> {
        "[a-z_]{1,6}".prop_map(|s| s)
    }

    prop_compose! {
        fn arb_triple()(
            subject in arb_entity_id(),
            predicate in arb_relation(),
            object in arb_entity_id(),
        ) -> (String, String, String) {
            (subject, predicate, object)
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn json_roundtrip_preserves_structure(
            triples in prop::collection::vec(arb_triple(), 1..20),
        ) {
            use lattix::{KnowledgeGraph, Triple};

            let mut kg = KnowledgeGraph::new();
            for (subject, predicate, object) in &triples {
                kg.add_triple(Triple::new(subject.as_str(), predicate.as_str(), object.as_str()));
            }

            // Serialize to JSON
            let json = serde_json::to_string(&kg).expect("JSON serialization failed");

            // Deserialize back
            let recovered: KnowledgeGraph = serde_json::from_str(&json)
                .expect("JSON deserialization failed");

            // Verify counts match
            prop_assert_eq!(
                kg.entity_count(),
                recovered.entity_count(),
                "Entity count changed after JSON roundtrip"
            );
            prop_assert_eq!(
                kg.triple_count(),
                recovered.triple_count(),
                "Triple count changed after JSON roundtrip"
            );
        }
    }
}

mod stress_props {
    use super::*;

    fn arb_entity_id() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9]{0,5}".prop_map(|s| s)
    }

    fn arb_relation() -> impl Strategy<Value = String> {
        "[a-z]{1,4}".prop_map(|s| s)
    }

    prop_compose! {
        fn arb_triple()(
            subject in arb_entity_id(),
            predicate in arb_relation(),
            object in arb_entity_id(),
        ) -> (String, String, String) {
            (subject, predicate, object)
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        /// Test that adding many triples maintains consistency.
        /// This simulates a more realistic workload than single-triple tests.
        #[test]
        fn bulk_add_maintains_consistency(
            triples in prop::collection::vec(arb_triple(), 50..200),
        ) {
            use lattix::{KnowledgeGraph, Triple};
            use std::collections::{HashSet, HashMap};

            let mut kg = KnowledgeGraph::new();
            let mut expected_entities: HashSet<String> = HashSet::new();
            let mut expected_relations: HashMap<String, HashSet<String>> = HashMap::new();

            for (subject, predicate, object) in &triples {
                kg.add_triple(Triple::new(subject.as_str(), predicate.as_str(), object.as_str()));
                expected_entities.insert(subject.clone());
                expected_entities.insert(object.clone());

                expected_relations
                    .entry(subject.clone())
                    .or_default()
                    .insert(object.clone());
            }

            // Check entity count
            prop_assert_eq!(kg.entity_count(), expected_entities.len());

            // Check that every expected relation exists
            for (subject, objects) in &expected_relations {
                let actual_objects: HashSet<_> = kg.relations_from(subject.as_str())
                    .iter()
                    .map(|t| t.object.as_str().to_string())
                    .collect();

                for expected_obj in objects {
                    prop_assert!(
                        actual_objects.contains(expected_obj),
                        "Missing relation: {} -> {}",
                        subject, expected_obj
                    );
                }
            }
        }

        /// Test that the graph handles self-loops correctly.
        #[test]
        fn self_loops_handled(
            entity in arb_entity_id(),
            predicate in arb_relation(),
        ) {
            use lattix::{KnowledgeGraph, Triple};

            let mut kg = KnowledgeGraph::new();
            kg.add_triple(Triple::new(entity.as_str(), predicate.as_str(), entity.as_str()));

            // Only one entity should exist
            prop_assert_eq!(kg.entity_count(), 1);

            // The entity should have both outgoing and incoming relations
            let outgoing = kg.relations_from(entity.as_str());
            let incoming = kg.relations_to(entity.as_str());

            prop_assert!(!outgoing.is_empty(), "Self-loop should appear in relations_from");
            prop_assert!(!incoming.is_empty(), "Self-loop should appear in relations_to");
        }
    }
}

mod invariant_props {
    use super::*;

    fn arb_entity_id() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9]{0,8}".prop_map(|s| s)
    }

    fn arb_relation() -> impl Strategy<Value = String> {
        "[a-z_]{1,6}".prop_map(|s| s)
    }

    prop_compose! {
        fn arb_triple()(
            subject in arb_entity_id(),
            predicate in arb_relation(),
            object in arb_entity_id(),
        ) -> (String, String, String) {
            (subject, predicate, object)
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn all_triple_entities_exist(
            triples in prop::collection::vec(arb_triple(), 1..30),
        ) {
            use lattix::{KnowledgeGraph, Triple};

            let mut kg = KnowledgeGraph::new();
            for (subject, predicate, object) in &triples {
                kg.add_triple(Triple::new(subject.as_str(), predicate.as_str(), object.as_str()));
            }

            // Every triple's subject and object should be retrievable
            for triple in kg.triples() {
                prop_assert!(
                    kg.get_entity(&triple.subject).is_some(),
                    "Triple subject {} not in entity index",
                    triple.subject
                );
                prop_assert!(
                    kg.get_entity(&triple.object).is_some(),
                    "Triple object {} not in entity index",
                    triple.object
                );
            }
        }

        #[test]
        fn neighbors_bidirectional_consistency(
            triples in prop::collection::vec(arb_triple(), 5..30),
        ) {
            use lattix::{KnowledgeGraph, Triple};
            use std::collections::HashSet;

            let mut kg = KnowledgeGraph::new();
            for (subject, predicate, object) in &triples {
                kg.add_triple(Triple::new(subject.as_str(), predicate.as_str(), object.as_str()));
            }

            // For each entity, outgoing edges should match incoming edges at targets
            for entity in kg.entities() {
                let entity_id = entity.id.as_str();

                // Outgoing: relations_from(entity) -> objects
                let outgoing: HashSet<_> = kg.relations_from(entity_id)
                    .iter()
                    .map(|t| t.object.as_str().to_string())
                    .collect();

                // Each outgoing target should have us in their incoming
                for target in &outgoing {
                    let incoming_to_target: HashSet<_> = kg.relations_to(target.as_str())
                        .iter()
                        .map(|t| t.subject.as_str().to_string())
                        .collect();

                    prop_assert!(
                        incoming_to_target.contains(entity_id),
                        "Entity {} has outgoing to {}, but {} doesn't have incoming from {}",
                        entity_id, target, target, entity_id
                    );
                }
            }
        }
    }
}
