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
            prop_assert_eq!(original.subject(), parsed.subject());
            prop_assert_eq!(original.predicate(), parsed.predicate());
            prop_assert_eq!(original.object(), parsed.object());
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

            let actual = triple.confidence().unwrap();
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
                relations.iter().any(|t| t.object().as_str() == object && t.predicate().as_str() == predicate),
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
                relations.iter().any(|t| t.subject().as_str() == subject && t.predicate().as_str() == predicate),
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
                    .map(|t| t.object().as_str().to_string())
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

mod clear_props {
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
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Clearing a graph must fully reset all state -- no residual data
        /// in derived indexes (subject_index, object_index, relation_type_cache).
        #[test]
        fn clear_resets_all_state(
            triples in prop::collection::vec(arb_triple(), 1..50),
        ) {
            use lattix::{KnowledgeGraph, Triple};

            let mut kg = KnowledgeGraph::new();
            for (subject, predicate, object) in &triples {
                kg.add_triple(Triple::new(subject.as_str(), predicate.as_str(), object.as_str()));
            }

            kg.clear();

            prop_assert_eq!(kg.entity_count(), 0);
            prop_assert_eq!(kg.triple_count(), 0);
            prop_assert_eq!(kg.relation_type_count(), 0);

            // Derived indexes must be empty too
            for (subject, _, _) in &triples {
                prop_assert!(
                    kg.relations_from(subject.as_str()).is_empty(),
                    "relations_from should be empty after clear"
                );
            }
        }

        /// Adding triples after clear should work identically to a fresh graph.
        #[test]
        fn add_after_clear_matches_fresh(
            first_batch in prop::collection::vec(arb_triple(), 1..20),
            second_batch in prop::collection::vec(arb_triple(), 1..20),
        ) {
            use lattix::{KnowledgeGraph, Triple};

            // Build, clear, re-add
            let mut kg = KnowledgeGraph::new();
            for (s, p, o) in &first_batch {
                kg.add_triple(Triple::new(s.as_str(), p.as_str(), o.as_str()));
            }
            kg.clear();
            for (s, p, o) in &second_batch {
                kg.add_triple(Triple::new(s.as_str(), p.as_str(), o.as_str()));
            }

            // Fresh graph with only second batch
            let mut fresh = KnowledgeGraph::new();
            for (s, p, o) in &second_batch {
                fresh.add_triple(Triple::new(s.as_str(), p.as_str(), o.as_str()));
            }

            prop_assert_eq!(kg.entity_count(), fresh.entity_count());
            prop_assert_eq!(kg.triple_count(), fresh.triple_count());
            prop_assert_eq!(kg.relation_type_count(), fresh.relation_type_count());
        }
    }
}

mod remove_triple_props {
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
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Removing a triple decreases triple_count by 1.
        #[test]
        fn remove_then_count(
            triples in prop::collection::vec(arb_triple(), 1..30),
            remove_idx in any::<prop::sample::Index>(),
        ) {
            use lattix::{KnowledgeGraph, Triple, RelationType, EntityId};

            let mut kg = KnowledgeGraph::new();
            for (s, p, o) in &triples {
                kg.add_triple(Triple::new(s.as_str(), p.as_str(), o.as_str()));
            }

            let before = kg.triple_count();
            let idx = remove_idx.index(triples.len());
            let (ref s, ref p, ref o) = triples[idx];
            let removed = kg.remove_triple(
                &EntityId::new(s.as_str()),
                &RelationType::new(p.as_str()),
                &EntityId::new(o.as_str()),
            );

            prop_assert!(removed, "Triple should have been found");
            prop_assert_eq!(kg.triple_count(), before - 1);
        }

        /// Removing a nonexistent triple returns false and changes nothing.
        #[test]
        fn remove_nonexistent(
            triples in prop::collection::vec(arb_triple(), 0..20),
        ) {
            use lattix::{KnowledgeGraph, Triple, RelationType, EntityId};

            let mut kg = KnowledgeGraph::new();
            for (s, p, o) in &triples {
                kg.add_triple(Triple::new(s.as_str(), p.as_str(), o.as_str()));
            }

            let before = kg.triple_count();
            let removed = kg.remove_triple(
                &EntityId::new("ZZZZ_nonexistent"),
                &RelationType::new("no_such_rel"),
                &EntityId::new("ZZZZ_also_missing"),
            );

            prop_assert!(!removed);
            prop_assert_eq!(kg.triple_count(), before);
        }

        /// Add, remove, re-add should work correctly.
        #[test]
        fn add_remove_add(
            subject in arb_entity_id(),
            predicate in arb_relation(),
            object in arb_entity_id(),
        ) {
            use lattix::{KnowledgeGraph, Triple, RelationType, EntityId};

            let mut kg = KnowledgeGraph::new();
            kg.add_triple(Triple::new(subject.as_str(), predicate.as_str(), object.as_str()));
            prop_assert_eq!(kg.triple_count(), 1);

            let removed = kg.remove_triple(
                &EntityId::new(subject.as_str()),
                &RelationType::new(predicate.as_str()),
                &EntityId::new(object.as_str()),
            );
            prop_assert!(removed);
            prop_assert_eq!(kg.triple_count(), 0);

            // Re-add
            kg.add_triple(Triple::new(subject.as_str(), predicate.as_str(), object.as_str()));
            prop_assert_eq!(kg.triple_count(), 1);

            // Verify indexes work after re-add
            let rels = kg.relations_from(subject.as_str());
            prop_assert!(!rels.is_empty(), "relations_from should find the re-added triple");
            let rels_to = kg.relations_to(object.as_str());
            prop_assert!(!rels_to.is_empty(), "relations_to should find the re-added triple");
        }

        /// After removal, indexes remain consistent (bidirectional check).
        #[test]
        fn remove_preserves_index_consistency(
            triples in prop::collection::vec(arb_triple(), 2..30),
            remove_idx in any::<prop::sample::Index>(),
        ) {
            use lattix::{KnowledgeGraph, Triple, RelationType, EntityId};

            let mut kg = KnowledgeGraph::new();
            for (s, p, o) in &triples {
                kg.add_triple(Triple::new(s.as_str(), p.as_str(), o.as_str()));
            }

            let idx = remove_idx.index(triples.len());
            let (ref s, ref p, ref o) = triples[idx];
            kg.remove_triple(
                &EntityId::new(s.as_str()),
                &RelationType::new(p.as_str()),
                &EntityId::new(o.as_str()),
            );

            // Verify bidirectional consistency for all remaining triples
            for triple in kg.triples() {
                let from_rels = kg.relations_from(triple.subject().as_str());
                prop_assert!(
                    from_rels.iter().any(|t| *t.predicate() == *triple.predicate() && *t.object() == *triple.object()),
                    "Triple {:?} not found via relations_from",
                    triple
                );

                let to_rels = kg.relations_to(triple.object().as_str());
                prop_assert!(
                    to_rels.iter().any(|t| *t.predicate() == *triple.predicate() && *t.subject() == *triple.subject()),
                    "Triple {:?} not found via relations_to",
                    triple
                );
            }
        }
    }
}

mod format_roundtrip_props {
    use super::*;

    /// Generate valid IRI strings for entity names.
    fn iri_string() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9]{1,20}".prop_map(|s| format!("http://example.org/{}", s))
    }

    /// Generate relation IRIs.
    fn iri_relation() -> impl Strategy<Value = String> {
        "[a-z][a-z_]{1,10}".prop_map(|s| format!("http://example.org/{}", s))
    }

    prop_compose! {
        fn arb_iri_triple()(
            subject in iri_string(),
            predicate in iri_relation(),
            object in iri_string(),
        ) -> (String, String, String) {
            (subject, predicate, object)
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// N-Triples round-trip: serialize then parse back, triple count must match.
        #[test]
        fn ntriples_roundtrip(
            triples in prop::collection::vec(arb_iri_triple(), 1..20),
        ) {
            use lattix::{KnowledgeGraph, Triple};
            use lattix::formats::NTriples;

            let mut kg = KnowledgeGraph::new();
            for (s, p, o) in &triples {
                kg.add_triple(Triple::new(s.as_str(), p.as_str(), o.as_str()));
            }

            let serialized = NTriples::to_string(&kg).expect("NTriples serialization failed");
            let recovered = NTriples::parse(&serialized).expect("NTriples parse failed");

            prop_assert_eq!(
                kg.triple_count(),
                recovered.triple_count(),
                "Triple count changed after N-Triples roundtrip: serialized=\n{}",
                serialized
            );
        }

        /// JSON serde round-trip at scale: verify triple_count, entity_count, and
        /// that relations_from still works (index rebuilt correctly).
        #[test]
        fn json_serde_roundtrip_at_scale(
            triples in prop::collection::vec(arb_iri_triple(), 10..100),
        ) {
            use lattix::{KnowledgeGraph, Triple};

            let mut kg = KnowledgeGraph::new();
            for (s, p, o) in &triples {
                kg.add_triple(Triple::new(s.as_str(), p.as_str(), o.as_str()));
            }

            let json = serde_json::to_string(&kg).expect("JSON serialization failed");
            let recovered: KnowledgeGraph =
                serde_json::from_str(&json).expect("JSON deserialization failed");

            prop_assert_eq!(
                kg.triple_count(),
                recovered.triple_count(),
                "Triple count changed after JSON roundtrip"
            );
            prop_assert_eq!(
                kg.entity_count(),
                recovered.entity_count(),
                "Entity count changed after JSON roundtrip"
            );

            // Verify relations_from works on the first entity (index rebuilt).
            let first_subject = &triples[0].0;
            let original_rels = kg.relations_from(first_subject.as_str());
            let recovered_rels = recovered.relations_from(first_subject.as_str());
            prop_assert_eq!(
                original_rels.len(),
                recovered_rels.len(),
                "relations_from count differs for '{}' after JSON roundtrip",
                first_subject
            );
        }

        /// JSON-LD round-trip: serialize then parse back, triple count must match.
        #[test]
        fn jsonld_roundtrip(
            triples in prop::collection::vec(arb_iri_triple(), 1..20),
        ) {
            use lattix::{KnowledgeGraph, Triple};
            use lattix::formats::JsonLd;

            let mut kg = KnowledgeGraph::new();
            for (s, p, o) in &triples {
                kg.add_triple(Triple::new(s.as_str(), p.as_str(), o.as_str()));
            }

            let serialized = JsonLd::to_string(&kg).expect("JSON-LD serialization failed");
            let recovered = JsonLd::parse(&serialized).expect("JSON-LD parse failed");

            prop_assert_eq!(
                kg.triple_count(),
                recovered.triple_count(),
                "Triple count changed after JSON-LD roundtrip"
            );
            prop_assert_eq!(
                kg.entity_count(),
                recovered.entity_count(),
                "Entity count changed after JSON-LD roundtrip"
            );
        }

        /// Turtle round-trip: serialize then parse back, triple count must match.
        #[test]
        fn turtle_roundtrip(
            triples in prop::collection::vec(arb_iri_triple(), 1..20),
        ) {
            use lattix::{KnowledgeGraph, Triple};
            use lattix::formats::Turtle;

            let mut kg = KnowledgeGraph::new();
            for (s, p, o) in &triples {
                kg.add_triple(Triple::new(s.as_str(), p.as_str(), o.as_str()));
            }

            let serialized = Turtle::to_string(&kg).expect("Turtle serialization failed");
            let recovered = Turtle::read(std::io::Cursor::new(serialized.as_bytes()), None)
                .expect("Turtle parse failed");

            prop_assert_eq!(
                kg.triple_count(),
                recovered.triple_count(),
                "Triple count changed after Turtle roundtrip: serialized=\n{}",
                serialized
            );
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
                    kg.get_entity(triple.subject()).is_some(),
                    "Triple subject {} not in entity index",
                    triple.subject()
                );
                prop_assert!(
                    kg.get_entity(triple.object()).is_some(),
                    "Triple object {} not in entity index",
                    triple.object()
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
                    .map(|t| t.object().as_str().to_string())
                    .collect();

                // Each outgoing target should have us in their incoming
                for target in &outgoing {
                    let incoming_to_target: HashSet<_> = kg.relations_to(target.as_str())
                        .iter()
                        .map(|t| t.subject().as_str().to_string())
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

mod hypergraph_props {
    use super::*;

    fn arb_entity_id() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9]{0,8}".prop_map(|s| s)
    }

    fn arb_relation() -> impl Strategy<Value = String> {
        "[a-z_]{1,6}".prop_map(|s| s)
    }

    prop_compose! {
        fn arb_hyper_triple()(
            subject in arb_entity_id(),
            predicate in arb_relation(),
            object in arb_entity_id(),
            qual_keys in prop::collection::vec(arb_relation(), 0..3),
            qual_vals in prop::collection::vec(arb_entity_id(), 0..3),
        ) -> (String, String, String, Vec<(String, String)>) {
            let qualifiers: Vec<(String, String)> = qual_keys.into_iter()
                .zip(qual_vals)
                .collect();
            (subject, predicate, object, qualifiers)
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Roundtrip: add hyper-triples, verify find_by_entity returns them.
        #[test]
        fn hyper_triple_roundtrip_find_by_entity(
            items in prop::collection::vec(arb_hyper_triple(), 1..20),
        ) {
            use lattix::{HyperGraph, HyperTriple};

            let mut hg = HyperGraph::new();
            for (s, p, o, quals) in &items {
                let mut ht = HyperTriple::from_parts(s.as_str(), p.as_str(), o.as_str());
                for (k, v) in quals {
                    ht = ht.with_qualifier(k.as_str(), v.as_str());
                }
                hg.add_hyper_triple(ht);
            }

            // Every subject should be findable
            for (s, _, _, _) in &items {
                let results = hg.find_by_entity(s.as_str());
                prop_assert!(
                    !results.is_empty(),
                    "find_by_entity('{}') returned empty, but it was added as a subject",
                    s
                );
            }
        }

        /// Qualifier handling: add hyper-triples with qualifiers, verify find_by_qualifier.
        #[test]
        fn hyper_triple_find_by_qualifier(
            items in prop::collection::vec(arb_hyper_triple(), 1..15),
        ) {
            use lattix::{HyperGraph, HyperTriple};

            let mut hg = HyperGraph::new();
            for (s, p, o, quals) in &items {
                let mut ht = HyperTriple::from_parts(s.as_str(), p.as_str(), o.as_str());
                for (k, v) in quals {
                    ht = ht.with_qualifier(k.as_str(), v.as_str());
                }
                hg.add_hyper_triple(ht);
            }

            // For each item with qualifiers, find_by_qualifier should return it
            for (s, _, _, quals) in &items {
                for (k, v) in quals {
                    let results = hg.find_by_qualifier(k.as_str(), v.as_str());
                    // At least the one we added should be present (may not be unique
                    // if multiple items share the same qualifier key-value).
                    let found = results.iter().any(|ht| ht.core.subject().as_str() == s);
                    prop_assert!(
                        found,
                        "find_by_qualifier('{}', '{}') did not return hyper-triple with subject '{}'",
                        k, v, s
                    );
                }
            }
        }

        /// Reification: verify reify() produces valid triples.
        #[test]
        fn hyperedge_reify_produces_valid_triples(
            relation in arb_relation(),
            bindings in prop::collection::vec(
                (arb_relation(), arb_entity_id()), 1..6
            ),
        ) {
            use lattix::HyperEdge;

            let mut he = HyperEdge::new(relation.as_str());
            for (role, entity) in &bindings {
                he = he.with_binding(role.as_str(), entity.as_str());
            }

            let reified = he.reify("_:test_node");

            // Should have 1 (rdf:type) + N (bindings) triples
            prop_assert_eq!(
                reified.len(),
                1 + bindings.len(),
                "Reified triple count mismatch: expected {}, got {}",
                1 + bindings.len(),
                reified.len()
            );

            // First triple is rdf:type
            prop_assert_eq!(reified[0].predicate().as_str(), "rdf:type");
            prop_assert_eq!(reified[0].object().as_str(), relation.as_str());

            // Each subsequent triple should have the intermediate as subject
            for triple in &reified {
                prop_assert_eq!(
                    triple.subject().as_str(),
                    "_:test_node",
                    "Reified triple subject should be the intermediate node"
                );
            }
        }
    }
}

mod dual_storage_consistency {
    use super::*;

    fn arb_entity_id() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9]{0,6}".prop_map(|s| s)
    }

    fn arb_relation() -> impl Strategy<Value = String> {
        "[a-z_]{1,5}".prop_map(|s| s)
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

    /// Operation: true = add, false = remove (from existing).
    fn arb_op() -> impl Strategy<Value = bool> {
        prop::bool::weighted(0.7) // 70% add, 30% remove
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Interleave add_triple and remove_triple, then assert:
        /// - petgraph edge count == triple vec length
        /// - All triples in vec are findable via petgraph traversal
        /// - Subject/object/predicate indexes are consistent
        #[test]
        fn interleaved_add_remove_consistency(
            initial in prop::collection::vec(arb_triple(), 5..30),
            ops in prop::collection::vec((arb_op(), any::<prop::sample::Index>()), 5..30),
        ) {
            use lattix::{KnowledgeGraph, Triple};

            let mut kg = KnowledgeGraph::new();
            for (s, p, o) in &initial {
                kg.add_triple(Triple::new(s.as_str(), p.as_str(), o.as_str()));
            }

            // Apply interleaved ops
            for (is_add, idx) in &ops {
                if *is_add {
                    // Re-add a triple from the initial set (may create duplicates)
                    let i = idx.index(initial.len());
                    let (ref s, ref p, ref o) = initial[i];
                    kg.add_triple(Triple::new(s.as_str(), p.as_str(), o.as_str()));
                } else {
                    // Remove a triple that currently exists (if any)
                    let count = kg.triple_count();
                    if count > 0 {
                        let i = idx.index(count);
                        let triple: Triple = kg.triples().nth(i).unwrap().clone();
                        kg.remove_triple(
                            triple.subject(),
                            triple.predicate(),
                            triple.object(),
                        );
                    }
                }
            }

            // Invariant 1: petgraph edge count == triple vec length
            let pg_edges = kg.as_petgraph().edge_count();
            let vec_len = kg.triple_count();
            prop_assert_eq!(
                pg_edges,
                vec_len,
                "petgraph edge count ({}) != triple vec length ({})",
                pg_edges,
                vec_len
            );

            // Invariant 2: all triples findable via subject_index
            for triple in kg.triples() {
                let from_rels = kg.relations_from(triple.subject().as_str());
                prop_assert!(
                    from_rels.iter().any(|t|
                        t.predicate() == triple.predicate() && t.object() == triple.object()
                    ),
                    "Triple ({}, {}, {}) not found via relations_from",
                    triple.subject(), triple.predicate(), triple.object()
                );

                let to_rels = kg.relations_to(triple.object().as_str());
                prop_assert!(
                    to_rels.iter().any(|t|
                        t.predicate() == triple.predicate() && t.subject() == triple.subject()
                    ),
                    "Triple ({}, {}, {}) not found via relations_to",
                    triple.subject(), triple.predicate(), triple.object()
                );

                let pred_rels = kg.triples_with_relation(triple.predicate().as_str());
                prop_assert!(
                    pred_rels.iter().any(|t|
                        t.subject() == triple.subject() && t.object() == triple.object()
                    ),
                    "Triple ({}, {}, {}) not found via triples_with_relation",
                    triple.subject(), triple.predicate(), triple.object()
                );
            }

            // Invariant 3: petgraph traversal finds edges for all triples
            for triple in kg.triples() {
                let src = kg.get_node_index(triple.subject());
                let dst = kg.get_node_index(triple.object());
                prop_assert!(src.is_some(), "Subject node missing from petgraph");
                prop_assert!(dst.is_some(), "Object node missing from petgraph");

                let has_edge = kg.has_edge(
                    triple.subject().as_str(),
                    triple.object().as_str()
                );
                prop_assert!(
                    has_edge,
                    "petgraph has no edge for triple ({}, {}, {})",
                    triple.subject(), triple.predicate(), triple.object()
                );
            }
        }
    }
}

mod ntriples_lenient_props {
    use std::io::Write;

    #[test]
    fn lenient_skips_malformed_and_reports_count() {
        use lattix::KnowledgeGraph;

        let dir = std::env::temp_dir();
        let path = dir.join("lattix_test_lenient.nt");

        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "<http://ex.org/A> <http://ex.org/r> <http://ex.org/B> .").unwrap();
            writeln!(f, "this is not valid ntriples").unwrap();
            writeln!(f, "# comment line").unwrap();
            writeln!(f, "").unwrap();
            writeln!(f, "also bad").unwrap();
            writeln!(f, "<http://ex.org/C> <http://ex.org/r> <http://ex.org/D> .").unwrap();
        }

        let (kg, skipped) = KnowledgeGraph::from_ntriples_file_lenient(&path).unwrap();

        assert_eq!(kg.triple_count(), 2, "should parse 2 valid triples");
        assert_eq!(skipped, 2, "should skip 2 malformed lines");

        // Also verify that from_ntriples_file (the original lenient API) still works
        let kg2 = KnowledgeGraph::from_ntriples_file(&path).unwrap();
        assert_eq!(kg2.triple_count(), 2);

        std::fs::remove_file(path).ok();
    }
}
