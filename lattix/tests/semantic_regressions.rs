#![allow(missing_docs)]

#[cfg(feature = "algo")]
mod parallel_edge_semantics {
    use lattix::algo::centrality::{
        betweenness_centrality, degree_centrality, eigenvector_centrality, hits, katz_centrality,
        BetweennessConfig, EigenvectorConfig, HitsConfig, KatzConfig,
    };
    use lattix::algo::pagerank::{pagerank, PageRankConfig};
    use lattix::algo::ppr::{personalized_pagerank, PprConfig};
    use lattix::{KnowledgeGraph, Triple};

    fn diamond_with_duplicate() -> KnowledgeGraph {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("A", "rel", "C"));
        kg.add_triple(Triple::new("B", "rel", "D"));
        kg.add_triple(Triple::new("C", "rel", "D"));
        kg
    }

    fn diamond_without_duplicate() -> KnowledgeGraph {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("A", "rel", "C"));
        kg.add_triple(Triple::new("B", "rel", "D"));
        kg.add_triple(Triple::new("C", "rel", "D"));
        kg
    }

    fn assert_score_maps_close(
        left: &std::collections::HashMap<lattix::EntityId, f64>,
        right: &std::collections::HashMap<lattix::EntityId, f64>,
    ) {
        assert_eq!(left.len(), right.len());
        for (entity, value) in left {
            let other = right.get(entity).expect("same entity set");
            assert!(
                (value - other).abs() < 1e-9,
                "{} differs: {value} vs {other}",
                entity.as_str()
            );
        }
    }

    #[test]
    fn duplicate_edges_do_not_skew_shortest_path_centrality() {
        let kg = diamond_with_duplicate();
        let scores = betweenness_centrality(
            &kg,
            BetweennessConfig {
                normalized: false,
                undirected: false,
            },
        );

        let b = scores.get("B").unwrap();
        let c = scores.get("C").unwrap();
        assert!(
            (b - c).abs() < 1e-9,
            "parallel A->B edge should not make B more central than C: {b} vs {c}"
        );
    }

    #[test]
    fn duplicate_edges_do_not_count_as_extra_neighbors() {
        let kg = diamond_with_duplicate();
        let degrees = degree_centrality(&kg);

        let a = degrees.get("A").unwrap();
        assert_eq!(a.out_degree, 2);
        assert!((a.out_normalized - (2.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn duplicate_edges_do_not_change_iterative_centralities() {
        let unique = diamond_without_duplicate();
        let duplicated = diamond_with_duplicate();

        assert_score_maps_close(
            &katz_centrality(&unique, KatzConfig::default()),
            &katz_centrality(&duplicated, KatzConfig::default()),
        );
        assert_score_maps_close(
            &eigenvector_centrality(&unique, EigenvectorConfig::default()),
            &eigenvector_centrality(&duplicated, EigenvectorConfig::default()),
        );
        assert_score_maps_close(
            &pagerank(&unique, PageRankConfig::default()),
            &pagerank(&duplicated, PageRankConfig::default()),
        );
        assert_score_maps_close(
            &personalized_pagerank(&unique, "A", PprConfig::default()),
            &personalized_pagerank(&duplicated, "A", PprConfig::default()),
        );

        let unique_hits = hits(&unique, HitsConfig::default());
        let duplicated_hits = hits(&duplicated, HitsConfig::default());
        for (entity, scores) in &unique_hits {
            let other = duplicated_hits.get(entity).expect("same entity set");
            assert!((scores.hub - other.hub).abs() < 1e-9);
            assert!((scores.authority - other.authority).abs() < 1e-9);
        }
    }
}

#[cfg(feature = "formats")]
mod rdf_format_regressions {
    use lattix::formats::{NQuads, NTriples, Quad, Turtle};
    use lattix::{KnowledgeGraph, Triple};
    use std::collections::HashMap;
    use std::io::Write;

    #[test]
    fn ntriples_writer_escapes_literal_lexical_forms() {
        let input = "<http://ex.org/s> <http://ex.org/p> \"he said \\\"hi\\\"\\nnext\"@en-US .\n";
        let kg = NTriples::parse(input).unwrap();

        let output = NTriples::to_string(&kg).unwrap();
        assert!(output.contains("\\\"hi\\\""));
        assert!(output.contains("\\nnext"));

        let reparsed = NTriples::parse(&output).unwrap();
        assert_eq!(kg.triple_count(), reparsed.triple_count());
        assert_eq!(
            kg.triples().next().unwrap().object(),
            reparsed.triples().next().unwrap().object()
        );
    }

    #[test]
    fn ntriples_normalizes_language_tags_and_preserves_typed_literals() {
        let input = concat!(
            "<http://ex.org/s> <http://ex.org/name> \"chat\"@fr-CA .\n",
            "<http://ex.org/s> <http://ex.org/count> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
        );
        let kg = NTriples::parse(input).unwrap();
        let objects: Vec<_> = kg
            .triples()
            .map(|triple| triple.object().as_str().to_string())
            .collect();

        assert!(objects.contains(&"\"chat\"@fr-ca".to_string()));
        assert!(objects.contains(&"\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>".to_string()));

        let output = NTriples::to_string(&kg).unwrap();
        let reparsed = NTriples::parse(&output).unwrap();
        let reparsed_objects: Vec<_> = reparsed
            .triples()
            .map(|triple| triple.object().as_str().to_string())
            .collect();
        assert_eq!(objects, reparsed_objects);
    }

    #[test]
    fn invalid_local_percent_encoding_is_rejected_by_parser() {
        let input = "<urn:lattix:local:%GG> <http://ex.org/p> <urn:lattix:local:good%20id> .\n";

        assert!(NTriples::parse(input).is_err());
    }

    #[test]
    fn knowledge_graph_file_parsing_matches_oxttl_parser() {
        let input = "<http://ex.org/s> <http://ex.org/p> \"a\\\"b\"@en-US .\n";
        let expected = NTriples::parse(input).unwrap();

        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(input.as_bytes()).unwrap();

        let parsed = KnowledgeGraph::from_ntriples_file_strict(file.path()).unwrap();
        assert_eq!(
            expected.triples().next().unwrap().object(),
            parsed.triples().next().unwrap().object()
        );
    }

    #[test]
    fn turtle_writer_uses_prefixes() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new(
            "http://example.org/Alice",
            "http://example.org/knows",
            "http://example.org/Bob",
        ));

        let prefixes: HashMap<String, String> =
            [("ex".to_string(), "http://example.org/".to_string())]
                .into_iter()
                .collect();
        let mut buf = Vec::new();
        Turtle::write(&kg, &mut buf, &prefixes).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(output.contains("@prefix ex: <http://example.org/> ."));
        assert!(output.contains("ex:Alice ex:knows ex:Bob ."));
    }

    #[test]
    fn turtle_resolves_base_iris_and_preserves_typed_literals() {
        let input = concat!(
            "@base <http://example.org/> .\n",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n",
            "<alice> <age> \"42\"^^xsd:integer .\n",
        );
        let kg = Turtle::read(std::io::Cursor::new(input), Some("http://example.org/")).unwrap();
        let triple = kg.triples().next().unwrap();

        assert_eq!(triple.subject().as_str(), "http://example.org/alice");
        assert_eq!(triple.predicate().as_str(), "http://example.org/age");
        assert_eq!(
            triple.object().as_str(),
            "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"
        );
    }

    #[test]
    fn nquads_decodes_local_terms_and_encodes_local_graph_names() {
        let input = concat!(
            "<urn:lattix:local:subject%201> ",
            "<urn:lattix:local:predicate%2Fone> ",
            "\"value\" ",
            "<urn:lattix:local:graph%201> .",
        );
        let quad = Quad::from_nquads(input).unwrap();

        assert_eq!(quad.triple.subject().as_str(), "subject 1");
        assert_eq!(quad.triple.predicate().as_str(), "predicate/one");
        assert_eq!(quad.triple.object().as_str(), "\"value\"");
        assert_eq!(quad.graph.as_deref(), Some("graph 1"));

        let output = quad.to_nquads();
        assert!(output.contains("<urn:lattix:local:graph%201>"));

        let graphs = NQuads::read(std::io::Cursor::new(output)).unwrap();
        let graph = graphs.get(&Some("graph 1".to_string())).unwrap();
        let reparsed = graph.triples().next().unwrap();
        assert_eq!(reparsed.predicate().as_str(), "predicate/one");
    }
}

mod graph_semantics {
    use lattix::{HyperGraph, HyperTriple, KnowledgeGraph, Triple};

    #[test]
    fn ntriples_roundtrips_local_ids_with_quotes_and_unicode() {
        let original = Triple::new("Alice \"A\"", "related to", "café \"central\"");

        let line = original.to_ntriples();
        assert!(line.contains("urn:lattix:local:"));

        let parsed = Triple::from_ntriples(&line).unwrap();
        assert_eq!(parsed.subject(), original.subject());
        assert_eq!(parsed.predicate(), original.predicate());
        assert_eq!(parsed.object(), original.object());
    }

    #[test]
    fn find_path_uses_first_inserted_parallel_predicate() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "knows", "B"));
        kg.add_triple(Triple::new("A", "hates", "B"));
        kg.add_triple(Triple::new("A", "loves", "B"));

        let path = kg.find_path("A", "B").unwrap();
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].predicate().as_str(), "knows");
    }

    #[test]
    fn find_path_self_returns_empty_path() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));

        let path = kg.find_path("A", "A").unwrap();
        assert!(path.is_empty());
    }

    #[test]
    fn hyper_triple_reification_anchors_statement_nodes() {
        let mut hg = HyperGraph::new();
        hg.add_hyper_triple(
            HyperTriple::from_parts("Einstein", "won", "Nobel").with_qualifier("year", "1921"),
        );

        let triples = hg.to_reified_triples();
        let has_subject = triples.iter().any(|t| {
            t.subject().as_str() == "_:stmt_0"
                && t.predicate().as_str() == "rdf:subject"
                && t.object().as_str() == "Einstein"
        });
        let has_predicate = triples.iter().any(|t| {
            t.subject().as_str() == "_:stmt_0"
                && t.predicate().as_str() == "rdf:predicate"
                && t.object().as_str() == "won"
        });
        let has_object = triples.iter().any(|t| {
            t.subject().as_str() == "_:stmt_0"
                && t.predicate().as_str() == "rdf:object"
                && t.object().as_str() == "Nobel"
        });
        let has_qualifier = triples.iter().any(|t| {
            t.subject().as_str() == "_:stmt_0"
                && t.predicate().as_str() == "year"
                && t.object().as_str() == "1921"
        });

        assert!(has_subject);
        assert!(has_predicate);
        assert!(has_object);
        assert!(has_qualifier);
    }
}

mod hetero_regressions {
    use lattix::{EdgeType, HeteroGraph, NodeType};

    #[test]
    fn metapath_neighbors_enforces_start_type() {
        let mut hg = HeteroGraph::new();
        let author = NodeType::new("author");
        let paper = NodeType::new("paper");
        let alice_idx = hg.add_node(author.clone(), "alice");
        hg.add_node(paper.clone(), "paper1");

        let writes = EdgeType::new("author", "writes", "paper");
        hg.add_edge(&writes, "alice", "paper1");

        let reachable = hg.metapath_neighbors(&paper, alice_idx, &[writes]);
        assert!(reachable.is_empty());
    }
}

mod query_api {
    use lattix::{KnowledgeGraph, Triple};

    #[test]
    fn query_execute_is_lazy_iterator() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("Alice", "knows", "Bob"));
        kg.add_triple(Triple::new("Alice", "knows", "Carol"));

        let first = kg.query().subject("Alice").execute().next().unwrap();
        assert_eq!(first.object().as_str(), "Bob");
    }
}
