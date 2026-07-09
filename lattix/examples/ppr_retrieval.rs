//! Passage retrieval with Personalized PageRank over a toy knowledge graph.
//!
//! Run:
//! `cargo run --example ppr_retrieval`

use std::collections::{HashMap, HashSet};

use lattix::algo::ppr::{personalized_pagerank, PprConfig};
use lattix::{EntityId, KnowledgeGraph, Triple};

#[derive(Debug)]
struct Passage {
    id: &'static str,
    text: &'static str,
    entities: &'static [&'static str],
}

fn main() {
    let passages = [
        Passage {
            id: "p_ada",
            text: "Ada Lovelace wrote notes about the Analytical Engine.",
            entities: &["Ada Lovelace", "Analytical Engine"],
        },
        Passage {
            id: "p_engine",
            text: "The Analytical Engine was designed by Charles Babbage.",
            entities: &["Analytical Engine", "Charles Babbage"],
        },
        Passage {
            id: "p_babbage",
            text: "Charles Babbage studied mechanical computation.",
            entities: &["Charles Babbage", "mechanical computation"],
        },
        Passage {
            id: "p_hopper",
            text: "Grace Hopper worked on COBOL compilers.",
            entities: &["Grace Hopper", "COBOL", "compilers"],
        },
    ];

    let mut kg = KnowledgeGraph::new();
    for (subject, predicate, object) in [
        ("Ada Lovelace", "wrote_notes_about", "Analytical Engine"),
        ("Analytical Engine", "designed_by", "Charles Babbage"),
        ("Charles Babbage", "studied", "mechanical computation"),
        ("Grace Hopper", "worked_on", "COBOL"),
        ("Grace Hopper", "worked_on", "compilers"),
    ] {
        kg.add_triple(Triple::new(subject, predicate, object));
    }

    let query = "who designed the machine Ada Lovelace wrote notes about?";
    let seed = "Ada Lovelace";

    let ppr_scores = personalized_pagerank(
        &kg,
        seed,
        PprConfig {
            damping: 0.85,
            max_iterations: 100,
            tolerance: 1e-9,
        },
    );

    let graph_ranked = rank_by_graph(&passages, &ppr_scores, &[seed]);
    let lexical_ranked = rank_by_lexical(&passages, query);

    println!("query: {query}");
    println!("seed:  {seed}");

    println!("\nlexical baseline:");
    for (passage, score) in lexical_ranked.iter().take(3) {
        println!("  {:<10} score={score:<2} {}", passage.id, passage.text);
    }

    println!("\ngraph retrieval:");
    for (passage, score) in graph_ranked.iter().take(3) {
        println!("  {:<10} score={score:.5} {}", passage.id, passage.text);
    }

    assert_eq!(lexical_ranked[0].0.id, "p_ada");
    assert_eq!(graph_ranked[0].0.id, "p_engine");
    assert!(
        ppr_scores
            .get("Charles Babbage")
            .copied()
            .unwrap_or_default()
            > ppr_scores.get("Grace Hopper").copied().unwrap_or_default()
    );
}

fn rank_by_graph<'a>(
    passages: &'a [Passage],
    scores: &HashMap<EntityId, f64>,
    excluded_entities: &[&str],
) -> Vec<(&'a Passage, f64)> {
    let excluded_entities = excluded_entities.iter().copied().collect::<HashSet<_>>();
    let mut ranked = passages
        .iter()
        .map(|passage| {
            let score = passage
                .entities
                .iter()
                .filter(|entity| !excluded_entities.contains(*entity))
                .map(|entity| scores.get(*entity).copied().unwrap_or_default())
                .sum();
            (passage, score)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(compare_scored_passages);
    ranked
}

fn rank_by_lexical<'a>(passages: &'a [Passage], query: &str) -> Vec<(&'a Passage, usize)> {
    let query_terms = tokenize(query);
    let mut ranked = passages
        .iter()
        .map(|passage| {
            let score = tokenize(passage.text).intersection(&query_terms).count();
            (passage, score)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.id.cmp(b.0.id)));
    ranked
}

fn tokenize(text: &str) -> HashSet<String> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|term| term.len() > 2)
        .map(str::to_ascii_lowercase)
        .collect()
}

fn compare_scored_passages(left: &(&Passage, f64), right: &(&Passage, f64)) -> std::cmp::Ordering {
    right
        .1
        .partial_cmp(&left.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.0.id.cmp(right.0.id))
}
