//! PageRank over a small citation graph.
//!
//! Run:
//! `cargo run --example pagerank_demo`

use lattix::algo::pagerank::{pagerank, PageRankConfig};
use lattix::algo::top_n;
use lattix::{KnowledgeGraph, Triple};

fn main() {
    let mut kg = KnowledgeGraph::new();

    // A small citation-shaped graph. Survey papers and methods that receive
    // citations from several neighborhoods should rise in PageRank.
    for (subject, object) in [
        ("paper_a_intro", "paper_b_survey"),
        ("paper_a_intro", "paper_c_method"),
        ("paper_d_application", "paper_b_survey"),
        ("paper_d_application", "paper_c_method"),
        ("paper_e_followup", "paper_c_method"),
        ("paper_f_benchmark", "paper_b_survey"),
        ("paper_f_benchmark", "paper_c_method"),
        ("paper_g_case_study", "paper_d_application"),
        ("paper_h_notes", "paper_b_survey"),
    ] {
        kg.add_triple(Triple::new(subject, "cites", object));
    }

    let scores = pagerank(
        &kg,
        PageRankConfig {
            damping_factor: 0.85,
            max_iterations: 100,
            tolerance: 1e-9,
        },
    );
    let top = top_n(&scores, 5);

    println!("top PageRank scores:");
    for (entity, score) in &top {
        println!("  {entity:<20} {score:.5}");
    }

    assert_eq!(scores.len(), kg.entity_count());
    assert!(matches!(
        top[0].0.as_str(),
        "paper_b_survey" | "paper_c_method"
    ));
    assert!(matches!(
        top[1].0.as_str(),
        "paper_b_survey" | "paper_c_method"
    ));
    assert!((top[0].1 - top[1].1).abs() < 1e-9);
}
