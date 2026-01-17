//! Integration tests for temporal graph operations.

use lattix_temporal::{TemporalEdge, TemporalGraph, TemporalQuery, TimeWindow};

#[test]
fn test_empty_graph() {
    let tg = TemporalGraph::new();
    assert_eq!(tg.num_edges(), 0);
    assert_eq!(tg.num_nodes(), 0);
}

#[test]
fn test_add_edges() {
    let mut tg = TemporalGraph::new();
    tg.add_edge(TemporalEdge::new(0, 1, 100));
    tg.add_edge(TemporalEdge::new(1, 2, 150));
    tg.add_edge(TemporalEdge::new(0, 2, 200));

    assert_eq!(tg.num_edges(), 3);
    assert_eq!(tg.num_nodes(), 3);
}

#[test]
fn test_edge_properties() {
    let edge = TemporalEdge::new(0, 1, 100);
    assert_eq!(edge.src, 0);
    assert_eq!(edge.dst, 1);
    assert_eq!(edge.time, 100);
}

#[test]
fn test_edges_in_window() {
    let mut tg = TemporalGraph::new();
    tg.add_edge(TemporalEdge::new(0, 1, 100));
    tg.add_edge(TemporalEdge::new(1, 2, 150));
    tg.add_edge(TemporalEdge::new(0, 2, 200));
    tg.add_edge(TemporalEdge::new(2, 3, 300));

    let edges = tg.edges_in_window(100, 200);
    assert_eq!(edges.len(), 3);

    let edges_narrow = tg.edges_in_window(140, 160);
    assert_eq!(edges_narrow.len(), 1);
}

#[test]
fn test_outgoing_in_window() {
    let mut tg = TemporalGraph::new();
    tg.add_edge(TemporalEdge::new(0, 1, 100));
    tg.add_edge(TemporalEdge::new(0, 2, 150));
    tg.add_edge(TemporalEdge::new(0, 3, 200));

    let edges = tg.outgoing_in_window(0, 0, 120);
    assert_eq!(edges.len(), 1);

    let all_edges = tg.outgoing_in_window(0, 0, 250);
    assert_eq!(all_edges.len(), 3);
}

#[test]
fn test_incoming_in_window() {
    let mut tg = TemporalGraph::new();
    tg.add_edge(TemporalEdge::new(0, 2, 100));
    tg.add_edge(TemporalEdge::new(1, 2, 150));
    tg.add_edge(TemporalEdge::new(3, 2, 200));

    let edges = tg.incoming_in_window(2, 0, 160);
    assert_eq!(edges.len(), 2);
}

#[test]
fn test_time_window_contains() {
    let window = TimeWindow::new(100, 200);
    assert!(window.contains(100));
    assert!(window.contains(150));
    assert!(window.contains(200));
    assert!(!window.contains(50));
    assert!(!window.contains(250));
}

#[test]
fn test_time_window_overlap() {
    let w1 = TimeWindow::new(100, 200);
    let w2 = TimeWindow::new(150, 250);
    let w3 = TimeWindow::new(300, 400);

    assert!(w1.overlaps(&w2));
    assert!(w2.overlaps(&w1));
    assert!(!w1.overlaps(&w3));
}

#[test]
fn test_time_window_intersect() {
    let w1 = TimeWindow::new(100, 200);
    let w2 = TimeWindow::new(150, 250);

    let intersection = w1.intersect(&w2).unwrap();
    assert_eq!(intersection.start, 150);
    assert_eq!(intersection.end, 200);

    let w3 = TimeWindow::new(300, 400);
    assert!(w1.intersect(&w3).is_none());
}

#[test]
fn test_temporal_paths() {
    let mut tg = TemporalGraph::new();
    // Create a temporal chain: 0 -> 1 -> 2 (respecting causality)
    tg.add_edge(TemporalEdge::new(0, 1, 100));
    tg.add_edge(TemporalEdge::new(1, 2, 150));

    let paths = tg.temporal_paths(0, 2, 0, 200, 3);

    // Should find path 0 -> 1 -> 2
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].len(), 2);
}

#[test]
fn test_temporal_paths_causality_violated() {
    let mut tg = TemporalGraph::new();
    // Edge 1->2 happens BEFORE 0->1, so no valid temporal path
    tg.add_edge(TemporalEdge::new(1, 2, 50));
    tg.add_edge(TemporalEdge::new(0, 1, 100));

    let paths = tg.temporal_paths(0, 2, 0, 200, 3);

    // No valid path because causality is violated
    assert!(paths.is_empty());
}

#[test]
fn test_time_range() {
    let mut tg = TemporalGraph::new();
    tg.add_edge(TemporalEdge::new(0, 1, 100));
    tg.add_edge(TemporalEdge::new(1, 2, 300));

    let (min, max) = tg.time_range();
    assert_eq!(min, 100);
    assert_eq!(max, 300);
}

#[test]
fn test_degree() {
    let mut tg = TemporalGraph::new();
    tg.add_edge(TemporalEdge::new(0, 1, 100));
    tg.add_edge(TemporalEdge::new(0, 2, 150));
    tg.add_edge(TemporalEdge::new(0, 3, 200));
    tg.add_edge(TemporalEdge::new(1, 0, 250));

    assert_eq!(tg.out_degree(0), 3);
    assert_eq!(tg.in_degree(0), 1);
}

#[test]
fn test_temporal_query_in_window() {
    let query = TemporalQuery::in_window(100, 200);
    assert!(query.src.is_none());
    assert!(query.dst.is_none());
    assert_eq!(query.window.start, 100);
    assert_eq!(query.window.end, 200);
}

#[test]
fn test_temporal_query_path() {
    let query = TemporalQuery::path(0, 5, 100, 500);
    assert_eq!(query.src, Some(0));
    assert_eq!(query.dst, Some(5));
    assert_eq!(query.max_hops, Some(10));
}

#[test]
fn test_temporal_query_with_constraints() {
    let query = TemporalQuery::path(0, 5, 100, 500)
        .with_max_hops(3)
        .with_gap_constraints(10, 100);

    assert_eq!(query.max_hops, Some(3));
    assert_eq!(query.min_gap, Some(10));
    assert_eq!(query.max_gap, Some(100));
}
