//! Temporal graph storage.

use crate::edge::{TemporalEdge, Timestamp};
use crate::{Error, Result};
use smallvec::SmallVec;
use std::collections::HashMap;

/// A temporal graph storing edges with timestamps.
///
/// Optimized for temporal queries:
/// - Edges are sorted by time for efficient window queries
/// - Adjacency lists for fast neighbor lookup
/// - Time index for range queries
#[derive(Debug, Clone)]
pub struct TemporalGraph {
    /// All edges, sorted by timestamp.
    edges: Vec<TemporalEdge>,
    /// Adjacency list: node -> outgoing edge indices.
    adj_out: HashMap<u32, SmallVec<[usize; 8]>>,
    /// Reverse adjacency: node -> incoming edge indices.
    adj_in: HashMap<u32, SmallVec<[usize; 8]>>,
    /// Whether edges are currently sorted.
    sorted: bool,
    /// Minimum timestamp.
    min_time: Timestamp,
    /// Maximum timestamp.
    max_time: Timestamp,
}

impl Default for TemporalGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalGraph {
    /// Create an empty temporal graph.
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            adj_out: HashMap::new(),
            adj_in: HashMap::new(),
            sorted: true,
            min_time: Timestamp::MAX,
            max_time: Timestamp::MIN,
        }
    }

    /// Create with estimated capacity.
    pub fn with_capacity(edges: usize, nodes: usize) -> Self {
        Self {
            edges: Vec::with_capacity(edges),
            adj_out: HashMap::with_capacity(nodes),
            adj_in: HashMap::with_capacity(nodes),
            sorted: true,
            min_time: Timestamp::MAX,
            max_time: Timestamp::MIN,
        }
    }

    /// Add a temporal edge.
    pub fn add_edge(&mut self, edge: TemporalEdge) {
        let idx = self.edges.len();

        // Update time bounds
        self.min_time = self.min_time.min(edge.time);
        self.max_time = self.max_time.max(edge.time);

        // Check if still sorted
        if !self.edges.is_empty() && edge.time < self.edges.last().unwrap().time {
            self.sorted = false;
        }

        // Add to adjacency lists
        self.adj_out.entry(edge.src).or_default().push(idx);
        self.adj_in.entry(edge.dst).or_default().push(idx);

        // Ensure nodes exist in both maps
        self.adj_out.entry(edge.dst).or_default();
        self.adj_in.entry(edge.src).or_default();

        self.edges.push(edge);
    }

    /// Ensure edges are sorted by time.
    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.edges.sort();
            self.rebuild_adjacency();
            self.sorted = true;
        }
    }

    /// Rebuild adjacency lists after sorting.
    fn rebuild_adjacency(&mut self) {
        self.adj_out.clear();
        self.adj_in.clear();

        for (idx, edge) in self.edges.iter().enumerate() {
            self.adj_out.entry(edge.src).or_default().push(idx);
            self.adj_in.entry(edge.dst).or_default().push(idx);
            self.adj_out.entry(edge.dst).or_default();
            self.adj_in.entry(edge.src).or_default();
        }
    }

    /// Get all edges in a time window [start, end].
    pub fn edges_in_window(&mut self, start: Timestamp, end: Timestamp) -> Vec<&TemporalEdge> {
        self.ensure_sorted();

        // Binary search for start
        let start_idx = self.edges.partition_point(|e| e.time < start);

        self.edges[start_idx..]
            .iter()
            .take_while(|e| e.time <= end)
            .collect()
    }

    /// Get outgoing edges from a node within a time window.
    pub fn outgoing_in_window(
        &mut self,
        node: u32,
        start: Timestamp,
        end: Timestamp,
    ) -> Vec<&TemporalEdge> {
        self.ensure_sorted();

        let Some(indices) = self.adj_out.get(&node) else {
            return vec![];
        };

        indices
            .iter()
            .map(|&i| &self.edges[i])
            .filter(|e| e.in_window(start, end))
            .collect()
    }

    /// Get incoming edges to a node within a time window.
    pub fn incoming_in_window(
        &mut self,
        node: u32,
        start: Timestamp,
        end: Timestamp,
    ) -> Vec<&TemporalEdge> {
        self.ensure_sorted();

        let Some(indices) = self.adj_in.get(&node) else {
            return vec![];
        };

        indices
            .iter()
            .map(|&i| &self.edges[i])
            .filter(|e| e.in_window(start, end))
            .collect()
    }

    /// Find temporal paths from src to dst respecting causality.
    ///
    /// A valid temporal path requires t(e1) < t(e2) < ... < t(en).
    pub fn temporal_paths(
        &mut self,
        src: u32,
        dst: u32,
        start: Timestamp,
        end: Timestamp,
        max_hops: usize,
    ) -> Vec<Vec<TemporalEdge>> {
        self.ensure_sorted();

        let mut paths = Vec::new();
        let mut current_path = Vec::new();

        self.dfs_temporal_paths(src, dst, start, end, max_hops, &mut current_path, &mut paths);

        paths
    }

    fn dfs_temporal_paths(
        &self,
        current: u32,
        dst: u32,
        min_time: Timestamp,
        end: Timestamp,
        remaining_hops: usize,
        current_path: &mut Vec<TemporalEdge>,
        paths: &mut Vec<Vec<TemporalEdge>>,
    ) {
        if current == dst && !current_path.is_empty() {
            paths.push(current_path.clone());
            return;
        }

        if remaining_hops == 0 {
            return;
        }

        let Some(indices) = self.adj_out.get(&current) else {
            return;
        };

        for &idx in indices {
            let edge = &self.edges[idx];

            // Must be after min_time (causality) and before end
            if edge.time > min_time && edge.time <= end {
                current_path.push(*edge);
                self.dfs_temporal_paths(
                    edge.dst,
                    dst,
                    edge.time,
                    end,
                    remaining_hops - 1,
                    current_path,
                    paths,
                );
                current_path.pop();
            }
        }
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.adj_out.len()
    }

    /// Time range of the graph.
    pub fn time_range(&self) -> (Timestamp, Timestamp) {
        (self.min_time, self.max_time)
    }

    /// Iterator over all edges.
    pub fn edges(&self) -> impl Iterator<Item = &TemporalEdge> {
        self.edges.iter()
    }

    /// Get all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = u32> + '_ {
        self.adj_out.keys().copied()
    }

    /// Out-degree of a node (total, not time-filtered).
    pub fn out_degree(&self, node: u32) -> usize {
        self.adj_out.get(&node).map_or(0, |v| v.len())
    }

    /// In-degree of a node (total, not time-filtered).
    pub fn in_degree(&self, node: u32) -> usize {
        self.adj_in.get(&node).map_or(0, |v| v.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_graph_basic() {
        let mut tg = TemporalGraph::new();

        tg.add_edge(TemporalEdge::new(0, 1, 100));
        tg.add_edge(TemporalEdge::new(1, 2, 200));
        tg.add_edge(TemporalEdge::new(2, 3, 300));

        assert_eq!(tg.num_edges(), 3);
        assert_eq!(tg.num_nodes(), 4);
        assert_eq!(tg.time_range(), (100, 300));
    }

    #[test]
    fn test_edges_in_window() {
        let mut tg = TemporalGraph::new();

        tg.add_edge(TemporalEdge::new(0, 1, 100));
        tg.add_edge(TemporalEdge::new(1, 2, 200));
        tg.add_edge(TemporalEdge::new(2, 3, 300));

        let edges = tg.edges_in_window(150, 250);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].time, 200);
    }

    #[test]
    fn test_temporal_paths() {
        let mut tg = TemporalGraph::new();

        // 0 -> 1 -> 2 with increasing times
        tg.add_edge(TemporalEdge::new(0, 1, 100));
        tg.add_edge(TemporalEdge::new(1, 2, 200));

        // Alternative path with wrong time order (won't work)
        tg.add_edge(TemporalEdge::new(0, 3, 300));
        tg.add_edge(TemporalEdge::new(3, 2, 150));  // This happens BEFORE 0->3

        let paths = tg.temporal_paths(0, 2, 0, 500, 3);

        // Only the 0->1->2 path should be valid
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), 2);
    }

    #[test]
    fn test_outgoing_in_window() {
        let mut tg = TemporalGraph::new();

        tg.add_edge(TemporalEdge::new(0, 1, 100));
        tg.add_edge(TemporalEdge::new(0, 2, 200));
        tg.add_edge(TemporalEdge::new(0, 3, 300));

        let edges = tg.outgoing_in_window(0, 150, 250);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].dst, 2);
    }
}
