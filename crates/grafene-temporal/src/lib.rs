//! Temporal graph primitives.
//!
//! Provides data structures and algorithms for time-evolving graphs:
//! - Temporal edges with timestamps
//! - Temporal queries (time windows, temporal paths)
//! - Temporal aggregation
//!
//! # Temporal Graph Models
//!
//! Two main models are supported:
//!
//! 1. **Discrete snapshots**: Graph at times t1, t2, ..., tn
//! 2. **Continuous edges**: Each edge has (start, end) timestamps
//!
//! # Example
//!
//! ```rust,ignore
//! use grafene_temporal::{TemporalGraph, TemporalEdge};
//!
//! let mut tg = TemporalGraph::new();
//!
//! // Add edges with timestamps
//! tg.add_edge(TemporalEdge::new(0, 1, 100));  // node 0 -> 1 at time 100
//! tg.add_edge(TemporalEdge::new(1, 2, 150));  // node 1 -> 2 at time 150
//!
//! // Query edges in time window
//! let edges = tg.edges_in_window(100, 200);
//!
//! // Find temporal paths (respecting causality: t1 < t2 < t3)
//! let paths = tg.temporal_paths(0, 2, 100, 200);
//! ```
//!
//! # Temporal Message Passing
//!
//! For GNN-style learning on temporal graphs, see the `message` module.

mod edge;
mod graph;
mod query;

pub use edge::{TemporalEdge, Timestamp};
pub use graph::TemporalGraph;
pub use query::{TemporalQuery, TimeWindow};

/// Error types for temporal operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid time range: start {start} > end {end}")]
    InvalidTimeRange { start: Timestamp, end: Timestamp },

    #[error("node not found: {0}")]
    NodeNotFound(u32),
}

/// Result type alias.
pub type Result<T> = std::result::Result<T, Error>;
