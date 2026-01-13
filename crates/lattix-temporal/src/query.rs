//! Temporal query types and utilities.

use crate::edge::Timestamp;

/// A time window for temporal queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeWindow {
    /// Start time (inclusive).
    pub start: Timestamp,
    /// End time (inclusive).
    pub end: Timestamp,
}

impl TimeWindow {
    /// Create a new time window.
    pub fn new(start: Timestamp, end: Timestamp) -> Self {
        Self { start, end }
    }

    /// Create a window centered at a time with given radius.
    pub fn centered(center: Timestamp, radius: Timestamp) -> Self {
        Self {
            start: center.saturating_sub(radius),
            end: center.saturating_add(radius),
        }
    }

    /// Check if a timestamp is within this window.
    pub fn contains(&self, time: Timestamp) -> bool {
        time >= self.start && time <= self.end
    }

    /// Duration of the window.
    pub fn duration(&self) -> Timestamp {
        self.end.saturating_sub(self.start)
    }

    /// Check if two windows overlap.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Intersection of two windows, if any.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);

        if start <= end {
            Some(Self { start, end })
        } else {
            None
        }
    }
}

/// Temporal query specification.
#[derive(Debug, Clone)]
pub struct TemporalQuery {
    /// Source node (optional, for path queries).
    pub src: Option<u32>,
    /// Target node (optional, for path queries).
    pub dst: Option<u32>,
    /// Time window.
    pub window: TimeWindow,
    /// Maximum hops for path queries.
    pub max_hops: Option<usize>,
    /// Minimum time gap between consecutive edges.
    pub min_gap: Option<Timestamp>,
    /// Maximum time gap between consecutive edges.
    pub max_gap: Option<Timestamp>,
}

impl TemporalQuery {
    /// Create a basic window query.
    pub fn in_window(start: Timestamp, end: Timestamp) -> Self {
        Self {
            src: None,
            dst: None,
            window: TimeWindow::new(start, end),
            max_hops: None,
            min_gap: None,
            max_gap: None,
        }
    }

    /// Create a path query.
    pub fn path(src: u32, dst: u32, start: Timestamp, end: Timestamp) -> Self {
        Self {
            src: Some(src),
            dst: Some(dst),
            window: TimeWindow::new(start, end),
            max_hops: Some(10),
            min_gap: None,
            max_gap: None,
        }
    }

    /// Set maximum hops.
    pub fn with_max_hops(mut self, hops: usize) -> Self {
        self.max_hops = Some(hops);
        self
    }

    /// Set time gap constraints.
    pub fn with_gap_constraints(mut self, min: Timestamp, max: Timestamp) -> Self {
        self.min_gap = Some(min);
        self.max_gap = Some(max);
        self
    }
}

/// Temporal aggregation functions.
#[allow(dead_code)] // Public API for users of the crate
pub mod aggregate {
    use super::*;
    use crate::edge::TemporalEdge;

    /// Count edges in time window.
    pub fn count(edges: &[TemporalEdge], window: &TimeWindow) -> usize {
        edges.iter().filter(|e| window.contains(e.time)).count()
    }

    /// Count edges per time bucket.
    pub fn histogram(
        edges: &[TemporalEdge],
        window: &TimeWindow,
        bucket_size: Timestamp,
    ) -> Vec<(Timestamp, usize)> {
        let mut counts: std::collections::BTreeMap<Timestamp, usize> =
            std::collections::BTreeMap::new();

        for edge in edges {
            if window.contains(edge.time) {
                let bucket = (edge.time - window.start) / bucket_size * bucket_size + window.start;
                *counts.entry(bucket).or_default() += 1;
            }
        }

        counts.into_iter().collect()
    }

    /// Compute temporal degree (edges per node in window).
    pub fn temporal_degree(
        edges: &[TemporalEdge],
        window: &TimeWindow,
    ) -> std::collections::HashMap<u32, (usize, usize)> {
        let mut degrees: std::collections::HashMap<u32, (usize, usize)> =
            std::collections::HashMap::new();

        for edge in edges {
            if window.contains(edge.time) {
                degrees.entry(edge.src).or_default().0 += 1;  // out-degree
                degrees.entry(edge.dst).or_default().1 += 1;  // in-degree
            }
        }

        degrees
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_window() {
        let w = TimeWindow::new(100, 200);

        assert!(w.contains(100));
        assert!(w.contains(150));
        assert!(w.contains(200));
        assert!(!w.contains(99));
        assert!(!w.contains(201));
    }

    #[test]
    fn test_window_intersection() {
        let w1 = TimeWindow::new(100, 200);
        let w2 = TimeWindow::new(150, 250);
        let w3 = TimeWindow::new(300, 400);

        let i12 = w1.intersect(&w2).unwrap();
        assert_eq!(i12.start, 150);
        assert_eq!(i12.end, 200);

        assert!(w1.intersect(&w3).is_none());
    }

    #[test]
    fn test_centered_window() {
        let w = TimeWindow::centered(100, 50);
        assert_eq!(w.start, 50);
        assert_eq!(w.end, 150);
    }
}
