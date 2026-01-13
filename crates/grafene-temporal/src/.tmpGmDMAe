//! Temporal edge types.

/// Timestamp type (milliseconds since epoch or arbitrary units).
pub type Timestamp = u64;

/// A directed edge with a timestamp.
///
/// Represents an interaction or relationship that occurred at a specific time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemporalEdge {
    /// Source node ID.
    pub src: u32,
    /// Target node ID.
    pub dst: u32,
    /// Time when the edge occurred.
    pub time: Timestamp,
    /// Optional edge weight/label.
    pub weight: Option<i32>,
}

impl TemporalEdge {
    /// Create a new temporal edge.
    pub fn new(src: u32, dst: u32, time: Timestamp) -> Self {
        Self {
            src,
            dst,
            time,
            weight: None,
        }
    }

    /// Create with weight.
    pub fn with_weight(src: u32, dst: u32, time: Timestamp, weight: i32) -> Self {
        Self {
            src,
            dst,
            time,
            weight: Some(weight),
        }
    }

    /// Check if this edge is before another in time.
    pub fn before(&self, other: &Self) -> bool {
        self.time < other.time
    }

    /// Check if this edge is within a time window.
    pub fn in_window(&self, start: Timestamp, end: Timestamp) -> bool {
        self.time >= start && self.time <= end
    }
}

impl PartialOrd for TemporalEdge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TemporalEdge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.time.cmp(&other.time)
            .then_with(|| self.src.cmp(&other.src))
            .then_with(|| self.dst.cmp(&other.dst))
    }
}

/// A temporal edge with duration (interval).
///
/// Represents a relationship active over a time interval [start, end).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntervalEdge {
    /// Source node ID.
    pub src: u32,
    /// Target node ID.
    pub dst: u32,
    /// Start time (inclusive).
    pub start: Timestamp,
    /// End time (exclusive).
    pub end: Timestamp,
}

impl IntervalEdge {
    /// Create a new interval edge.
    pub fn new(src: u32, dst: u32, start: Timestamp, end: Timestamp) -> Self {
        Self { src, dst, start, end }
    }

    /// Check if edge is active at a given time.
    pub fn active_at(&self, time: Timestamp) -> bool {
        time >= self.start && time < self.end
    }

    /// Duration of the edge.
    pub fn duration(&self) -> Timestamp {
        self.end.saturating_sub(self.start)
    }

    /// Check if two interval edges overlap in time.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_edge_ordering() {
        let e1 = TemporalEdge::new(0, 1, 100);
        let e2 = TemporalEdge::new(0, 1, 200);
        let e3 = TemporalEdge::new(0, 2, 100);

        assert!(e1 < e2);  // Earlier time
        assert!(e1 < e3);  // Same time, different dst
    }

    #[test]
    fn test_temporal_edge_window() {
        let e = TemporalEdge::new(0, 1, 150);

        assert!(e.in_window(100, 200));
        assert!(e.in_window(150, 150));
        assert!(!e.in_window(100, 149));
        assert!(!e.in_window(151, 200));
    }

    #[test]
    fn test_interval_edge() {
        let e = IntervalEdge::new(0, 1, 100, 200);

        assert!(e.active_at(100));
        assert!(e.active_at(150));
        assert!(!e.active_at(200));  // End is exclusive
        assert!(!e.active_at(50));
        assert_eq!(e.duration(), 100);
    }

    #[test]
    fn test_interval_overlap() {
        let e1 = IntervalEdge::new(0, 1, 100, 200);
        let e2 = IntervalEdge::new(0, 1, 150, 250);
        let e3 = IntervalEdge::new(0, 1, 200, 300);

        assert!(e1.overlaps(&e2));  // [100,200) and [150,250) overlap
        assert!(!e1.overlaps(&e3)); // [100,200) and [200,300) don't overlap
    }
}
