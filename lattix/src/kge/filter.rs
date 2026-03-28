//! Filter index for filtered link prediction evaluation.
//!
//! The standard KGE evaluation protocol (Bordes et al., 2013) filters out
//! known-true triples when ranking candidates. This module provides the
//! index for that filtering.

use super::InternedDataset;
use std::collections::{HashMap, HashSet};

/// Pre-built index for filtered link prediction evaluation.
///
/// Indexes all known triples by `(head, relation)` for tail prediction
/// and by `(relation, tail)` for head prediction. Used to exclude
/// known-true entities from the ranking during evaluation.
///
/// # Example
///
/// ```
/// use lattix::kge::{Dataset, Triple, FilterIndex};
///
/// let ds = Dataset::new(
///     vec![Triple::new("a", "r", "b"), Triple::new("a", "r", "c")],
///     vec![Triple::new("b", "r", "c")],
///     vec![],
/// ).into_interned();
///
/// let filter = FilterIndex::from_dataset(&ds);
/// // When evaluating (a, r, ?), filter out b and c (both are known tails)
/// assert_eq!(filter.known_tails(0, 0).len(), 2);
/// ```
pub struct FilterIndex {
    /// (head, relation) -> set of known tail entities
    by_head_rel: HashMap<(usize, usize), HashSet<usize>>,
    /// (relation, tail) -> set of known head entities
    by_rel_tail: HashMap<(usize, usize), HashSet<usize>>,
    /// All known triples as a flat set for membership checks
    all: HashSet<(usize, usize, usize)>,
}

impl FilterIndex {
    /// Build a filter index from all splits of an interned dataset.
    ///
    /// Uses train + valid + test (the standard Bordes filtered protocol).
    pub fn from_dataset(ds: &InternedDataset) -> Self {
        let mut by_head_rel: HashMap<(usize, usize), HashSet<usize>> = HashMap::new();
        let mut by_rel_tail: HashMap<(usize, usize), HashSet<usize>> = HashMap::new();
        let mut all = HashSet::new();

        for t in ds.train.iter().chain(&ds.valid).chain(&ds.test) {
            by_head_rel
                .entry((t.head, t.relation))
                .or_default()
                .insert(t.tail);
            by_rel_tail
                .entry((t.relation, t.tail))
                .or_default()
                .insert(t.head);
            all.insert(t.as_tuple());
        }

        Self {
            by_head_rel,
            by_rel_tail,
            all,
        }
    }

    /// Known tail entities for a given (head, relation) pair.
    ///
    /// Returns an empty slice-like set if no tails are known.
    pub fn known_tails(&self, head: usize, relation: usize) -> &HashSet<usize> {
        static EMPTY: std::sync::LazyLock<HashSet<usize>> = std::sync::LazyLock::new(HashSet::new);
        self.by_head_rel.get(&(head, relation)).unwrap_or(&EMPTY)
    }

    /// Known head entities for a given (relation, tail) pair.
    ///
    /// Returns an empty set if no heads are known.
    pub fn known_heads(&self, relation: usize, tail: usize) -> &HashSet<usize> {
        static EMPTY: std::sync::LazyLock<HashSet<usize>> = std::sync::LazyLock::new(HashSet::new);
        self.by_rel_tail.get(&(relation, tail)).unwrap_or(&EMPTY)
    }

    /// Check whether a specific triple is known.
    pub fn contains(&self, head: usize, relation: usize, tail: usize) -> bool {
        self.all.contains(&(head, relation, tail))
    }

    /// Total number of known triples in the index.
    pub fn len(&self) -> usize {
        self.all.len()
    }

    /// True if the index contains no triples.
    pub fn is_empty(&self) -> bool {
        self.all.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kge::{Dataset, Triple};

    fn sample_dataset() -> InternedDataset {
        Dataset::new(
            vec![
                Triple::new("a", "r1", "b"),
                Triple::new("a", "r1", "c"),
                Triple::new("b", "r2", "c"),
            ],
            vec![Triple::new("a", "r2", "c")],
            vec![Triple::new("c", "r1", "a")],
        )
        .into_interned()
    }

    #[test]
    fn filter_index_known_tails() {
        let ds = sample_dataset();
        let filter = FilterIndex::from_dataset(&ds);

        let a = ds.entities.id("a").unwrap();
        let r1 = ds.relations.id("r1").unwrap();
        let tails = filter.known_tails(a, r1);
        assert_eq!(tails.len(), 2); // b and c
    }

    #[test]
    fn filter_index_known_heads() {
        let ds = sample_dataset();
        let filter = FilterIndex::from_dataset(&ds);

        let c = ds.entities.id("c").unwrap();
        let r1 = ds.relations.id("r1").unwrap();
        let heads = filter.known_heads(r1, c);
        assert_eq!(heads.len(), 1); // only a
    }

    #[test]
    fn filter_index_contains() {
        let ds = sample_dataset();
        let filter = FilterIndex::from_dataset(&ds);

        let a = ds.entities.id("a").unwrap();
        let b = ds.entities.id("b").unwrap();
        let r1 = ds.relations.id("r1").unwrap();
        assert!(filter.contains(a, r1, b));
        assert!(!filter.contains(b, r1, a));
    }

    #[test]
    fn filter_index_len() {
        let ds = sample_dataset();
        let filter = FilterIndex::from_dataset(&ds);
        assert_eq!(filter.len(), 5);
    }

    #[test]
    fn filter_index_unknown_pair_returns_empty() {
        let ds = sample_dataset();
        let filter = FilterIndex::from_dataset(&ds);
        assert!(filter.known_tails(999, 999).is_empty());
        assert!(filter.known_heads(999, 999).is_empty());
    }
}
