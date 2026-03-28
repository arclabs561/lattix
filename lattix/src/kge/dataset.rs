//! Dataset types for KGE benchmark train/valid/test splits.

use super::{Triple, TripleIds};
use std::collections::HashMap;

/// Bidirectional string-to-integer vocabulary.
///
/// Assigns stable IDs to strings in insertion order. Provides O(1) lookup
/// in both directions.
#[derive(Debug, Clone, Default)]
pub struct Vocab {
    to_id: HashMap<String, usize>,
    from_id: Vec<String>,
}

impl Vocab {
    /// Number of interned items.
    pub fn len(&self) -> usize {
        self.from_id.len()
    }

    /// True if no items are interned.
    pub fn is_empty(&self) -> bool {
        self.from_id.is_empty()
    }

    /// Intern a string, returning its stable ID.
    ///
    /// If the string was already interned, returns the existing ID.
    pub fn intern(&mut self, s: String) -> usize {
        if let Some(&id) = self.to_id.get(&s) {
            return id;
        }
        let id = self.from_id.len();
        self.from_id.push(s.clone());
        self.to_id.insert(s, id);
        id
    }

    /// Get the string for an ID, or `None` if out of range.
    pub fn get(&self, id: usize) -> Option<&str> {
        self.from_id.get(id).map(|s| s.as_str())
    }

    /// Get the ID for a string, or `None` if not interned.
    pub fn id(&self, s: &str) -> Option<usize> {
        self.to_id.get(s).copied()
    }
}

/// Raw dataset with train/valid/test splits of string triples.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Training triples.
    pub train: Vec<Triple>,
    /// Validation triples.
    pub valid: Vec<Triple>,
    /// Test triples.
    pub test: Vec<Triple>,
}

impl Dataset {
    /// Create a new dataset from pre-split triple vectors.
    pub fn new(train: Vec<Triple>, valid: Vec<Triple>, test: Vec<Triple>) -> Self {
        Self { train, valid, test }
    }

    /// Total number of triples across all splits.
    pub fn len(&self) -> usize {
        self.train.len() + self.valid.len() + self.test.len()
    }

    /// True if all splits are empty.
    pub fn is_empty(&self) -> bool {
        self.train.is_empty() && self.valid.is_empty() && self.test.is_empty()
    }

    /// Collect all unique entity names across all splits.
    pub fn entities(&self) -> std::collections::HashSet<String> {
        let mut entities = std::collections::HashSet::new();
        for t in self.train.iter().chain(&self.valid).chain(&self.test) {
            entities.insert(t.head.clone());
            entities.insert(t.tail.clone());
        }
        entities
    }

    /// Collect all unique relation names across all splits.
    pub fn relations(&self) -> std::collections::HashSet<String> {
        let mut relations = std::collections::HashSet::new();
        for t in self.train.iter().chain(&self.valid).chain(&self.test) {
            relations.insert(t.relation.clone());
        }
        relations
    }

    /// Convert to an interned representation with integer IDs.
    ///
    /// Entities and relations are assigned IDs in first-encounter order
    /// (training triples first, then validation, then test).
    pub fn into_interned(self) -> InternedDataset {
        let mut entities = Vocab::default();
        let mut relations = Vocab::default();

        let mut intern = |t: Triple| -> TripleIds {
            let head = entities.intern(t.head);
            let relation = relations.intern(t.relation);
            let tail = entities.intern(t.tail);
            TripleIds::new(head, relation, tail)
        };

        let train = self.train.into_iter().map(&mut intern).collect();
        let valid = self.valid.into_iter().map(&mut intern).collect();
        let test = self.test.into_iter().map(&mut intern).collect();

        InternedDataset {
            train,
            valid,
            test,
            entities,
            relations,
        }
    }
}

/// Dataset with integer-interned entity and relation IDs.
///
/// This is the recommended form for training and evaluation hot paths.
/// String lookups go through the [`Vocab`] fields.
#[derive(Debug, Clone)]
pub struct InternedDataset {
    /// Training triples (interned IDs).
    pub train: Vec<TripleIds>,
    /// Validation triples (interned IDs).
    pub valid: Vec<TripleIds>,
    /// Test triples (interned IDs).
    pub test: Vec<TripleIds>,
    /// Entity vocabulary (ID <-> string).
    pub entities: Vocab,
    /// Relation vocabulary (ID <-> string).
    pub relations: Vocab,
}

impl InternedDataset {
    /// Number of unique entities.
    pub fn num_entities(&self) -> usize {
        self.entities.len()
    }

    /// Number of unique relations.
    pub fn num_relations(&self) -> usize {
        self.relations.len()
    }

    /// All triples across all splits as `(head, relation, tail)` tuples.
    pub fn all_triples(&self) -> Vec<(usize, usize, usize)> {
        self.train
            .iter()
            .chain(&self.valid)
            .chain(&self.test)
            .map(|t| t.as_tuple())
            .collect()
    }

    /// Create from pre-mapped integer arrays (for OGB or similar pipelines).
    ///
    /// Entity and relation names are auto-generated as `e0`, `e1`, ... and
    /// `r0`, `r1`, ...
    ///
    /// # Panics
    ///
    /// Panics if any entity ID >= `num_entities` or relation ID >= `num_relations`.
    pub fn from_arrays(
        train: &[(usize, usize, usize)],
        valid: &[(usize, usize, usize)],
        test: &[(usize, usize, usize)],
        num_entities: usize,
        num_relations: usize,
    ) -> Self {
        let check = |triples: &[(usize, usize, usize)], label: &str| {
            for (i, &(h, r, t)) in triples.iter().enumerate() {
                assert!(
                    h < num_entities && t < num_entities,
                    "{label}[{i}]: entity ID out of range (h={h}, t={t}, num_entities={num_entities})"
                );
                assert!(
                    r < num_relations,
                    "{label}[{i}]: relation ID out of range (r={r}, num_relations={num_relations})"
                );
            }
        };
        check(train, "train");
        check(valid, "valid");
        check(test, "test");

        let to_ids = |triples: &[(usize, usize, usize)]| -> Vec<TripleIds> {
            triples
                .iter()
                .map(|&(h, r, t)| TripleIds::new(h, r, t))
                .collect()
        };

        let mut entities = Vocab::default();
        for i in 0..num_entities {
            entities.intern(format!("e{i}"));
        }
        let mut relations = Vocab::default();
        for i in 0..num_relations {
            relations.intern(format!("r{i}"));
        }

        Self {
            train: to_ids(train),
            valid: to_ids(valid),
            test: to_ids(test),
            entities,
            relations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_roundtrip() {
        let mut v = Vocab::default();
        let id_a = v.intern("alice".into());
        let id_b = v.intern("bob".into());
        let id_a2 = v.intern("alice".into());
        assert_eq!(id_a, id_a2);
        assert_ne!(id_a, id_b);
        assert_eq!(v.get(id_a), Some("alice"));
        assert_eq!(v.id("bob"), Some(id_b));
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn dataset_into_interned() {
        let ds = Dataset::new(
            vec![Triple::new("a", "r", "b"), Triple::new("b", "r", "c")],
            vec![Triple::new("a", "r", "c")],
            vec![],
        );
        let interned = ds.into_interned();
        assert_eq!(interned.num_entities(), 3);
        assert_eq!(interned.num_relations(), 1);
        assert_eq!(interned.train.len(), 2);
        assert_eq!(interned.valid.len(), 1);

        let t0 = interned.train[0];
        assert_eq!(interned.entities.get(t0.head), Some("a"));
        assert_eq!(interned.relations.get(t0.relation), Some("r"));
        assert_eq!(interned.entities.get(t0.tail), Some("b"));
    }

    #[test]
    fn from_arrays_validates_and_roundtrips() {
        let train = vec![(0, 0, 1), (1, 0, 2)];
        let valid = vec![(0, 0, 2)];
        let test = vec![(2, 0, 0)];
        let ds = InternedDataset::from_arrays(&train, &valid, &test, 3, 1);

        assert_eq!(ds.num_entities(), 3);
        assert_eq!(ds.num_relations(), 1);
        assert_eq!(ds.all_triples().len(), 4);
        assert_eq!(ds.entities.get(0), Some("e0"));
    }

    #[test]
    #[should_panic(expected = "entity ID out of range")]
    fn from_arrays_rejects_bad_entity() {
        InternedDataset::from_arrays(&[(5, 0, 0)], &[], &[], 3, 1);
    }

    #[test]
    #[should_panic(expected = "relation ID out of range")]
    fn from_arrays_rejects_bad_relation() {
        InternedDataset::from_arrays(&[(0, 5, 0)], &[], &[], 3, 1);
    }

    #[test]
    fn dataset_entities_and_relations() {
        let ds = Dataset::new(
            vec![Triple::new("a", "r1", "b"), Triple::new("b", "r2", "c")],
            vec![],
            vec![],
        );
        assert_eq!(ds.entities().len(), 3);
        assert_eq!(ds.relations().len(), 2);
    }

    #[test]
    fn empty_dataset() {
        let ds = Dataset::new(vec![], vec![], vec![]);
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);
        let interned = ds.into_interned();
        assert_eq!(interned.num_entities(), 0);
        assert_eq!(interned.num_relations(), 0);
    }
}
