//! Lightweight triple types for KGE benchmarks.
//!
//! [`Triple`] uses bare strings (head/relation/tail) for dataset loading.
//! [`TripleIds`] uses integer IDs for training and evaluation hot paths.
//!
//! These are deliberately simpler than [`crate::Triple`] (which uses typed
//! [`EntityId`]/[`RelationType`] with optional confidence and provenance).
//! Use [`From`] conversions to bridge between them.

/// A raw triple with string identifiers.
///
/// The KGE community convention uses head/relation/tail naming (as opposed
/// to subject/predicate/object in the RDF world).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    /// Head entity name.
    pub head: String,
    /// Relation name.
    pub relation: String,
    /// Tail entity name.
    pub tail: String,
}

impl Triple {
    /// Create a new triple.
    pub fn new(
        head: impl Into<String>,
        relation: impl Into<String>,
        tail: impl Into<String>,
    ) -> Self {
        Self {
            head: head.into(),
            relation: relation.into(),
            tail: tail.into(),
        }
    }
}

/// Convert a KGE triple into a lattix core triple.
///
/// Maps head -> subject, relation -> predicate, tail -> object.
impl From<Triple> for crate::Triple {
    fn from(t: Triple) -> Self {
        crate::Triple::new(t.head, t.relation, t.tail)
    }
}

/// Convert a lattix core triple into a KGE triple.
///
/// Maps subject -> head, predicate -> relation, object -> tail.
impl From<crate::Triple> for Triple {
    fn from(t: crate::Triple) -> Self {
        Self::new(
            t.subject().as_str(),
            t.predicate().as_str(),
            t.object().as_str(),
        )
    }
}

/// A triple stored as interned integer IDs.
///
/// Indices reference the [`Vocab`](super::Vocab) in the parent
/// [`InternedDataset`](super::InternedDataset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TripleIds {
    /// Head entity ID.
    pub head: usize,
    /// Relation ID.
    pub relation: usize,
    /// Tail entity ID.
    pub tail: usize,
}

impl TripleIds {
    /// Create a new interned triple.
    pub fn new(head: usize, relation: usize, tail: usize) -> Self {
        Self {
            head,
            relation,
            tail,
        }
    }

    /// Return as a `(head, relation, tail)` tuple.
    pub fn as_tuple(&self) -> (usize, usize, usize) {
        (self.head, self.relation, self.tail)
    }
}
