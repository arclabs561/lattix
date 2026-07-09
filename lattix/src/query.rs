use crate::{EntityId, KnowledgeGraph, RelationType, Triple};

/// A fluent query builder for matching triples by subject, predicate, and/or object.
///
/// # Example
///
/// ```
/// use lattix::{KnowledgeGraph, Triple};
///
/// let mut kg = KnowledgeGraph::new();
/// kg.add_triple(Triple::new("Alice", "knows", "Bob"));
/// kg.add_triple(Triple::new("Alice", "works_at", "Acme"));
/// kg.add_triple(Triple::new("Bob", "knows", "Charlie"));
///
/// // Find all triples where Alice is the subject
/// let results: Vec<_> = kg.query().subject("Alice").execute().collect();
/// assert_eq!(results.len(), 2);
///
/// // Find all "knows" triples
/// let results: Vec<_> = kg.query().predicate("knows").execute().collect();
/// assert_eq!(results.len(), 2);
///
/// // Find the specific triple
/// let results: Vec<_> = kg.query().subject("Alice").predicate("knows").execute().collect();
/// assert_eq!(results.len(), 1);
/// ```
pub struct TripleQuery<'a> {
    kg: &'a KnowledgeGraph,
    subject: Option<EntityId>,
    predicate: Option<RelationType>,
    object: Option<EntityId>,
}

impl<'a> TripleQuery<'a> {
    pub(crate) fn new(kg: &'a KnowledgeGraph) -> Self {
        Self {
            kg,
            subject: None,
            predicate: None,
            object: None,
        }
    }

    /// Filter by subject entity.
    pub fn subject(mut self, s: impl Into<EntityId>) -> Self {
        self.subject = Some(s.into());
        self
    }

    /// Filter by predicate (relation type).
    pub fn predicate(mut self, p: impl Into<RelationType>) -> Self {
        self.predicate = Some(p.into());
        self
    }

    /// Filter by object entity.
    pub fn object(mut self, o: impl Into<EntityId>) -> Self {
        self.object = Some(o.into());
        self
    }

    /// Execute the query, returning matching triples.
    ///
    /// Uses the most selective index available:
    /// - If subject is set, starts from `relations_from` (subject index)
    /// - If object is set (no subject), starts from `relations_to` (object index)
    /// - If only predicate is set, starts from `triples_with_relation` (predicate index)
    /// - If nothing is set, returns all triples
    pub fn execute(&self) -> impl Iterator<Item = &'a Triple> + '_ {
        let candidates = if let Some(ref s) = self.subject {
            CandidateIndices::from_slice(self.kg.subject_indices(s).unwrap_or(&[]))
        } else if let Some(ref o) = self.object {
            CandidateIndices::from_slice(self.kg.object_indices(o).unwrap_or(&[]))
        } else if let Some(ref p) = self.predicate {
            CandidateIndices::from_slice(self.kg.predicate_indices(p).unwrap_or(&[]))
        } else {
            CandidateIndices::All(self.kg.all_triple_indices())
        };

        candidates
            .filter_map(|idx| self.kg.triple_at_index(idx))
            .filter(|t| self.matches(t))
    }

    /// Count matching triples without collecting them.
    pub fn count(&self) -> usize {
        self.execute().count()
    }

    /// Check if any triple matches.
    pub fn exists(&self) -> bool {
        self.execute().next().is_some()
    }

    fn matches(&self, t: &Triple) -> bool {
        if let Some(ref s) = self.subject {
            if *t.subject() != *s {
                return false;
            }
        }
        if let Some(ref p) = self.predicate {
            if *t.predicate() != *p {
                return false;
            }
        }
        if let Some(ref o) = self.object {
            if *t.object() != *o {
                return false;
            }
        }
        true
    }
}

enum CandidateIndices<'a> {
    Indexed(std::slice::Iter<'a, usize>),
    All(std::ops::Range<usize>),
}

impl<'a> CandidateIndices<'a> {
    fn from_slice(indices: &'a [usize]) -> Self {
        Self::Indexed(indices.iter())
    }
}

impl Iterator for CandidateIndices<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Indexed(indices) => indices.next().copied(),
            Self::All(indices) => indices.next(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{KnowledgeGraph, Triple};

    fn sample_kg() -> KnowledgeGraph {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("Alice", "knows", "Bob"));
        kg.add_triple(Triple::new("Alice", "works_at", "Acme"));
        kg.add_triple(Triple::new("Bob", "knows", "Charlie"));
        kg.add_triple(Triple::new("Charlie", "works_at", "Acme"));
        kg
    }

    #[test]
    fn subject_only() {
        let kg = sample_kg();
        let results: Vec<_> = kg.query().subject("Alice").execute().collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn predicate_only() {
        let kg = sample_kg();
        let results: Vec<_> = kg.query().predicate("knows").execute().collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn object_only() {
        let kg = sample_kg();
        let results: Vec<_> = kg.query().object("Acme").execute().collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn subject_and_predicate() {
        let kg = sample_kg();
        let results: Vec<_> = kg
            .query()
            .subject("Alice")
            .predicate("knows")
            .execute()
            .collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object().as_str(), "Bob");
    }

    #[test]
    fn subject_and_object() {
        let kg = sample_kg();
        let results: Vec<_> = kg
            .query()
            .subject("Alice")
            .object("Bob")
            .execute()
            .collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].predicate().as_str(), "knows");
    }

    #[test]
    fn predicate_and_object() {
        let kg = sample_kg();
        let results: Vec<_> = kg
            .query()
            .predicate("works_at")
            .object("Acme")
            .execute()
            .collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn all_three_filters() {
        let kg = sample_kg();
        let results: Vec<_> = kg
            .query()
            .subject("Alice")
            .predicate("knows")
            .object("Bob")
            .execute()
            .collect();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn no_filters_returns_all() {
        let kg = sample_kg();
        let results: Vec<_> = kg.query().execute().collect();
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn no_matches() {
        let kg = sample_kg();
        assert!(kg.query().subject("Nobody").execute().next().is_none());
    }

    #[test]
    fn count_method() {
        let kg = sample_kg();
        assert_eq!(kg.query().predicate("knows").count(), 2);
    }

    #[test]
    fn exists_method() {
        let kg = sample_kg();
        assert!(kg.query().subject("Alice").exists());
        assert!(!kg.query().subject("Nobody").exists());
    }
}
