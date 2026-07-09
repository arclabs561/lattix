//! Triple type for knowledge graphs.
//!
//! A triple represents a (subject, predicate, object) statement.

use crate::{EntityId, RelationType, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A (subject, predicate, object) triple.
///
/// This is the fundamental unit of a knowledge graph.
///
/// # Example
///
/// ```rust
/// use lattix::Triple;
///
/// let triple = Triple::new("Apple", "founded_by", "Steve Jobs");
/// assert_eq!(triple.subject().as_str(), "Apple");
/// assert_eq!(triple.predicate().as_str(), "founded_by");
/// assert_eq!(triple.object().as_str(), "Steve Jobs");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Triple {
    /// Subject entity.
    subject: EntityId,

    /// Predicate (relation type).
    predicate: RelationType,

    /// Object entity.
    object: EntityId,

    /// Optional confidence score in `[0.0, 1.0]`. Defaults to `None`, which is
    /// treated as full confidence (`1.0`) when the triple is added to a
    /// [`KnowledgeGraph`].
    confidence: Option<f32>,

    /// Source document or provenance.
    source: Option<String>,
}

impl Triple {
    /// Create a new triple.
    pub fn new(
        subject: impl Into<EntityId>,
        predicate: impl Into<RelationType>,
        object: impl Into<EntityId>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            confidence: None,
            source: None,
        }
    }

    /// Get the subject entity.
    pub fn subject(&self) -> &EntityId {
        &self.subject
    }

    /// Get the predicate (relation type).
    pub fn predicate(&self) -> &RelationType {
        &self.predicate
    }

    /// Get the object entity.
    pub fn object(&self) -> &EntityId {
        &self.object
    }

    /// Get the confidence score.
    pub fn confidence(&self) -> Option<f32> {
        self.confidence
    }

    /// Get the source/provenance.
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Set confidence score.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Set source/provenance.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Parse from N-Triples format.
    ///
    /// Format: `<subject> <predicate> <object> .`
    ///
    /// # Example
    ///
    /// ```rust
    /// use lattix::Triple;
    ///
    /// let line = r#"<http://example.org/Apple> <http://example.org/founded_by> <http://example.org/Steve_Jobs> ."#;
    /// let triple = Triple::from_ntriples(line).unwrap();
    /// ```
    pub fn from_ntriples(line: &str) -> Result<Self> {
        crate::rdf::parse_ntriples_line(line)
    }

    /// Convert to N-Triples format.
    pub fn to_ntriples(&self) -> String {
        format!(
            "{} {} {} .",
            crate::rdf::render_iri_or_blank(self.subject.as_str()),
            crate::rdf::render_iri_or_blank(self.predicate.as_str()),
            crate::rdf::render_object(self.object.as_str())
        )
    }
}

impl fmt::Display for Triple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.subject, self.predicate, self.object)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_creation() {
        let t = Triple::new("Apple", "founded_by", "Steve Jobs");
        assert_eq!(t.subject().as_str(), "Apple");
        assert_eq!(t.predicate().as_str(), "founded_by");
        assert_eq!(t.object().as_str(), "Steve Jobs");
    }

    #[test]
    fn test_ntriples_roundtrip() {
        let original = Triple::new(
            "http://example.org/Apple",
            "http://example.org/founded_by",
            "http://example.org/Steve_Jobs",
        );

        let ntriples = original.to_ntriples();
        let parsed = Triple::from_ntriples(&ntriples).unwrap();

        assert_eq!(original.subject(), parsed.subject());
        assert_eq!(original.predicate(), parsed.predicate());
        assert_eq!(original.object(), parsed.object());
    }

    #[test]
    fn test_parse_ntriples() {
        let line = r#"<http://example.org/Apple> <http://example.org/type> <http://example.org/Company> ."#;
        let triple = Triple::from_ntriples(line).unwrap();

        assert_eq!(triple.subject().as_str(), "http://example.org/Apple");
        assert_eq!(triple.predicate().as_str(), "http://example.org/type");
        assert_eq!(triple.object().as_str(), "http://example.org/Company");
    }
}
