//! Triple type for knowledge graphs.
//!
//! A triple represents a (subject, predicate, object) statement.

use crate::{EntityId, Error, RelationType, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A (subject, predicate, object) triple.
///
/// This is the fundamental unit of a knowledge graph.
///
/// # Example
///
/// ```rust
/// use lattice_core::Triple;
///
/// let triple = Triple::new("Apple", "founded_by", "Steve Jobs");
/// assert_eq!(triple.subject.as_str(), "Apple");
/// assert_eq!(triple.predicate.as_str(), "founded_by");
/// assert_eq!(triple.object.as_str(), "Steve Jobs");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Triple {
    /// Subject entity.
    pub subject: EntityId,

    /// Predicate (relation type).
    pub predicate: RelationType,

    /// Object entity.
    pub object: EntityId,

    /// Confidence score (0.0 to 1.0).
    pub confidence: Option<f32>,

    /// Source document or provenance.
    pub source: Option<String>,
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
    /// use lattice_core::Triple;
    ///
    /// let line = r#"<http://example.org/Apple> <http://example.org/founded_by> <http://example.org/Steve_Jobs> ."#;
    /// let triple = Triple::from_ntriples(line).unwrap();
    /// ```
    pub fn from_ntriples(line: &str) -> Result<Self> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return Err(Error::ParseTriple("Empty or comment line".into()));
        }

        // Simple N-Triples parser
        // Format: <subject> <predicate> <object> .
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut in_uri = false;
        let mut in_literal = false;
        let mut escape_next = false;

        for c in line.chars() {
            if escape_next {
                current.push(c);
                escape_next = false;
                continue;
            }

            match c {
                '\\' => {
                    escape_next = true;
                    current.push(c);
                }
                '<' if !in_literal => {
                    in_uri = true;
                }
                '>' if in_uri && !in_literal => {
                    in_uri = false;
                    parts.push(current.clone());
                    current.clear();
                }
                '"' if !in_uri => {
                    in_literal = !in_literal;
                    current.push(c);
                }
                '.' if !in_uri && !in_literal && current.is_empty() => {
                    // End of triple
                    break;
                }
                _ if in_uri || in_literal => {
                    current.push(c);
                }
                _ => {}
            }
        }

        if parts.len() < 3 {
            return Err(Error::InvalidNTriples(format!(
                "Expected 3 parts, got {}: {}",
                parts.len(),
                line
            )));
        }

        Ok(Self::new(parts[0].clone(), parts[1].clone(), parts[2].clone()))
    }

    /// Convert to N-Triples format.
    pub fn to_ntriples(&self) -> String {
        format!(
            "<{}> <{}> <{}> .",
            self.subject, self.predicate, self.object
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
        assert_eq!(t.subject.as_str(), "Apple");
        assert_eq!(t.predicate.as_str(), "founded_by");
        assert_eq!(t.object.as_str(), "Steve Jobs");
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

        assert_eq!(original.subject, parsed.subject);
        assert_eq!(original.predicate, parsed.predicate);
        assert_eq!(original.object, parsed.object);
    }

    #[test]
    fn test_parse_ntriples() {
        let line = r#"<http://example.org/Apple> <http://example.org/type> <http://example.org/Company> ."#;
        let triple = Triple::from_ntriples(line).unwrap();

        assert_eq!(triple.subject.as_str(), "http://example.org/Apple");
        assert_eq!(triple.predicate.as_str(), "http://example.org/type");
        assert_eq!(triple.object.as_str(), "http://example.org/Company");
    }
}
