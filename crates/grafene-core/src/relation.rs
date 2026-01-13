//! Relation types for knowledge graphs.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A relation type (edge label) in a knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RelationType(pub String);

impl RelationType {
    /// Create a new relation type.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the relation type as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RelationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for RelationType {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for RelationType {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// A relation instance in a knowledge graph.
///
/// This represents an edge between two entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Relation {
    /// The relation type.
    pub relation_type: RelationType,

    /// Confidence score (0.0 to 1.0).
    pub confidence: Option<f32>,

    /// Source document or provenance.
    pub source: Option<String>,

    /// Additional properties.
    #[serde(default)]
    pub properties: std::collections::HashMap<String, serde_json::Value>,
}

impl Relation {
    /// Create a new relation.
    pub fn new(relation_type: impl Into<RelationType>) -> Self {
        Self {
            relation_type: relation_type.into(),
            confidence: None,
            source: None,
            properties: std::collections::HashMap::new(),
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
}

impl fmt::Display for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.relation_type)
    }
}
