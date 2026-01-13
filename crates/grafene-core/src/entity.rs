//! Entity types for knowledge graphs.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for an entity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(pub String);

impl EntityId {
    /// Create a new entity ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for EntityId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for EntityId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// An entity (node) in a knowledge graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier.
    pub id: EntityId,

    /// Human-readable label.
    pub label: Option<String>,

    /// Entity type (e.g., "Person", "Organization").
    pub entity_type: Option<String>,

    /// Additional properties as key-value pairs.
    #[serde(default)]
    pub properties: std::collections::HashMap<String, serde_json::Value>,
}

impl Entity {
    /// Create a new entity with just an ID.
    pub fn new(id: impl Into<EntityId>) -> Self {
        Self {
            id: id.into(),
            label: None,
            entity_type: None,
            properties: std::collections::HashMap::new(),
        }
    }

    /// Set the label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the entity type.
    pub fn with_type(mut self, entity_type: impl Into<String>) -> Self {
        self.entity_type = Some(entity_type.into());
        self
    }

    /// Add a property.
    pub fn with_property(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }
}

impl fmt::Display for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref label) = self.label {
            write!(f, "{} ({})", label, self.id)
        } else {
            write!(f, "{}", self.id)
        }
    }
}
