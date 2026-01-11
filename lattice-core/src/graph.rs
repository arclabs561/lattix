//! Knowledge graph implementation using petgraph.

use crate::{Entity, EntityId, Relation, RelationType, Result, Triple};
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// A knowledge graph built from triples.
///
/// Uses petgraph's directed graph internally for efficient traversal.
///
/// # Example
///
/// ```rust
/// use lattice_core::{KnowledgeGraph, Triple};
///
/// let mut kg = KnowledgeGraph::new();
///
/// kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));
/// kg.add_triple(Triple::new("Apple", "headquartered_in", "Cupertino"));
///
/// assert_eq!(kg.entity_count(), 3);
/// assert_eq!(kg.triple_count(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct KnowledgeGraph {
    /// The underlying directed graph.
    graph: DiGraph<Entity, Relation>,

    /// Map from entity ID to node index.
    entity_index: HashMap<EntityId, NodeIndex>,

    /// All triples (for iteration and export).
    triples: Vec<Triple>,
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeGraph {
    /// Create an empty knowledge graph.
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            entity_index: HashMap::new(),
            triples: Vec::new(),
        }
    }

    /// Create a knowledge graph with estimated capacity.
    pub fn with_capacity(entities: usize, triples: usize) -> Self {
        Self {
            graph: DiGraph::with_capacity(entities, triples),
            entity_index: HashMap::with_capacity(entities),
            triples: Vec::with_capacity(triples),
        }
    }

    /// Load from N-Triples file.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let kg = KnowledgeGraph::from_ntriples_file("knowledge.nt")?;
    /// ```
    pub fn from_ntriples_file(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut kg = Self::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            match Triple::from_ntriples(line) {
                Ok(triple) => kg.add_triple(triple),
                Err(_) => {
                    // Skip malformed lines (could log warning)
                    continue;
                }
            }
        }

        Ok(kg)
    }

    /// Save to N-Triples file.
    pub fn to_ntriples_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;

        for triple in &self.triples {
            writeln!(file, "{}", triple.to_ntriples())?;
        }

        Ok(())
    }

    /// Add a triple to the graph.
    pub fn add_triple(&mut self, triple: Triple) {
        // Ensure subject entity exists
        let subject_idx = self.get_or_create_entity(&triple.subject);

        // Ensure object entity exists
        let object_idx = self.get_or_create_entity(&triple.object);

        // Create relation
        let relation = Relation::new(triple.predicate.clone())
            .with_confidence(triple.confidence.unwrap_or(1.0));

        // Add edge
        self.graph.add_edge(subject_idx, object_idx, relation);

        // Store triple
        self.triples.push(triple);
    }

    /// Get or create an entity node.
    fn get_or_create_entity(&mut self, id: &EntityId) -> NodeIndex {
        if let Some(&idx) = self.entity_index.get(id) {
            return idx;
        }

        let entity = Entity::new(id.clone());
        let idx = self.graph.add_node(entity);
        self.entity_index.insert(id.clone(), idx);
        idx
    }

    /// Get an entity by ID.
    pub fn get_entity(&self, id: &EntityId) -> Option<&Entity> {
        self.entity_index
            .get(id)
            .map(|&idx| &self.graph[idx])
    }

    /// Update an entity's metadata.
    pub fn update_entity(&mut self, id: &EntityId, entity: Entity) -> bool {
        if let Some(&idx) = self.entity_index.get(id) {
            self.graph[idx] = entity;
            true
        } else {
            false
        }
    }

    /// Get all triples where the given entity is the subject.
    pub fn relations_from(&self, subject: impl Into<EntityId>) -> Vec<&Triple> {
        let subject = subject.into();
        self.triples
            .iter()
            .filter(|t| t.subject == subject)
            .collect()
    }

    /// Get all triples where the given entity is the object.
    pub fn relations_to(&self, object: impl Into<EntityId>) -> Vec<&Triple> {
        let object = object.into();
        self.triples
            .iter()
            .filter(|t| t.object == object)
            .collect()
    }

    /// Get all triples with a given relation type.
    pub fn triples_with_relation(&self, relation: impl Into<RelationType>) -> Vec<&Triple> {
        let relation = relation.into();
        self.triples
            .iter()
            .filter(|t| t.predicate == relation)
            .collect()
    }

    /// Find a path between two entities.
    ///
    /// Returns the sequence of triples forming the shortest path, if one exists.
    pub fn find_path(
        &self,
        from: impl Into<EntityId>,
        to: impl Into<EntityId>,
    ) -> Option<Vec<Triple>> {
        let from = from.into();
        let to = to.into();

        let from_idx = self.entity_index.get(&from)?;
        let to_idx = self.entity_index.get(&to)?;

        // Use BFS to find shortest path
        use petgraph::algo::astar;

        let path = astar(
            &self.graph,
            *from_idx,
            |n| n == *to_idx,
            |_| 1,
            |_| 0,
        )?;

        // Convert path to triples
        let mut triples = Vec::new();
        let nodes: Vec<_> = path.1;

        for window in nodes.windows(2) {
            let (src, dst) = (window[0], window[1]);
            let src_entity = &self.graph[src];
            let dst_entity = &self.graph[dst];

            // Find the triple connecting these entities
            if let Some(triple) = self.triples.iter().find(|t| {
                t.subject == src_entity.id && t.object == dst_entity.id
            }) {
                triples.push(triple.clone());
            }
        }

        Some(triples)
    }

    /// Number of entities.
    pub fn entity_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of triples.
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    /// Iterate over all entities.
    pub fn entities(&self) -> impl Iterator<Item = &Entity> {
        self.graph.node_weights()
    }

    /// Iterate over all triples.
    pub fn triples(&self) -> impl Iterator<Item = &Triple> {
        self.triples.iter()
    }

    /// Get the underlying petgraph for advanced operations.
    pub fn as_petgraph(&self) -> &DiGraph<Entity, Relation> {
        &self.graph
    }

    /// Get all unique relation types in the graph.
    pub fn relation_types(&self) -> Vec<&RelationType> {
        let mut types: Vec<_> = self
            .triples
            .iter()
            .map(|t| &t.predicate)
            .collect();
        types.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        types.dedup();
        types
    }
}

/// Statistics about a knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphStats {
    /// Number of entities.
    pub entity_count: usize,
    /// Number of triples.
    pub triple_count: usize,
    /// Number of unique relation types.
    pub relation_type_count: usize,
    /// Average out-degree (relations per subject).
    pub avg_out_degree: f64,
}

impl KnowledgeGraph {
    /// Compute statistics about the graph.
    pub fn stats(&self) -> KnowledgeGraphStats {
        let entity_count = self.entity_count();
        let triple_count = self.triple_count();
        let relation_type_count = self.relation_types().len();

        let avg_out_degree = if entity_count > 0 {
            triple_count as f64 / entity_count as f64
        } else {
            0.0
        };

        KnowledgeGraphStats {
            entity_count,
            triple_count,
            relation_type_count,
            avg_out_degree,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_triples() {
        let mut kg = KnowledgeGraph::new();

        kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));
        kg.add_triple(Triple::new("Apple", "headquartered_in", "Cupertino"));
        kg.add_triple(Triple::new("Steve Jobs", "born_in", "San Francisco"));

        assert_eq!(kg.entity_count(), 4); // Apple, Steve Jobs, Cupertino, San Francisco
        assert_eq!(kg.triple_count(), 3);
    }

    #[test]
    fn test_relations_from() {
        let mut kg = KnowledgeGraph::new();

        kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));
        kg.add_triple(Triple::new("Apple", "headquartered_in", "Cupertino"));
        kg.add_triple(Triple::new("Microsoft", "founded_by", "Bill Gates"));

        let apple_relations = kg.relations_from("Apple");
        assert_eq!(apple_relations.len(), 2);
    }

    #[test]
    fn test_find_path() {
        let mut kg = KnowledgeGraph::new();

        kg.add_triple(Triple::new("A", "connects", "B"));
        kg.add_triple(Triple::new("B", "connects", "C"));
        kg.add_triple(Triple::new("C", "connects", "D"));

        let path = kg.find_path("A", "D");
        assert!(path.is_some());
        assert_eq!(path.unwrap().len(), 3);
    }

    #[test]
    fn test_stats() {
        let mut kg = KnowledgeGraph::new();

        kg.add_triple(Triple::new("A", "r1", "B"));
        kg.add_triple(Triple::new("A", "r2", "C"));
        kg.add_triple(Triple::new("B", "r1", "C"));

        let stats = kg.stats();
        assert_eq!(stats.entity_count, 3);
        assert_eq!(stats.triple_count, 3);
        assert_eq!(stats.relation_type_count, 2);
    }
}
