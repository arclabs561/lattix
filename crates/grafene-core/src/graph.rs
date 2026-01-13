use crate::{Entity, EntityId, Relation, RelationType, Result, Triple};
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// A knowledge graph built from triples.
///
/// Uses petgraph's directed graph internally for efficient traversal.
/// Maintains indexes for O(1) entity lookup and O(d) relation queries.
///
/// # Example
///
/// ```rust
/// use grafene_core::{KnowledgeGraph, Triple};
///
/// let mut kg = KnowledgeGraph::new();
///
/// kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));
/// kg.add_triple(Triple::new("Apple", "headquartered_in", "Cupertino"));
///
/// assert_eq!(kg.entity_count(), 3);
/// assert_eq!(kg.triple_count(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// The underlying directed graph.
    graph: DiGraph<Entity, Relation>,

    /// Map from entity ID to node index.
    entity_index: HashMap<EntityId, NodeIndex>,

    /// All triples (for iteration and export).
    triples: Vec<Triple>,

    /// Index: subject EntityId -> triple indices (for O(d) relations_from)
    #[serde(skip, default)]
    subject_index: HashMap<EntityId, Vec<usize>>,

    /// Index: object EntityId -> triple indices (for O(d) relations_to)
    #[serde(skip, default)]
    object_index: HashMap<EntityId, Vec<usize>>,

    /// Cached relation types (for O(1) relation_types)
    #[serde(skip, default)]
    relation_type_cache: HashSet<RelationType>,
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
            subject_index: HashMap::new(),
            object_index: HashMap::new(),
            relation_type_cache: HashSet::new(),
        }
    }

    /// Create a knowledge graph with estimated capacity.
    pub fn with_capacity(entities: usize, triples: usize) -> Self {
        Self {
            graph: DiGraph::with_capacity(entities, triples),
            entity_index: HashMap::with_capacity(entities),
            triples: Vec::with_capacity(triples),
            subject_index: HashMap::with_capacity(entities),
            object_index: HashMap::with_capacity(entities),
            relation_type_cache: HashSet::new(),
        }
    }

    /// Rebuild indexes after deserialization.
    fn rebuild_indexes(&mut self) {
        self.subject_index.clear();
        self.object_index.clear();
        self.relation_type_cache.clear();

        for (idx, triple) in self.triples.iter().enumerate() {
            self.subject_index
                .entry(triple.subject.clone())
                .or_default()
                .push(idx);
            self.object_index
                .entry(triple.object.clone())
                .or_default()
                .push(idx);
            self.relation_type_cache.insert(triple.predicate.clone());
        }
    }

    /// Load from N-Triples file.
    pub fn from_ntriples_file(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut kg = Self::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Ok(triple) = Triple::from_ntriples(line) {
                kg.add_triple(triple);
            }
        }

        Ok(kg)
    }

    /// Load from JSON adjacency list file (decksage format).
    pub fn from_json_adjacency_file(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let adj: HashMap<String, Vec<String>> = serde_json::from_reader(reader)?;

        let mut kg = Self::new();
        kg.entity_index.reserve(adj.len());

        for (head, neighbors) in adj {
            for tail in neighbors {
                kg.add_triple(Triple::new(head.as_str(), "related_to", tail.as_str()));
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

    /// Load from binary file (bincode).
    #[cfg(feature = "binary")]
    pub fn from_binary_file(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut kg: Self = bincode::deserialize_from(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        kg.rebuild_indexes();
        Ok(kg)
    }

    /// Save to binary file (bincode).
    #[cfg(feature = "binary")]
    pub fn to_binary_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        bincode::serialize_into(&mut writer, self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
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

        // Update indexes
        let triple_idx = self.triples.len();
        self.subject_index
            .entry(triple.subject.clone())
            .or_default()
            .push(triple_idx);
        self.object_index
            .entry(triple.object.clone())
            .or_default()
            .push(triple_idx);
        self.relation_type_cache.insert(triple.predicate.clone());

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
        self.entity_index.get(id).map(|&idx| &self.graph[idx])
    }

    /// Get node index for an entity.
    pub fn get_node_index(&self, id: &EntityId) -> Option<NodeIndex> {
        self.entity_index.get(id).copied()
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
    /// O(d) where d is the out-degree of the entity.
    pub fn relations_from(&self, subject: impl Into<EntityId>) -> Vec<&Triple> {
        let subject = subject.into();
        match self.subject_index.get(&subject) {
            Some(indices) => indices.iter().map(|&i| &self.triples[i]).collect(),
            None => vec![],
        }
    }

    /// Get all triples where the given entity is the object.
    /// O(d) where d is the in-degree of the entity.
    pub fn relations_to(&self, object: impl Into<EntityId>) -> Vec<&Triple> {
        let object = object.into();
        match self.object_index.get(&object) {
            Some(indices) => indices.iter().map(|&i| &self.triples[i]).collect(),
            None => vec![],
        }
    }

    /// Get all triples with a given relation type.
    /// Note: This is still O(N) as we don't index by relation type.
    /// Consider adding a relation_index if this is a hot path.
    pub fn triples_with_relation(&self, relation: impl Into<RelationType>) -> Vec<&Triple> {
        let relation = relation.into();
        self.triples
            .iter()
            .filter(|t| t.predicate == relation)
            .collect()
    }

    /// Find a path between two entities.
    pub fn find_path(
        &self,
        from: impl Into<EntityId>,
        to: impl Into<EntityId>,
    ) -> Option<Vec<Triple>> {
        let from = from.into();
        let to = to.into();

        let from_idx = self.entity_index.get(&from)?;
        let to_idx = self.entity_index.get(&to)?;

        use petgraph::algo::astar;

        let path = astar(&self.graph, *from_idx, |n| n == *to_idx, |_| 1, |_| 0)?;

        // Convert path to triples using graph edges (O(d) per hop, not O(N))
        let mut triples = Vec::new();
        let nodes: Vec<_> = path.1;

        for window in nodes.windows(2) {
            let (src, dst) = (window[0], window[1]);

            // Find edge between src and dst
            if let Some(edge) = self.graph.find_edge(src, dst) {
                let relation = &self.graph[edge];
                let src_entity = &self.graph[src];
                let dst_entity = &self.graph[dst];
                triples.push(Triple {
                    subject: src_entity.id.clone(),
                    predicate: relation.relation_type.clone(),
                    object: dst_entity.id.clone(),
                    confidence: relation.confidence,
                    source: relation.source.clone(),
                });
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

    /// Get all unique relation types in the graph. O(1).
    pub fn relation_types(&self) -> Vec<&RelationType> {
        self.relation_type_cache.iter().collect()
    }

    /// Number of unique relation types. O(1).
    pub fn relation_type_count(&self) -> usize {
        self.relation_type_cache.len()
    }

    /// Get neighbors of an entity (outgoing edges). O(d).
    pub fn neighbors(&self, entity: impl Into<EntityId>) -> Vec<&Entity> {
        let entity = entity.into();
        match self.entity_index.get(&entity) {
            Some(&idx) => self.graph.neighbors(idx).map(|n| &self.graph[n]).collect(),
            None => vec![],
        }
    }

    /// Get neighbor IDs of an entity (outgoing edges). O(d).
    pub fn neighbor_ids(&self, entity: impl Into<EntityId>) -> Vec<&EntityId> {
        let entity = entity.into();
        match self.entity_index.get(&entity) {
            Some(&idx) => self
                .graph
                .neighbors(idx)
                .map(|n| &self.graph[n].id)
                .collect(),
            None => vec![],
        }
    }

    /// Check if an edge exists between two entities. O(d).
    pub fn has_edge(&self, from: impl Into<EntityId>, to: impl Into<EntityId>) -> bool {
        let from = from.into();
        let to = to.into();

        let Some(&from_idx) = self.entity_index.get(&from) else {
            return false;
        };
        let Some(&to_idx) = self.entity_index.get(&to) else {
            return false;
        };

        self.graph.find_edge(from_idx, to_idx).is_some()
    }

    /// Out-degree of an entity. O(1).
    pub fn out_degree(&self, entity: impl Into<EntityId>) -> usize {
        let entity = entity.into();
        match self.entity_index.get(&entity) {
            Some(&idx) => self.graph.neighbors(idx).count(),
            None => 0,
        }
    }

    /// In-degree of an entity. O(d) where d is in-degree.
    pub fn in_degree(&self, entity: impl Into<EntityId>) -> usize {
        let entity = entity.into();
        match self.object_index.get(&entity) {
            Some(indices) => indices.len(),
            None => 0,
        }
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
        let relation_type_count = self.relation_type_count();

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

        assert_eq!(kg.entity_count(), 4);
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

        // Verify O(d) behavior: relations_from should only return Apple's triples
        let ms_relations = kg.relations_from("Microsoft");
        assert_eq!(ms_relations.len(), 1);
    }

    #[test]
    fn test_relations_to() {
        let mut kg = KnowledgeGraph::new();

        kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));
        kg.add_triple(Triple::new("NeXT", "founded_by", "Steve Jobs"));
        kg.add_triple(Triple::new("Microsoft", "founded_by", "Bill Gates"));

        let jobs_relations = kg.relations_to("Steve Jobs");
        assert_eq!(jobs_relations.len(), 2);
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

    #[test]
    fn test_relation_types_cached() {
        let mut kg = KnowledgeGraph::new();

        kg.add_triple(Triple::new("A", "rel1", "B"));
        kg.add_triple(Triple::new("B", "rel2", "C"));
        kg.add_triple(Triple::new("C", "rel1", "D")); // Duplicate type

        assert_eq!(kg.relation_type_count(), 2);

        let types = kg.relation_types();
        assert_eq!(types.len(), 2);
    }

    #[test]
    fn test_neighbors() {
        let mut kg = KnowledgeGraph::new();

        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("A", "rel", "C"));
        kg.add_triple(Triple::new("B", "rel", "D"));

        let a_neighbors = kg.neighbor_ids("A");
        assert_eq!(a_neighbors.len(), 2);

        assert_eq!(kg.out_degree("A"), 2);
        assert_eq!(kg.in_degree("B"), 1);
    }

    #[test]
    fn test_has_edge() {
        let mut kg = KnowledgeGraph::new();

        kg.add_triple(Triple::new("A", "rel", "B"));

        assert!(kg.has_edge("A", "B"));
        assert!(!kg.has_edge("B", "A"));
        assert!(!kg.has_edge("A", "C"));
    }

    #[cfg(feature = "binary")]
    #[test]
    fn test_binary_roundtrip() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "r", "B"));

        let path = std::env::temp_dir().join("test_kg.bin");
        kg.to_binary_file(&path).unwrap();

        let loaded = KnowledgeGraph::from_binary_file(&path).unwrap();
        assert_eq!(loaded.entity_count(), 2);
        assert_eq!(loaded.triple_count(), 1);

        // Verify indexes were rebuilt
        assert_eq!(loaded.relations_from("A").len(), 1);
        assert_eq!(loaded.relation_type_count(), 1);

        std::fs::remove_file(path).unwrap();
    }
}
