use crate::{Entity, EntityId, Error, Relation, RelationType, Result, Triple};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
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
/// use lattix::{KnowledgeGraph, Triple};
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

    /// Index: subject EntityId -> triple indices (for O(d) relations_from)
    subject_index: HashMap<EntityId, Vec<usize>>,

    /// Index: object EntityId -> triple indices (for O(d) relations_to)
    object_index: HashMap<EntityId, Vec<usize>>,

    /// Index: predicate RelationType -> triple indices (for O(d) triples_with_relation)
    predicate_index: HashMap<RelationType, Vec<usize>>,
}

/// Serde-only view of `KnowledgeGraph`.
///
/// We serialize the core storage (`graph`, `entity_index`, `triples`) and rebuild all derived
/// indexes on deserialize. This avoids a sharp edge where `#[serde(skip)]` indexes would be empty
/// after deserialization (silently breaking `relations_from` / `relations_to` / `relation_types`).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KnowledgeGraphSerde {
    graph: DiGraph<Entity, Relation>,
    entity_index: HashMap<EntityId, NodeIndex>,
    triples: Vec<Triple>,
}

#[derive(Serialize)]
struct KnowledgeGraphSer<'a> {
    graph: &'a DiGraph<Entity, Relation>,
    entity_index: &'a HashMap<EntityId, NodeIndex>,
    triples: &'a Vec<Triple>,
}

impl serde::Serialize for KnowledgeGraph {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        KnowledgeGraphSer {
            graph: &self.graph,
            entity_index: &self.entity_index,
            triples: &self.triples,
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for KnowledgeGraph {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = KnowledgeGraphSerde::deserialize(deserializer)?;
        let mut kg = Self {
            graph: raw.graph,
            entity_index: raw.entity_index,
            triples: raw.triples,
            subject_index: HashMap::new(),
            object_index: HashMap::new(),
            predicate_index: HashMap::new(),
        };
        kg.rebuild_indexes();
        Ok(kg)
    }
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
            predicate_index: HashMap::new(),
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
            predicate_index: HashMap::new(),
        }
    }

    /// Remove all entities, triples, and indexes.
    pub fn clear(&mut self) {
        self.graph.clear();
        self.entity_index.clear();
        self.triples.clear();
        self.subject_index.clear();
        self.object_index.clear();
        self.predicate_index.clear();
    }

    /// Rebuild indexes derived from `triples`.
    ///
    /// This is intended for deserialization paths that load `graph` + `triples` but skip
    /// the derived indexes.
    pub fn rebuild_indexes(&mut self) {
        self.subject_index.clear();
        self.object_index.clear();
        self.predicate_index.clear();

        for (idx, triple) in self.triples.iter().enumerate() {
            self.subject_index
                .entry(triple.subject().clone())
                .or_default()
                .push(idx);
            self.object_index
                .entry(triple.object().clone())
                .or_default()
                .push(idx);
            self.predicate_index
                .entry(triple.predicate().clone())
                .or_default()
                .push(idx);
        }
    }

    /// Leniently load from an N-Triples file, skipping lines that fail to parse.
    ///
    /// Use [`from_ntriples_file_strict`](Self::from_ntriples_file_strict) for fail-fast behavior,
    /// or [`from_ntriples_file_lenient`](Self::from_ntriples_file_lenient) to get a count of
    /// skipped lines.
    pub fn from_ntriples_file(path: impl AsRef<Path>) -> Result<Self> {
        let (kg, _skipped) = Self::from_ntriples_file_lenient(path)?;
        Ok(kg)
    }

    /// Leniently load from an N-Triples file, returning both the graph and the
    /// number of non-empty, non-comment lines that failed to parse.
    ///
    /// This makes the lenient behavior observable: callers can log or assert on
    /// the skip count without switching to strict mode.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use lattix::KnowledgeGraph;
    /// let (kg, skipped) = KnowledgeGraph::from_ntriples_file_lenient("data.nt")?;
    /// if skipped > 0 {
    ///     eprintln!("{} malformed lines skipped", skipped);
    /// }
    /// # Ok::<(), lattix::Error>(())
    /// ```
    pub fn from_ntriples_file_lenient(path: impl AsRef<Path>) -> Result<(Self, usize)> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut kg = Self::new();
        let mut skipped = 0usize;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            match Triple::from_ntriples(line) {
                Ok(triple) => kg.add_triple(triple),
                Err(_) => skipped += 1,
            }
        }

        Ok((kg, skipped))
    }

    /// Load from an N-Triples file, returning an error on the first unparseable line.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidNTriples`] with the line number and content if any
    /// non-empty, non-comment line fails to parse.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use lattix::KnowledgeGraph;
    /// let kg = KnowledgeGraph::from_ntriples_file_strict("data.nt")?;
    /// assert!(kg.triple_count() > 0);
    /// # Ok::<(), lattix::Error>(())
    /// ```
    pub fn from_ntriples_file_strict(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut kg = Self::new();

        for (line_no, line) in reader.lines().enumerate() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let triple = Triple::from_ntriples(line)
                .map_err(|_| Error::InvalidNTriples(format!("line {}: {}", line_no + 1, line)))?;
            kg.add_triple(triple);
        }

        Ok(kg)
    }

    /// Load from JSON adjacency list file.
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
        let mut kg: Self =
            bincode::deserialize_from(reader).map_err(|e| Error::Serialization(e.into()))?;
        kg.rebuild_indexes();
        Ok(kg)
    }

    /// Save to binary file (bincode).
    #[cfg(feature = "binary")]
    pub fn to_binary_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        bincode::serialize_into(&mut writer, self).map_err(|e| Error::Serialization(e.into()))?;
        Ok(())
    }

    /// Add a triple to the graph.
    ///
    /// A triple with `None` confidence is treated as having confidence `1.0`.
    ///
    /// Duplicate triples are stored, not deduplicated. Adding the same
    /// (subject, predicate, object) twice will increment `triple_count()` by 2
    /// and create two parallel edges in the underlying petgraph.
    pub fn add_triple(&mut self, triple: Triple) {
        // Ensure subject entity exists
        let subject_idx = self.get_or_create_entity(triple.subject());

        // Ensure object entity exists
        let object_idx = self.get_or_create_entity(triple.object());

        // Create relation
        let relation = Relation::new(triple.predicate().clone())
            .with_confidence(triple.confidence().unwrap_or(1.0));

        // Add edge
        self.graph.add_edge(subject_idx, object_idx, relation);

        // Update indexes
        let triple_idx = self.triples.len();
        self.subject_index
            .entry(triple.subject().clone())
            .or_default()
            .push(triple_idx);
        self.object_index
            .entry(triple.object().clone())
            .or_default()
            .push(triple_idx);
        self.predicate_index
            .entry(triple.predicate().clone())
            .or_default()
            .push(triple_idx);

        // Store triple
        self.triples.push(triple);
    }

    /// Remove a triple from the graph.
    ///
    /// Returns `true` if the triple was found and removed, `false` if it was not present.
    /// Entity nodes are kept even if they have no remaining edges (keeps `entity_index` stable).
    /// Uses `swap_remove` internally, so triple iteration order is not preserved.
    pub fn remove_triple(
        &mut self,
        subject: &EntityId,
        predicate: &RelationType,
        object: &EntityId,
    ) -> bool {
        // Find the triple index
        let triple_idx = match self.triples.iter().position(|t| {
            *t.subject() == *subject && *t.predicate() == *predicate && *t.object() == *object
        }) {
            Some(idx) => idx,
            None => return false,
        };

        let removed = self.triples.swap_remove(triple_idx);

        // Remove the corresponding edge from petgraph.
        // There may be parallel edges between the same nodes, so we must find the one
        // with the matching relation_type.
        if let (Some(&src_idx), Some(&dst_idx)) = (
            self.entity_index.get(subject),
            self.entity_index.get(object),
        ) {
            let mut edge_to_remove = None;
            for edge_ref in self.graph.edges_connecting(src_idx, dst_idx) {
                if edge_ref.weight().relation_type == *predicate {
                    edge_to_remove = Some(edge_ref.id());
                    break;
                }
            }
            if let Some(edge_id) = edge_to_remove {
                self.graph.remove_edge(edge_id);
            }
        }

        // Remove triple_idx from subject_index, object_index, and predicate_index
        if let Some(indices) = self.subject_index.get_mut(removed.subject()) {
            indices.retain(|&i| i != triple_idx);
        }
        if let Some(indices) = self.object_index.get_mut(removed.object()) {
            indices.retain(|&i| i != triple_idx);
        }
        if let Some(indices) = self.predicate_index.get_mut(removed.predicate()) {
            indices.retain(|&i| i != triple_idx);
            if indices.is_empty() {
                self.predicate_index.remove(removed.predicate());
            }
        }

        // If swap_remove moved the last element into triple_idx, update its index entries.
        // After swap_remove, self.triples.len() is the old last index.
        let swapped_from = self.triples.len(); // the old index of the element now at triple_idx
        if triple_idx < swapped_from {
            // An element was swapped: update its entries from swapped_from -> triple_idx
            let swapped = &self.triples[triple_idx];
            if let Some(indices) = self.subject_index.get_mut(swapped.subject()) {
                for i in indices.iter_mut() {
                    if *i == swapped_from {
                        *i = triple_idx;
                        break;
                    }
                }
            }
            if let Some(indices) = self.object_index.get_mut(swapped.object()) {
                for i in indices.iter_mut() {
                    if *i == swapped_from {
                        *i = triple_idx;
                        break;
                    }
                }
            }
            if let Some(indices) = self.predicate_index.get_mut(swapped.predicate()) {
                for i in indices.iter_mut() {
                    if *i == swapped_from {
                        *i = triple_idx;
                        break;
                    }
                }
            }
        }

        true
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

    /// Insert or update an entity by id.
    ///
    /// Unlike `update_entity`, this will create a new entity node if one does not already exist.
    pub fn upsert_entity(&mut self, entity: Entity) {
        let id = entity.id.clone();
        if let Some(&idx) = self.entity_index.get(&id) {
            self.graph[idx] = entity;
            return;
        }

        let idx = self.graph.add_node(entity);
        self.entity_index.insert(id, idx);
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

    /// Get all triples with a given relation type. O(d) where d is the
    /// number of triples with that predicate.
    pub fn triples_with_relation(&self, relation: impl Into<RelationType>) -> Vec<&Triple> {
        let relation = relation.into();
        match self.predicate_index.get(&relation) {
            Some(indices) => indices.iter().map(|&i| &self.triples[i]).collect(),
            None => vec![],
        }
    }

    /// Find a path between two entities, returning the sequence of triples (edges)
    /// along the shortest path. Returns `None` if no path exists.
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
                let mut t = Triple::new(
                    src_entity.id.clone(),
                    relation.relation_type.clone(),
                    dst_entity.id.clone(),
                );
                if let Some(c) = relation.confidence {
                    t = t.with_confidence(c);
                }
                if let Some(ref s) = relation.source {
                    t = t.with_source(s.clone());
                }
                triples.push(t);
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
        self.predicate_index.keys().collect()
    }

    /// Number of unique relation types. O(1).
    pub fn relation_type_count(&self) -> usize {
        self.predicate_index.len()
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

    /// Create a query builder for pattern-matching triples.
    ///
    /// # Example
    ///
    /// ```
    /// use lattix::{KnowledgeGraph, Triple};
    ///
    /// let mut kg = KnowledgeGraph::new();
    /// kg.add_triple(Triple::new("Alice", "knows", "Bob"));
    /// kg.add_triple(Triple::new("Alice", "works_at", "Acme"));
    ///
    /// let results = kg.query().subject("Alice").predicate("knows").execute();
    /// assert_eq!(results.len(), 1);
    /// ```
    pub fn query(&self) -> crate::query::TripleQuery<'_> {
        crate::query::TripleQuery::new(self)
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

// -- Graph composition and extraction operations --

impl KnowledgeGraph {
    /// Merge all triples from another knowledge graph into this one.
    ///
    /// Duplicate triples are not deduplicated (matching existing add_triple behavior).
    ///
    /// # Example
    ///
    /// ```
    /// use lattix::{KnowledgeGraph, Triple};
    ///
    /// let mut kg1 = KnowledgeGraph::new();
    /// kg1.add_triple(Triple::new("A", "r", "B"));
    ///
    /// let mut kg2 = KnowledgeGraph::new();
    /// kg2.add_triple(Triple::new("C", "r", "D"));
    ///
    /// kg1.merge(&kg2);
    /// assert_eq!(kg1.triple_count(), 2);
    /// assert_eq!(kg1.entity_count(), 4);
    /// ```
    pub fn merge(&mut self, other: &KnowledgeGraph) {
        for triple in other.triples() {
            self.add_triple(triple.clone());
        }
    }

    /// Return a new knowledge graph containing triples in `self` but not in `other`.
    ///
    /// Comparison is by (subject, predicate, object) — confidence and source are ignored.
    ///
    /// # Example
    ///
    /// ```
    /// use lattix::{KnowledgeGraph, Triple};
    ///
    /// let mut kg1 = KnowledgeGraph::new();
    /// kg1.add_triple(Triple::new("A", "r", "B"));
    /// kg1.add_triple(Triple::new("C", "r", "D"));
    ///
    /// let mut kg2 = KnowledgeGraph::new();
    /// kg2.add_triple(Triple::new("A", "r", "B"));
    ///
    /// let diff = kg1.diff(&kg2);
    /// assert_eq!(diff.triple_count(), 1);
    /// ```
    pub fn diff(&self, other: &KnowledgeGraph) -> KnowledgeGraph {
        let other_set: HashSet<(&str, &str, &str)> = other
            .triples()
            .map(|t| {
                (
                    t.subject().as_str(),
                    t.predicate().as_str(),
                    t.object().as_str(),
                )
            })
            .collect();

        let mut result = KnowledgeGraph::new();
        for triple in self.triples() {
            let key = (
                triple.subject().as_str(),
                triple.predicate().as_str(),
                triple.object().as_str(),
            );
            if !other_set.contains(&key) {
                result.add_triple(triple.clone());
            }
        }
        result
    }

    /// Return a new knowledge graph containing triples present in both `self` and `other`.
    ///
    /// Comparison is by (subject, predicate, object) — confidence and source are ignored.
    ///
    /// # Example
    ///
    /// ```
    /// use lattix::{KnowledgeGraph, Triple};
    ///
    /// let mut kg1 = KnowledgeGraph::new();
    /// kg1.add_triple(Triple::new("A", "r", "B"));
    /// kg1.add_triple(Triple::new("C", "r", "D"));
    ///
    /// let mut kg2 = KnowledgeGraph::new();
    /// kg2.add_triple(Triple::new("A", "r", "B"));
    /// kg2.add_triple(Triple::new("E", "r", "F"));
    ///
    /// let common = kg1.intersection(&kg2);
    /// assert_eq!(common.triple_count(), 1);
    /// ```
    pub fn intersection(&self, other: &KnowledgeGraph) -> KnowledgeGraph {
        let other_set: HashSet<(&str, &str, &str)> = other
            .triples()
            .map(|t| {
                (
                    t.subject().as_str(),
                    t.predicate().as_str(),
                    t.object().as_str(),
                )
            })
            .collect();

        let mut result = KnowledgeGraph::new();
        for triple in self.triples() {
            let key = (
                triple.subject().as_str(),
                triple.predicate().as_str(),
                triple.object().as_str(),
            );
            if other_set.contains(&key) {
                result.add_triple(triple.clone());
            }
        }
        result
    }

    /// Extract a k-hop subgraph around an entity.
    ///
    /// Returns a new KnowledgeGraph containing all triples reachable within `depth` hops
    /// from the given entity (following edges in both directions).
    ///
    /// # Example
    ///
    /// ```
    /// use lattix::{KnowledgeGraph, Triple};
    ///
    /// let mut kg = KnowledgeGraph::new();
    /// kg.add_triple(Triple::new("A", "r", "B"));
    /// kg.add_triple(Triple::new("B", "r", "C"));
    /// kg.add_triple(Triple::new("C", "r", "D"));
    ///
    /// let sub = kg.subgraph_around("A", 1);
    /// assert_eq!(sub.triple_count(), 1); // Only A->B
    /// assert_eq!(sub.entity_count(), 2);
    ///
    /// let sub = kg.subgraph_around("B", 1);
    /// assert_eq!(sub.triple_count(), 2); // A->B and B->C
    /// ```
    pub fn subgraph_around(&self, entity: impl Into<EntityId>, depth: usize) -> KnowledgeGraph {
        let entity = entity.into();
        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut frontier: VecDeque<(EntityId, usize)> = VecDeque::new();

        visited.insert(entity.clone());
        frontier.push_back((entity, 0));

        // BFS to discover entities within depth hops
        while let Some((current, d)) = frontier.pop_front() {
            if d >= depth {
                continue;
            }

            // Follow outgoing edges
            for triple in self.relations_from(current.as_str()) {
                if visited.insert(triple.object().clone()) {
                    frontier.push_back((triple.object().clone(), d + 1));
                }
            }

            // Follow incoming edges
            for triple in self.relations_to(current.as_str()) {
                if visited.insert(triple.subject().clone()) {
                    frontier.push_back((triple.subject().clone(), d + 1));
                }
            }
        }

        // Collect all triples where both endpoints are in the visited set
        let mut result = KnowledgeGraph::new();
        for triple in self.triples() {
            if visited.contains(triple.subject()) && visited.contains(triple.object()) {
                result.add_triple(triple.clone());
            }
        }
        result
    }

    /// Extract a subgraph containing only triples with the given predicates.
    ///
    /// # Example
    ///
    /// ```
    /// use lattix::{KnowledgeGraph, Triple};
    ///
    /// let mut kg = KnowledgeGraph::new();
    /// kg.add_triple(Triple::new("A", "knows", "B"));
    /// kg.add_triple(Triple::new("A", "works_at", "C"));
    ///
    /// let sub = kg.subgraph_by_predicates(&["knows"]);
    /// assert_eq!(sub.triple_count(), 1);
    /// ```
    pub fn subgraph_by_predicates(&self, predicates: &[impl AsRef<str>]) -> KnowledgeGraph {
        let pred_set: HashSet<&str> = predicates.iter().map(|p| p.as_ref()).collect();
        let mut result = KnowledgeGraph::new();
        for triple in self.triples() {
            if pred_set.contains(triple.predicate().as_str()) {
                result.add_triple(triple.clone());
            }
        }
        result
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

    #[test]
    fn test_remove_triple() {
        let mut kg = KnowledgeGraph::new();

        kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));
        kg.add_triple(Triple::new("Apple", "headquartered_in", "Cupertino"));
        kg.add_triple(Triple::new("Steve Jobs", "born_in", "San Francisco"));

        assert_eq!(kg.triple_count(), 3);

        // Remove existing triple
        let removed = kg.remove_triple(&"Apple".into(), &"founded_by".into(), &"Steve Jobs".into());
        assert!(removed);
        assert_eq!(kg.triple_count(), 2);

        // Entities should remain (not removed)
        assert_eq!(kg.entity_count(), 4);

        // relations_from should no longer include the removed triple
        let apple_rels = kg.relations_from("Apple");
        assert_eq!(apple_rels.len(), 1);
        assert_eq!(apple_rels[0].predicate().as_str(), "headquartered_in");

        // Removing nonexistent triple returns false
        let not_removed =
            kg.remove_triple(&"Apple".into(), &"founded_by".into(), &"Steve Jobs".into());
        assert!(!not_removed);
        assert_eq!(kg.triple_count(), 2);
    }

    #[test]
    fn test_remove_triple_updates_relation_cache() {
        let mut kg = KnowledgeGraph::new();

        kg.add_triple(Triple::new("A", "rel1", "B"));
        kg.add_triple(Triple::new("B", "rel2", "C"));
        assert_eq!(kg.relation_type_count(), 2);

        kg.remove_triple(&"A".into(), &"rel1".into(), &"B".into());
        assert_eq!(kg.relation_type_count(), 1);

        // rel2 should still be there
        let types: Vec<_> = kg
            .relation_types()
            .iter()
            .map(|r| r.as_str().to_string())
            .collect();
        assert!(types.contains(&"rel2".to_string()));
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

    // -- merge tests --

    #[test]
    fn test_merge_basic() {
        let mut kg1 = KnowledgeGraph::new();
        kg1.add_triple(Triple::new("A", "r", "B"));

        let mut kg2 = KnowledgeGraph::new();
        kg2.add_triple(Triple::new("C", "r", "D"));

        kg1.merge(&kg2);
        assert_eq!(kg1.triple_count(), 2);
        assert_eq!(kg1.entity_count(), 4);
    }

    #[test]
    fn test_merge_overlapping_entities() {
        let mut kg1 = KnowledgeGraph::new();
        kg1.add_triple(Triple::new("A", "r1", "B"));

        let mut kg2 = KnowledgeGraph::new();
        kg2.add_triple(Triple::new("B", "r2", "C"));

        kg1.merge(&kg2);
        assert_eq!(kg1.triple_count(), 2);
        assert_eq!(kg1.entity_count(), 3); // A, B, C -- B shared
    }

    // -- diff tests --

    #[test]
    fn test_diff_basic() {
        let mut kg1 = KnowledgeGraph::new();
        kg1.add_triple(Triple::new("A", "r", "B"));
        kg1.add_triple(Triple::new("C", "r", "D"));

        let mut kg2 = KnowledgeGraph::new();
        kg2.add_triple(Triple::new("A", "r", "B"));

        let diff = kg1.diff(&kg2);
        assert_eq!(diff.triple_count(), 1);
        assert!(diff.has_edge("C", "D"));
    }

    #[test]
    fn test_diff_empty() {
        let mut kg1 = KnowledgeGraph::new();
        kg1.add_triple(Triple::new("A", "r", "B"));

        let mut kg2 = KnowledgeGraph::new();
        kg2.add_triple(Triple::new("A", "r", "B"));

        let diff = kg1.diff(&kg2);
        assert_eq!(diff.triple_count(), 0);
    }

    #[test]
    fn test_diff_no_overlap() {
        let mut kg1 = KnowledgeGraph::new();
        kg1.add_triple(Triple::new("A", "r", "B"));

        let mut kg2 = KnowledgeGraph::new();
        kg2.add_triple(Triple::new("C", "r", "D"));

        let diff = kg1.diff(&kg2);
        assert_eq!(diff.triple_count(), 1);
        assert!(diff.has_edge("A", "B"));
    }

    // -- intersection tests --

    #[test]
    fn test_intersection_basic() {
        let mut kg1 = KnowledgeGraph::new();
        kg1.add_triple(Triple::new("A", "r", "B"));
        kg1.add_triple(Triple::new("C", "r", "D"));

        let mut kg2 = KnowledgeGraph::new();
        kg2.add_triple(Triple::new("A", "r", "B"));
        kg2.add_triple(Triple::new("E", "r", "F"));

        let common = kg1.intersection(&kg2);
        assert_eq!(common.triple_count(), 1);
        assert!(common.has_edge("A", "B"));
    }

    #[test]
    fn test_intersection_empty() {
        let mut kg1 = KnowledgeGraph::new();
        kg1.add_triple(Triple::new("A", "r", "B"));

        let mut kg2 = KnowledgeGraph::new();
        kg2.add_triple(Triple::new("C", "r", "D"));

        let common = kg1.intersection(&kg2);
        assert_eq!(common.triple_count(), 0);
    }

    #[test]
    fn test_intersection_full_overlap() {
        let mut kg1 = KnowledgeGraph::new();
        kg1.add_triple(Triple::new("A", "r", "B"));

        let mut kg2 = KnowledgeGraph::new();
        kg2.add_triple(Triple::new("A", "r", "B"));

        let common = kg1.intersection(&kg2);
        assert_eq!(common.triple_count(), 1);
    }

    // -- subgraph_around tests --

    #[test]
    fn test_subgraph_around_depth_0() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "r", "B"));

        let sub = kg.subgraph_around("A", 0);
        assert_eq!(sub.triple_count(), 0); // No hops, no triples
    }

    #[test]
    fn test_subgraph_around_depth_1() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "r", "B"));
        kg.add_triple(Triple::new("B", "r", "C"));
        kg.add_triple(Triple::new("C", "r", "D"));

        let sub = kg.subgraph_around("A", 1);
        assert_eq!(sub.triple_count(), 1); // A->B only
        assert_eq!(sub.entity_count(), 2);
    }

    #[test]
    fn test_subgraph_around_depth_2() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "r", "B"));
        kg.add_triple(Triple::new("B", "r", "C"));
        kg.add_triple(Triple::new("C", "r", "D"));

        let sub = kg.subgraph_around("A", 2);
        assert_eq!(sub.triple_count(), 2); // A->B and B->C
        assert_eq!(sub.entity_count(), 3);
    }

    #[test]
    fn test_subgraph_around_nonexistent_entity() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "r", "B"));

        let sub = kg.subgraph_around("Z", 2);
        assert_eq!(sub.triple_count(), 0);
        assert_eq!(sub.entity_count(), 0);
    }

    // -- subgraph_by_predicates tests --

    #[test]
    fn test_subgraph_by_predicates_single() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "knows", "B"));
        kg.add_triple(Triple::new("A", "works_at", "C"));

        let sub = kg.subgraph_by_predicates(&["knows"]);
        assert_eq!(sub.triple_count(), 1);
        assert!(sub.has_edge("A", "B"));
    }

    #[test]
    fn test_subgraph_by_predicates_multiple() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "knows", "B"));
        kg.add_triple(Triple::new("A", "works_at", "C"));
        kg.add_triple(Triple::new("B", "lives_in", "D"));

        let sub = kg.subgraph_by_predicates(&["knows", "lives_in"]);
        assert_eq!(sub.triple_count(), 2);
    }

    #[test]
    fn test_subgraph_by_predicates_no_matches() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "knows", "B"));

        let sub = kg.subgraph_by_predicates(&["nonexistent"]);
        assert_eq!(sub.triple_count(), 0);
    }
}
