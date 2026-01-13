//! Heterogeneous graph support.
//!
//! This module provides types for working with heterogeneous graphs (graphs with
//! multiple node types and edge types), similar to PyTorch Geometric's HeteroData.
//!
//! # Example
//!
//! ```rust
//! use grafene_core::hetero::{HeteroGraph, NodeType, EdgeType};
//!
//! let mut hg = HeteroGraph::new();
//!
//! // Add typed nodes
//! let user_type = NodeType::new("user");
//! let item_type = NodeType::new("item");
//!
//! hg.add_node(user_type.clone(), "alice");
//! hg.add_node(user_type.clone(), "bob");
//! hg.add_node(item_type.clone(), "book1");
//!
//! // Add typed edge
//! let buys = EdgeType::new("user", "buys", "item");
//! hg.add_edge(&buys, "alice", "book1");
//! hg.add_edge(&buys, "bob", "book1");
//!
//! assert_eq!(hg.num_node_types(), 2);
//! assert_eq!(hg.num_edge_types(), 1);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A node type identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeType(pub String);

impl NodeType {
    /// Create a new node type.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the type name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<S: Into<String>> From<S> for NodeType {
    fn from(s: S) -> Self {
        Self(s.into())
    }
}

/// An edge type identifier, represented as (src_type, relation, dst_type).
///
/// This is the "canonical" edge type representation used in PyG.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeType {
    /// Source node type.
    pub src_type: NodeType,
    /// Relation name.
    pub relation: String,
    /// Destination node type.
    pub dst_type: NodeType,
}

impl EdgeType {
    /// Create a new edge type.
    pub fn new(
        src_type: impl Into<NodeType>,
        relation: impl Into<String>,
        dst_type: impl Into<NodeType>,
    ) -> Self {
        Self {
            src_type: src_type.into(),
            relation: relation.into(),
            dst_type: dst_type.into(),
        }
    }

    /// Get the reverse edge type (for undirected edges).
    pub fn reverse(&self) -> Self {
        Self {
            src_type: self.dst_type.clone(),
            relation: format!("rev_{}", self.relation),
            dst_type: self.src_type.clone(),
        }
    }
}

/// Node index within a specific node type.
pub type TypedNodeIndex = usize;

/// Edge storage for a specific edge type (COO format).
///
/// Stores edges as (source_idx, target_idx) pairs where indices
/// are local to their respective node types.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EdgeStore {
    /// Source node indices (local to src_type).
    pub src: Vec<TypedNodeIndex>,
    /// Target node indices (local to dst_type).
    pub dst: Vec<TypedNodeIndex>,
}

impl EdgeStore {
    /// Create an empty edge store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from edge index vectors.
    pub fn from_edges(src: Vec<TypedNodeIndex>, dst: Vec<TypedNodeIndex>) -> Self {
        debug_assert_eq!(src.len(), dst.len());
        Self { src, dst }
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.src.len()
    }

    /// Add an edge.
    pub fn add_edge(&mut self, src: TypedNodeIndex, dst: TypedNodeIndex) {
        self.src.push(src);
        self.dst.push(dst);
    }

    /// Iterate over (src, dst) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (TypedNodeIndex, TypedNodeIndex)> + '_ {
        self.src.iter().copied().zip(self.dst.iter().copied())
    }
}

/// Node store for a specific node type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeStore {
    /// Node IDs (string identifiers).
    pub ids: Vec<String>,
    /// Map from ID to local index.
    id_to_idx: HashMap<String, TypedNodeIndex>,
}

impl NodeStore {
    /// Create an empty node store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.ids.len()
    }

    /// Add a node, returning its local index.
    pub fn add_node(&mut self, id: impl Into<String>) -> TypedNodeIndex {
        let id = id.into();
        if let Some(&idx) = self.id_to_idx.get(&id) {
            return idx;
        }
        let idx = self.ids.len();
        self.id_to_idx.insert(id.clone(), idx);
        self.ids.push(id);
        idx
    }

    /// Get a node's index by ID.
    pub fn get_index(&self, id: &str) -> Option<TypedNodeIndex> {
        self.id_to_idx.get(id).copied()
    }

    /// Get a node's ID by index.
    pub fn get_id(&self, idx: TypedNodeIndex) -> Option<&str> {
        self.ids.get(idx).map(|s| s.as_str())
    }

    /// Check if a node exists.
    pub fn contains(&self, id: &str) -> bool {
        self.id_to_idx.contains_key(id)
    }
}

/// A heterogeneous graph with typed nodes and edges.
///
/// Similar to PyTorch Geometric's HeteroData, this stores separate
/// node and edge stores for each type, allowing efficient typed queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HeteroGraph {
    /// Nodes by type.
    node_stores: HashMap<NodeType, NodeStore>,
    /// Edges by type.
    edge_stores: HashMap<EdgeType, EdgeStore>,
}

impl HeteroGraph {
    /// Create an empty heterogeneous graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with estimated capacity.
    pub fn with_capacity(node_types: usize, edge_types: usize) -> Self {
        Self {
            node_stores: HashMap::with_capacity(node_types),
            edge_stores: HashMap::with_capacity(edge_types),
        }
    }

    /// Number of node types.
    pub fn num_node_types(&self) -> usize {
        self.node_stores.len()
    }

    /// Number of edge types.
    pub fn num_edge_types(&self) -> usize {
        self.edge_stores.len()
    }

    /// Get all node types.
    pub fn node_types(&self) -> impl Iterator<Item = &NodeType> {
        self.node_stores.keys()
    }

    /// Get all edge types.
    pub fn edge_types(&self) -> impl Iterator<Item = &EdgeType> {
        self.edge_stores.keys()
    }

    /// Add a node of the given type.
    pub fn add_node(&mut self, node_type: NodeType, id: impl Into<String>) -> TypedNodeIndex {
        self.node_stores
            .entry(node_type)
            .or_insert_with(NodeStore::new)
            .add_node(id)
    }

    /// Add an edge of the given type.
    ///
    /// Automatically creates source/target nodes if they don't exist.
    pub fn add_edge(&mut self, edge_type: &EdgeType, src_id: &str, dst_id: &str) {
        // Ensure nodes exist
        let src_idx = self.add_node(edge_type.src_type.clone(), src_id);
        let dst_idx = self.add_node(edge_type.dst_type.clone(), dst_id);

        // Add edge
        self.edge_stores
            .entry(edge_type.clone())
            .or_insert_with(EdgeStore::new)
            .add_edge(src_idx, dst_idx);
    }

    /// Add bidirectional edge (adds reverse edge automatically).
    pub fn add_edge_bidirectional(&mut self, edge_type: &EdgeType, src_id: &str, dst_id: &str) {
        self.add_edge(edge_type, src_id, dst_id);
        self.add_edge(&edge_type.reverse(), dst_id, src_id);
    }

    /// Get node store for a type.
    pub fn node_store(&self, node_type: &NodeType) -> Option<&NodeStore> {
        self.node_stores.get(node_type)
    }

    /// Get edge store for a type.
    pub fn edge_store(&self, edge_type: &EdgeType) -> Option<&EdgeStore> {
        self.edge_stores.get(edge_type)
    }

    /// Get mutable node store for a type.
    pub fn node_store_mut(&mut self, node_type: &NodeType) -> Option<&mut NodeStore> {
        self.node_stores.get_mut(node_type)
    }

    /// Get mutable edge store for a type.
    pub fn edge_store_mut(&mut self, edge_type: &EdgeType) -> Option<&mut EdgeStore> {
        self.edge_stores.get_mut(edge_type)
    }

    /// Number of nodes of a given type.
    pub fn num_nodes(&self, node_type: &NodeType) -> usize {
        self.node_stores
            .get(node_type)
            .map(|s| s.num_nodes())
            .unwrap_or(0)
    }

    /// Number of edges of a given type.
    pub fn num_edges(&self, edge_type: &EdgeType) -> usize {
        self.edge_stores
            .get(edge_type)
            .map(|s| s.num_edges())
            .unwrap_or(0)
    }

    /// Total number of nodes across all types.
    pub fn total_nodes(&self) -> usize {
        self.node_stores.values().map(|s| s.num_nodes()).sum()
    }

    /// Total number of edges across all types.
    pub fn total_edges(&self) -> usize {
        self.edge_stores.values().map(|s| s.num_edges()).sum()
    }

    /// Get node index by type and ID.
    pub fn get_node_index(&self, node_type: &NodeType, id: &str) -> Option<TypedNodeIndex> {
        self.node_stores.get(node_type)?.get_index(id)
    }

    /// Get node ID by type and index.
    pub fn get_node_id(&self, node_type: &NodeType, idx: TypedNodeIndex) -> Option<&str> {
        self.node_stores.get(node_type)?.get_id(idx)
    }

    /// Get neighbors of a node via a specific edge type.
    pub fn neighbors(
        &self,
        edge_type: &EdgeType,
        src_idx: TypedNodeIndex,
    ) -> Vec<TypedNodeIndex> {
        self.edge_stores
            .get(edge_type)
            .map(|store| {
                store
                    .iter()
                    .filter_map(|(s, d)| if s == src_idx { Some(d) } else { None })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get incoming neighbors (reverse direction).
    pub fn incoming_neighbors(
        &self,
        edge_type: &EdgeType,
        dst_idx: TypedNodeIndex,
    ) -> Vec<TypedNodeIndex> {
        self.edge_stores
            .get(edge_type)
            .map(|store| {
                store
                    .iter()
                    .filter_map(|(s, d)| if d == dst_idx { Some(s) } else { None })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Compute metapath-based neighbors.
    ///
    /// A metapath is a sequence of edge types, e.g., ["author-writes-paper", "paper-cites-paper"].
    /// Returns nodes reachable via the full metapath from the source.
    pub fn metapath_neighbors(
        &self,
        _start_type: &NodeType,
        start_idx: TypedNodeIndex,
        metapath: &[EdgeType],
    ) -> HashSet<TypedNodeIndex> {
        let mut current: HashSet<TypedNodeIndex> = [start_idx].into_iter().collect();

        for edge_type in metapath {
            let mut next = HashSet::new();
            for &idx in &current {
                for neighbor in self.neighbors(edge_type, idx) {
                    next.insert(neighbor);
                }
            }
            current = next;
        }

        current
    }
}

/// Statistics for a heterogeneous graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeteroGraphStats {
    /// Number of node types.
    pub num_node_types: usize,
    /// Number of edge types.
    pub num_edge_types: usize,
    /// Total nodes.
    pub total_nodes: usize,
    /// Total edges.
    pub total_edges: usize,
    /// Nodes per type.
    pub nodes_by_type: HashMap<String, usize>,
    /// Edges per type.
    pub edges_by_type: HashMap<String, usize>,
}

impl HeteroGraph {
    /// Get statistics about the graph.
    pub fn stats(&self) -> HeteroGraphStats {
        HeteroGraphStats {
            num_node_types: self.num_node_types(),
            num_edge_types: self.num_edge_types(),
            total_nodes: self.total_nodes(),
            total_edges: self.total_edges(),
            nodes_by_type: self
                .node_stores
                .iter()
                .map(|(t, s)| (t.0.clone(), s.num_nodes()))
                .collect(),
            edges_by_type: self
                .edge_stores
                .iter()
                .map(|(t, s)| (format!("{}->{}:{}", t.src_type.0, t.dst_type.0, t.relation), s.num_edges()))
                .collect(),
        }
    }
}

/// Convert from homogeneous KnowledgeGraph to HeteroGraph.
///
/// Uses predicate as edge type, with a default "entity" node type.
impl From<&crate::KnowledgeGraph> for HeteroGraph {
    fn from(kg: &crate::KnowledgeGraph) -> Self {
        let mut hg = HeteroGraph::new();
        let entity_type = NodeType::new("entity");

        for triple in kg.triples() {
            let edge_type = EdgeType::new(
                entity_type.clone(),
                triple.predicate.as_str(),
                entity_type.clone(),
            );
            hg.add_edge(&edge_type, triple.subject.as_str(), triple.object.as_str());
        }

        hg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hetero_graph_basic() {
        let mut hg = HeteroGraph::new();

        let user = NodeType::new("user");
        let item = NodeType::new("item");

        hg.add_node(user.clone(), "alice");
        hg.add_node(user.clone(), "bob");
        hg.add_node(item.clone(), "book1");
        hg.add_node(item.clone(), "book2");

        assert_eq!(hg.num_node_types(), 2);
        assert_eq!(hg.num_nodes(&user), 2);
        assert_eq!(hg.num_nodes(&item), 2);
    }

    #[test]
    fn test_hetero_graph_edges() {
        let mut hg = HeteroGraph::new();

        let buys = EdgeType::new("user", "buys", "item");
        hg.add_edge(&buys, "alice", "book1");
        hg.add_edge(&buys, "bob", "book1");
        hg.add_edge(&buys, "alice", "book2");

        assert_eq!(hg.num_edge_types(), 1);
        assert_eq!(hg.num_edges(&buys), 3);

        // Check neighbors
        let alice_idx = hg.get_node_index(&NodeType::new("user"), "alice").unwrap();
        let neighbors = hg.neighbors(&buys, alice_idx);
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_hetero_graph_bidirectional() {
        let mut hg = HeteroGraph::new();

        let follows = EdgeType::new("user", "follows", "user");
        hg.add_edge_bidirectional(&follows, "alice", "bob");

        assert_eq!(hg.num_edge_types(), 2); // follows and rev_follows
        assert_eq!(hg.total_edges(), 2);
    }

    #[test]
    fn test_metapath() {
        let mut hg = HeteroGraph::new();

        // Create a simple citation network
        let writes = EdgeType::new("author", "writes", "paper");
        let cites = EdgeType::new("paper", "cites", "paper");

        hg.add_edge(&writes, "alice", "paper1");
        hg.add_edge(&writes, "bob", "paper2");
        // Add bidirectional cites so reverse is available
        hg.add_edge_bidirectional(&cites, "paper2", "paper1");

        // Find papers that cite papers alice wrote
        // metapath: alice -[writes]-> paper1 <-[rev_cites]- paper2
        let alice_idx = hg.get_node_index(&NodeType::new("author"), "alice").unwrap();
        let metapath = vec![writes.clone(), cites.reverse()];

        let reachable = hg.metapath_neighbors(&NodeType::new("author"), alice_idx, &metapath);
        // paper2 cites paper1, so rev_cites from paper1 leads to paper2
        assert_eq!(reachable.len(), 1);
    }

    #[test]
    fn test_from_knowledge_graph() {
        let mut kg = crate::KnowledgeGraph::new();
        kg.add_triple(crate::Triple::new("Alice", "knows", "Bob"));
        kg.add_triple(crate::Triple::new("Bob", "works_at", "Acme"));

        let hg = HeteroGraph::from(&kg);

        assert_eq!(hg.num_node_types(), 1); // All "entity"
        assert_eq!(hg.num_edge_types(), 2); // "knows" and "works_at"
        assert_eq!(hg.total_nodes(), 3);
        assert_eq!(hg.total_edges(), 2);
    }
}
