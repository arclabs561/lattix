//! Neighbor sampling for Graph Neural Networks.
//!
//! Provides efficient mini-batch sampling for training GNNs like `GraphSAGE`.
//!
//! # Key Types
//!
//! - [`sample_neighbors`] - Sample up to k neighbors for each node (single hop)
//! - [`NeighborSampler`] - Multi-hop sampler for GraphSAGE-style training
//! - [`SubgraphBatch`] - Result of multi-hop sampling, contains induced subgraph

use crate::KnowledgeGraph;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use std::collections::{HashMap, HashSet};

/// Sample up to k neighbors for each node in the batch.
///
/// # Arguments
/// * `kg` - The knowledge graph
/// * `nodes` - Node IDs to sample neighbors for
/// * `k` - Maximum neighbors to sample per node
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Map of Node ID -> List of sampled neighbor IDs.
/// Nodes not in the graph get an empty list.
///
/// # Complexity
/// O(sum of min(k, degree) for all nodes)
#[must_use]
pub fn sample_neighbors(
    kg: &KnowledgeGraph,
    nodes: &[String],
    k: usize,
    seed: u64,
) -> HashMap<String, Vec<String>> {
    let mut rng = XorShiftRng::seed_from_u64(seed);
    let graph = kg.as_petgraph();
    let mut result = HashMap::with_capacity(nodes.len());

    for node_id in nodes {
        let entity_id = crate::EntityId::from(node_id.as_str());

        let neighbors = kg.get_node_index(&entity_id).map_or_else(Vec::new, |idx| {
            let all_neighbors: Vec<String> = graph
                .neighbors(idx)
                .map(|n_idx| graph[n_idx].id.0.clone())
                .collect();

            if all_neighbors.len() <= k {
                all_neighbors
            } else {
                all_neighbors
                    .choose_multiple(&mut rng, k)
                    .cloned()
                    .collect()
            }
        });

        result.insert(node_id.clone(), neighbors);
    }

    result
}

/// Sample neighbors with replacement (allows duplicates).
///
/// Useful when k > degree and you need exactly k samples.
///
/// # Complexity
/// O(k) per node regardless of degree
///
/// # Note
/// Returns empty vec for nodes with no neighbors (won't panic).
#[must_use]
pub fn sample_neighbors_with_replacement(
    kg: &KnowledgeGraph,
    nodes: &[String],
    k: usize,
    seed: u64,
) -> HashMap<String, Vec<String>> {
    let mut rng = XorShiftRng::seed_from_u64(seed);
    let graph = kg.as_petgraph();
    let mut result = HashMap::with_capacity(nodes.len());

    for node_id in nodes {
        let entity_id = crate::EntityId::from(node_id.as_str());

        let neighbors = kg.get_node_index(&entity_id).map_or_else(Vec::new, |idx| {
            let all_neighbors: Vec<_> = graph.neighbors(idx).collect();

            if all_neighbors.is_empty() {
                vec![]
            } else {
                (0..k)
                    .map(|_| {
                        // Safe: we checked is_empty above
                        let n_idx = *all_neighbors
                            .choose(&mut rng)
                            .unwrap_or_else(|| unreachable!("checked is_empty above"));
                        graph[n_idx].id.0.clone()
                    })
                    .collect()
            }
        });

        result.insert(node_id.clone(), neighbors);
    }

    result
}

/// A subgraph batch resulting from multi-hop neighbor sampling.
///
/// This is analogous to PyG's mini-batch structure for GraphSAGE.
/// Contains:
/// - The target (seed) nodes
/// - All sampled neighbor nodes per layer
/// - The induced edge structure
#[derive(Debug, Clone)]
pub struct SubgraphBatch {
    /// Target nodes (the original batch we want embeddings for).
    pub target_nodes: Vec<String>,
    /// All nodes in the subgraph (target + sampled neighbors).
    /// Ordered: target nodes first, then layer 1 neighbors, etc.
    pub all_nodes: Vec<String>,
    /// Map from node ID to index in all_nodes.
    pub node_to_idx: HashMap<String, usize>,
    /// Edges per layer: (source_idx, target_idx) pairs.
    /// Layer 0 = edges from layer 1 to targets.
    /// Layer i = edges from layer (i+1) to layer i.
    pub edges_per_layer: Vec<Vec<(usize, usize)>>,
    /// Number of nodes at each layer (including targets).
    pub layer_sizes: Vec<usize>,
}

impl SubgraphBatch {
    /// Number of layers (not counting targets).
    pub fn num_layers(&self) -> usize {
        self.edges_per_layer.len()
    }

    /// Total number of nodes in the batch.
    pub fn num_nodes(&self) -> usize {
        self.all_nodes.len()
    }

    /// Get indices of target nodes.
    pub fn target_indices(&self) -> impl Iterator<Item = usize> {
        0..self.target_nodes.len()
    }
}

/// Multi-hop neighbor sampler for GraphSAGE-style mini-batch training.
///
/// This samples a fixed number of neighbors at each hop, building up
/// a computation graph for efficient mini-batch GNN training.
///
/// # Example
///
/// ```rust
/// use lattix_core::{KnowledgeGraph, Triple};
/// use lattix_core::algo::sampling::NeighborSampler;
///
/// let mut kg = KnowledgeGraph::new();
/// kg.add_triple(Triple::new("A", "rel", "B"));
/// kg.add_triple(Triple::new("A", "rel", "C"));
/// kg.add_triple(Triple::new("B", "rel", "D"));
/// kg.add_triple(Triple::new("C", "rel", "D"));
///
/// // Sample 2 neighbors at each of 2 hops
/// let sampler = NeighborSampler::new(&kg, vec![2, 2]);
/// let batch = sampler.sample(&["A".to_string()], 42);
///
/// assert!(batch.all_nodes.contains(&"A".to_string()));
/// assert!(batch.num_layers() == 2);
/// ```
pub struct NeighborSampler<'a> {
    kg: &'a KnowledgeGraph,
    /// Number of neighbors to sample at each layer.
    /// fanout[0] = neighbors of targets, fanout[1] = 2-hop neighbors, etc.
    fanout: Vec<usize>,
}

impl<'a> NeighborSampler<'a> {
    /// Create a new sampler.
    ///
    /// # Arguments
    /// * `kg` - The knowledge graph
    /// * `fanout` - Number of neighbors to sample per layer (from targets outward)
    pub fn new(kg: &'a KnowledgeGraph, fanout: Vec<usize>) -> Self {
        Self { kg, fanout }
    }

    /// Sample a mini-batch subgraph starting from seed nodes.
    ///
    /// # Arguments
    /// * `seed_nodes` - Target nodes to sample neighborhoods for
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    /// A [`SubgraphBatch`] containing the induced subgraph.
    pub fn sample(&self, seed_nodes: &[String], seed: u64) -> SubgraphBatch {
        let mut rng = XorShiftRng::seed_from_u64(seed);
        let graph = self.kg.as_petgraph();

        // Track all nodes and their indices
        let mut all_nodes = Vec::new();
        let mut node_to_idx = HashMap::new();

        // Add target nodes first
        for node in seed_nodes {
            if !node_to_idx.contains_key(node) {
                node_to_idx.insert(node.clone(), all_nodes.len());
                all_nodes.push(node.clone());
            }
        }

        let target_count = all_nodes.len();
        let mut layer_sizes = vec![target_count];
        let mut edges_per_layer = Vec::new();

        // Current frontier of nodes to sample from
        let mut frontier: HashSet<String> = seed_nodes.iter().cloned().collect();

        // Sample each layer (outward from targets)
        for &num_neighbors in &self.fanout {
            let mut layer_edges = Vec::new();
            let mut next_frontier = HashSet::new();

            for node_id in &frontier {
                let src_idx = *node_to_idx
                    .get(node_id)
                    .expect("frontier node should exist");
                let entity_id = crate::EntityId::from(node_id.as_str());

                if let Some(node_idx) = self.kg.get_node_index(&entity_id) {
                    let all_neighbors: Vec<String> = graph
                        .neighbors(node_idx)
                        .map(|n_idx| graph[n_idx].id.0.clone())
                        .collect();

                    // Sample neighbors
                    let sampled: Vec<_> = if all_neighbors.len() <= num_neighbors {
                        all_neighbors
                    } else {
                        all_neighbors
                            .choose_multiple(&mut rng, num_neighbors)
                            .cloned()
                            .collect()
                    };

                    for neighbor in sampled {
                        // Get or create index for neighbor
                        let dst_idx = if let Some(&idx) = node_to_idx.get(&neighbor) {
                            idx
                        } else {
                            let idx = all_nodes.len();
                            node_to_idx.insert(neighbor.clone(), idx);
                            all_nodes.push(neighbor.clone());
                            idx
                        };

                        // Record edge from neighbor to source (message passing direction)
                        layer_edges.push((dst_idx, src_idx));
                        next_frontier.insert(neighbor);
                    }
                }
            }

            layer_sizes.push(all_nodes.len());
            edges_per_layer.push(layer_edges);
            frontier = next_frontier;
        }

        SubgraphBatch {
            target_nodes: seed_nodes.to_vec(),
            all_nodes,
            node_to_idx,
            edges_per_layer,
            layer_sizes,
        }
    }
}

/// Heterogeneous neighbor sampler for typed graphs.
///
/// Samples neighbors along specific edge types for heterogeneous GNNs.
pub struct HeteroNeighborSampler<'a> {
    kg: &'a crate::HeteroGraph,
    /// Fanout per edge type.
    fanout: HashMap<crate::EdgeType, Vec<usize>>,
}

impl<'a> HeteroNeighborSampler<'a> {
    /// Create a new heterogeneous sampler.
    ///
    /// # Arguments
    /// * `kg` - The heterogeneous graph
    /// * `fanout` - Map from edge type to layer-wise fanout
    pub fn new(kg: &'a crate::HeteroGraph, fanout: HashMap<crate::EdgeType, Vec<usize>>) -> Self {
        Self { kg, fanout }
    }

    /// Sample from specific seed nodes and node type.
    ///
    /// Returns a subgraph containing sampled neighbors via all specified edge types.
    pub fn sample(
        &self,
        seed_type: &crate::NodeType,
        seed_indices: &[usize],
        seed: u64,
    ) -> HeteroSubgraphBatch {
        let mut rng = XorShiftRng::seed_from_u64(seed);

        // Track sampled nodes per type
        let mut sampled_nodes: HashMap<crate::NodeType, Vec<usize>> = HashMap::new();
        sampled_nodes.insert(seed_type.clone(), seed_indices.to_vec());

        // Track edges per type
        let mut sampled_edges: HashMap<crate::EdgeType, Vec<(usize, usize)>> = HashMap::new();

        // For each edge type, sample neighbors
        for (edge_type, fanout_layers) in &self.fanout {
            let mut edges = Vec::new();
            let mut new_dst_nodes = Vec::new();

            // Get source nodes for this edge type (clone to avoid borrow conflict)
            let src_nodes: Vec<usize> = sampled_nodes
                .get(&edge_type.src_type)
                .cloned()
                .unwrap_or_default();

            for src_local_idx in src_nodes {
                // Sample neighbors via this edge type
                let neighbors = self.kg.neighbors(edge_type, src_local_idx);

                let num_sample = fanout_layers.first().copied().unwrap_or(0);
                let sampled: Vec<_> = if neighbors.len() <= num_sample {
                    neighbors
                } else {
                    neighbors
                        .choose_multiple(&mut rng, num_sample)
                        .copied()
                        .collect()
                };

                for dst_local_idx in sampled {
                    edges.push((src_local_idx, dst_local_idx));
                    new_dst_nodes.push(dst_local_idx);
                }
            }

            // Add destination nodes after the loop
            sampled_nodes
                .entry(edge_type.dst_type.clone())
                .or_default()
                .extend(new_dst_nodes);

            sampled_edges.insert(edge_type.clone(), edges);
        }

        // Deduplicate sampled nodes
        for nodes in sampled_nodes.values_mut() {
            nodes.sort_unstable();
            nodes.dedup();
        }

        HeteroSubgraphBatch {
            seed_type: seed_type.clone(),
            seed_indices: seed_indices.to_vec(),
            sampled_nodes,
            sampled_edges,
        }
    }
}

/// Heterogeneous subgraph batch.
#[derive(Debug, Clone)]
pub struct HeteroSubgraphBatch {
    /// Type of seed nodes.
    pub seed_type: crate::NodeType,
    /// Indices of seed nodes (local to their type).
    pub seed_indices: Vec<usize>,
    /// Sampled nodes per type.
    pub sampled_nodes: HashMap<crate::NodeType, Vec<usize>>,
    /// Sampled edges per type.
    pub sampled_edges: HashMap<crate::EdgeType, Vec<(usize, usize)>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn test_sample_neighbors() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("A", "rel", "C"));
        kg.add_triple(Triple::new("A", "rel", "D"));

        let result = sample_neighbors(&kg, &["A".into()], 2, 42);

        let sampled = result.get("A").unwrap();
        assert_eq!(sampled.len(), 2);
        for s in sampled {
            assert!(["B", "C", "D"].contains(&s.as_str()));
        }
    }

    #[test]
    fn test_sample_neighbors_all() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("A", "rel", "C"));

        // k > degree: should return all neighbors
        let result = sample_neighbors(&kg, &["A".into()], 10, 42);

        let sampled = result.get("A").unwrap();
        assert_eq!(sampled.len(), 2);
    }

    #[test]
    fn test_sample_neighbors_missing_node() {
        let kg = KnowledgeGraph::new();
        let result = sample_neighbors(&kg, &["NotExist".into()], 5, 42);

        assert!(result.get("NotExist").unwrap().is_empty());
    }

    #[test]
    fn test_sample_with_replacement() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));

        // k=5 but only 1 neighbor, with replacement should give 5
        let result = sample_neighbors_with_replacement(&kg, &["A".into()], 5, 42);

        let sampled = result.get("A").unwrap();
        assert_eq!(sampled.len(), 5);
        assert!(sampled.iter().all(|s| s == "B"));
    }

    #[test]
    fn test_neighbor_sampler_single_hop() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("A", "rel", "C"));
        kg.add_triple(Triple::new("A", "rel", "D"));

        let sampler = NeighborSampler::new(&kg, vec![2]);
        let batch = sampler.sample(&["A".to_string()], 42);

        assert_eq!(batch.target_nodes, vec!["A"]);
        assert!(batch.all_nodes.contains(&"A".to_string()));
        assert_eq!(batch.num_layers(), 1);
        // Target + 2 neighbors
        assert!(batch.all_nodes.len() <= 4);
        assert!(batch.all_nodes.len() >= 2);
    }

    #[test]
    fn test_neighbor_sampler_multi_hop() {
        let mut kg = KnowledgeGraph::new();
        // A -> B -> D
        // A -> C -> D
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("A", "rel", "C"));
        kg.add_triple(Triple::new("B", "rel", "D"));
        kg.add_triple(Triple::new("C", "rel", "D"));

        let sampler = NeighborSampler::new(&kg, vec![2, 2]);
        let batch = sampler.sample(&["A".to_string()], 42);

        assert_eq!(batch.num_layers(), 2);
        // Should have A, B, C, and potentially D
        assert!(batch.all_nodes.len() >= 3);
        assert!(batch.all_nodes.contains(&"A".to_string()));
    }
}
