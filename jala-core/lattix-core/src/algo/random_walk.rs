//! `Node2Vec`-style random walk generation.
//!
//! Implements biased 2nd-order random walks as described in:
//! Grover & Leskovec, "node2vec: Scalable Feature Learning for Networks" (KDD 2016)
//!
//! ## Performance Notes
//!
//! - Uses rejection sampling for O(1) expected time per step (vs O(d^2) naive)
//! - Caches previous node's neighbors in `HashSet` for O(1) membership test
//! - Parallelized across walk iterations via rayon

use crate::KnowledgeGraph;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use std::collections::HashSet;

/// Configuration for random walks.
#[derive(Debug, Clone, Copy)]
pub struct RandomWalkConfig {
    /// Length of each random walk.
    pub walk_length: usize,
    /// Number of walks to start from each node.
    pub num_walks: usize,
    /// Return parameter (p) - likelihood of returning to previous node.
    /// - p > 1: less likely to backtrack
    /// - p < 1: more likely to backtrack
    pub p: f32,
    /// In-out parameter (q) - controls BFS vs DFS behavior.
    /// - q > 1: BFS-like (local exploration)
    /// - q < 1: DFS-like (outward exploration)
    pub q: f32,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for RandomWalkConfig {
    fn default() -> Self {
        Self {
            walk_length: 80,
            num_walks: 10,
            p: 1.0,
            q: 1.0,
            seed: 42,
        }
    }
}

/// Generate random walks for all nodes in the graph.
///
/// # Arguments
/// * `kg` - The Knowledge Graph
/// * `config` - Walk configuration
///
/// # Returns
/// A vector of walks, where each walk is a vector of entity IDs.
#[must_use]
pub fn generate_walks(kg: &KnowledgeGraph, config: RandomWalkConfig) -> Vec<Vec<String>> {
    let walker = Node2Vec::new(kg, config);
    walker.walk()
}

/// `Node2Vec` random walker.
pub struct Node2Vec<'a> {
    kg: &'a KnowledgeGraph,
    config: RandomWalkConfig,
}

impl<'a> Node2Vec<'a> {
    /// Create a new `Node2Vec` walker.
    #[must_use]
    pub const fn new(kg: &'a KnowledgeGraph, config: RandomWalkConfig) -> Self {
        Self { kg, config }
    }

    /// Generate all random walks using parallel processing.
    #[must_use]
    pub fn walk(&self) -> Vec<Vec<String>> {
        let node_indices: Vec<_> = self.kg.as_petgraph().node_indices().collect();
        let is_unbiased = (self.config.p - 1.0).abs() < f32::EPSILON
            && (self.config.q - 1.0).abs() < f32::EPSILON;

        (0..self.config.num_walks)
            .into_par_iter()
            .flat_map(|iter_idx| {
                let mut rng = XorShiftRng::seed_from_u64(self.config.seed + iter_idx as u64);
                let mut walks = Vec::with_capacity(node_indices.len());

                // Shuffle start nodes to avoid bias
                let mut shuffled = node_indices.clone();
                shuffled.shuffle(&mut rng);

                for &start in &shuffled {
                    let walk = if is_unbiased {
                        self.unbiased_walk(start, &mut rng)
                    } else {
                        self.biased_walk(start, &mut rng)
                    };
                    walks.push(walk);
                }
                walks
            })
            .collect()
    }

    /// Uniform random walk (`DeepWalk`) - O(1) per step.
    fn unbiased_walk<R: Rng>(&self, start: petgraph::graph::NodeIndex, rng: &mut R) -> Vec<String> {
        let graph = self.kg.as_petgraph();
        let mut walk = Vec::with_capacity(self.config.walk_length);
        walk.push(graph[start].id.0.clone());

        let mut curr = start;
        for _ in 1..self.config.walk_length {
            let neighbors: Vec<_> = graph.neighbors(curr).collect();
            if neighbors.is_empty() {
                break;
            }
            curr = *neighbors
                .choose(rng)
                .unwrap_or_else(|| panic!("neighbors cannot be empty (checked above)"));
            walk.push(graph[curr].id.0.clone());
        }
        walk
    }

    /// Biased 2nd-order random walk - O(1) expected per step via rejection sampling.
    fn biased_walk<R: Rng>(&self, start: petgraph::graph::NodeIndex, rng: &mut R) -> Vec<String> {
        let graph = self.kg.as_petgraph();
        let mut walk = Vec::with_capacity(self.config.walk_length);
        walk.push(graph[start].id.0.clone());

        let mut curr = start;
        let mut prev: Option<petgraph::graph::NodeIndex> = None;
        let mut prev_neighbors: HashSet<petgraph::graph::NodeIndex> = HashSet::new();

        for _ in 1..self.config.walk_length {
            let neighbors: Vec<_> = graph.neighbors(curr).collect();
            if neighbors.is_empty() {
                break;
            }

            let next = if let Some(prev_node) = prev {
                self.sample_biased_rejection(rng, prev_node, &prev_neighbors, &neighbors)
            } else {
                // First step: uniform
                *neighbors
                    .choose(rng)
                    .unwrap_or_else(|| panic!("neighbors should not be empty"))
            };

            walk.push(graph[next].id.0.clone());

            // Update state: cache current's neighbors as they become "prev_neighbors"
            prev = Some(curr);
            prev_neighbors.clear();
            prev_neighbors.extend(graph.neighbors(curr));
            curr = next;
        }
        walk
    }

    /// Sample next node using rejection sampling - O(1) expected time.
    ///
    /// The key insight: instead of computing weights for all neighbors (O(d)),
    /// we sample uniformly and accept/reject based on bias. Expected trials ~2-3.
    fn sample_biased_rejection<R: Rng>(
        &self,
        rng: &mut R,
        prev_node: petgraph::graph::NodeIndex,
        prev_neighbors: &HashSet<petgraph::graph::NodeIndex>,
        neighbors: &[petgraph::graph::NodeIndex],
    ) -> petgraph::graph::NodeIndex {
        let p = f64::from(self.config.p);
        let q = f64::from(self.config.q);

        // Acceptance probabilities (unnormalized)
        // - Return to prev: 1/p
        // - Move to prev's neighbor (triangle): 1
        // - Move away: 1/q
        let max_prob = (1.0 / p).max(1.0).max(1.0 / q);

        loop {
            let candidate = *neighbors
                .choose(rng)
                .unwrap_or_else(|| panic!("neighbors should not be empty (checked by caller)"));
            let r: f64 = rng.random();

            let unnorm_prob = if candidate == prev_node {
                1.0 / p // Backtrack
            } else if prev_neighbors.contains(&candidate) {
                1.0 // Triangle (dist=1 from prev)
            } else {
                1.0 / q // Move away (dist=2 from prev)
            };

            if r < unnorm_prob / max_prob {
                return candidate;
            }
            // Reject and retry
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn test_random_walk_uniform() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "A"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "B"));

        let config = RandomWalkConfig {
            walk_length: 10,
            num_walks: 2,
            p: 1.0,
            q: 1.0,
            seed: 42,
        };

        let walks = generate_walks(&kg, config);
        assert_eq!(walks.len(), 3 * 2); // 3 nodes * 2 walks
        for walk in &walks {
            assert!(!walk.is_empty());
        }
    }

    #[test]
    fn test_random_walk_biased() {
        let mut kg = KnowledgeGraph::new();
        // Create a small graph: A <-> B <-> C <-> D
        for (a, b) in [("A", "B"), ("B", "C"), ("C", "D")] {
            kg.add_triple(Triple::new(a, "rel", b));
            kg.add_triple(Triple::new(b, "rel", a));
        }

        let config = RandomWalkConfig {
            walk_length: 20,
            num_walks: 5,
            p: 0.5, // More likely to backtrack
            q: 2.0, // Less likely to explore
            seed: 123,
        };

        let walks = generate_walks(&kg, config);
        assert_eq!(walks.len(), 4 * 5); // 4 nodes * 5 walks

        // With p=0.5 (backtrack likely) and q=2.0 (outward unlikely),
        // walks should tend to stay local
        for walk in &walks {
            assert!(walk.len() > 1);
        }
    }

    #[test]
    fn test_random_walk_reproducible() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));

        let config = RandomWalkConfig {
            walk_length: 10,
            num_walks: 3,
            seed: 999,
            ..Default::default()
        };

        let walks1 = generate_walks(&kg, config);
        let walks2 = generate_walks(&kg, config);

        // Same seed should produce same walks
        assert_eq!(walks1, walks2);
    }
}
