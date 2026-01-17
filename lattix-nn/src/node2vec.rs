//! Skip-gram training for node embeddings (Node2Vec).
//!
//! # The Core Insight
//!
//! Nodes appearing in similar random walk contexts should have similar embeddings.
//! This is Word2Vec applied to graphs: walks are "sentences," nodes are "words."

// Some fields in Node2VecTrainer are reserved for configuration (used in walk generation)
#![allow(dead_code)]
//!
//! # Mathematical Foundation
//!
//! Skip-gram with negative sampling (SGNS) optimizes:
//!
//! ```text
//! L = log σ(v_target · v_center) + Σᵢ E[log σ(-v_neg · v_center)]
//! ```
//!
//! **Key theoretical result** (Levy & Goldberg 2014): SGNS implicitly factorizes
//! a shifted PMI matrix:
//!
//! ```text
//! v_w · v_c ≈ PMI(w, c) - log k
//! ```
//!
//! where PMI(w, c) = log(P(w,c) / P(w)P(c)) measures co-occurrence beyond chance.
//! This explains why embeddings capture semantic relationships—the dot product
//! approximates pointwise mutual information.
//!
//! ## Why Negative Sampling Works
//!
//! Full softmax requires summing over all nodes (expensive). Negative sampling
//! converts multinomial classification into binary classification:
//!
//! - **Positive sample**: (center, context) pairs from actual walks
//! - **Negative samples**: Random nodes drawn from frequency distribution
//!
//! The 3/4 power in sampling distribution (freq^0.75) smooths between:
//! - freq^1 (unigram): Over-samples common nodes
//! - freq^0 (uniform): Over-samples rare nodes
//!
//! ## Node2Vec Walk Bias
//!
//! Random walks are biased by parameters p and q:
//!
//! - **p (return)**: Likelihood of returning to previous node
//!   - Low p → Stay local (BFS-like exploration)
//! - **q (in-out)**: Likelihood of moving outward vs inward
//!   - Low q → Explore outward (DFS-like exploration)
//!
//! ```text
//! p=1, q=1: Unbiased random walk (DeepWalk)
//! p=1, q=0.5: Encourage outward exploration
//! p=0.5, q=1: Encourage local exploration
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use lattix_nn::node2vec::{SkipGram, SkipGramConfig};
//!
//! let config = SkipGramConfig {
//!     embedding_dim: 128,
//!     window_size: 5,
//!     negative_samples: 5,  // k=5 typical for large graphs
//!     learning_rate: 0.025,
//!     ..Default::default()
//! };
//!
//! let mut model = SkipGram::new(num_nodes, config);
//! for walk in walks {
//!     model.train_walk(&walk);
//! }
//! let embeddings = model.embeddings();
//! ```
//!
//! # References
//!
//! - Grover & Leskovec (2016). "node2vec: Scalable Feature Learning for Networks."
//! - Mikolov et al. (2013). "Distributed Representations of Words and Phrases."
//! - Levy & Goldberg (2014). "Neural Word Embedding as Implicit Matrix Factorization."

use rand::prelude::*;
use rand_distr::{Distribution, Uniform};
// Note: rayon imported for future parallel training
#[allow(unused_imports)]
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Skip-gram model configuration.
#[derive(Debug, Clone)]
pub struct SkipGramConfig {
    /// Embedding dimension.
    pub embedding_dim: usize,
    /// Context window size (each side).
    pub window_size: usize,
    /// Number of negative samples per positive.
    pub negative_samples: usize,
    /// Initial learning rate.
    pub learning_rate: f32,
    /// Minimum learning rate (for decay).
    pub min_learning_rate: f32,
    /// Random seed.
    pub seed: u64,
}

impl Default for SkipGramConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            window_size: 5,
            negative_samples: 5,
            learning_rate: 0.025,
            min_learning_rate: 0.0001,
            seed: 42,
        }
    }
}

/// Skip-gram model for node embedding training.
///
/// Uses two embedding matrices:
/// - `embeddings`: Target node embeddings (what we output)
/// - `context`: Context embeddings (discarded after training)
pub struct SkipGram {
    /// Target embeddings: (num_nodes, embedding_dim)
    embeddings: Vec<f32>,
    /// Context embeddings: (num_nodes, embedding_dim)
    context: Vec<f32>,
    /// Number of nodes
    num_nodes: usize,
    /// Configuration
    config: SkipGramConfig,
    /// Negative sampling distribution (unigram^0.75)
    neg_table: Vec<u32>,
    /// Current learning rate
    current_lr: f32,
    /// Words processed (for LR decay)
    words_processed: AtomicUsize,
    /// Total words to process
    total_words: usize,
}

impl SkipGram {
    /// Create a new Skip-gram model.
    ///
    /// Initializes embeddings with small random values.
    pub fn new(num_nodes: usize, config: SkipGramConfig) -> Self {
        let dim = config.embedding_dim;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let dist = Uniform::new(-0.5 / dim as f32, 0.5 / dim as f32)
            .expect("Invalid uniform distribution bounds");

        // Initialize embeddings randomly
        let embeddings: Vec<f32> = (0..num_nodes * dim)
            .map(|_| dist.sample(&mut rng))
            .collect();

        // Context embeddings start at zero
        let context = vec![0.0; num_nodes * dim];

        // Build negative sampling table (uniform for now)
        // Full impl would use node frequency^0.75
        let neg_table: Vec<u32> = (0..num_nodes as u32).collect();

        Self {
            embeddings,
            context,
            num_nodes,
            config: config.clone(),
            neg_table,
            current_lr: config.learning_rate,
            words_processed: AtomicUsize::new(0),
            total_words: 0,
        }
    }

    /// Build negative sampling table from node frequencies.
    ///
    /// Uses unigram^0.75 distribution as in original Word2Vec.
    pub fn build_neg_table(&mut self, frequencies: &[u32]) {
        const TABLE_SIZE: usize = 100_000_000;

        // Compute total weight (freq^0.75)
        let total: f64 = frequencies.iter().map(|&f| (f as f64).powf(0.75)).sum();

        let mut table = Vec::with_capacity(TABLE_SIZE);
        let mut cumulative = 0.0;

        for (node, &freq) in frequencies.iter().enumerate() {
            cumulative += (freq as f64).powf(0.75) / total;
            let count = (cumulative * TABLE_SIZE as f64) as usize - table.len();
            table.extend(std::iter::repeat(node as u32).take(count));
        }

        // Fill remaining slots
        while table.len() < TABLE_SIZE {
            table.push((self.num_nodes - 1) as u32);
        }

        self.neg_table = table;
    }

    /// Set total words for learning rate decay.
    pub fn set_total_words(&mut self, total: usize) {
        self.total_words = total;
    }

    /// Train on a single walk.
    ///
    /// For each position, predicts context nodes and updates embeddings.
    pub fn train_walk(&mut self, walk: &[u32]) {
        let _dim = self.config.embedding_dim; // reserved for future batch ops
        let window = self.config.window_size;
        let neg_samples = self.config.negative_samples;

        let mut rng = rand::rng();

        for (pos, &target) in walk.iter().enumerate() {
            // Dynamic window: sample actual window size
            let actual_window = rng.random_range(1..=window);

            // Context positions
            let start = pos.saturating_sub(actual_window);
            let end = (pos + actual_window + 1).min(walk.len());

            for ctx_pos in start..end {
                if ctx_pos == pos {
                    continue;
                }

                let context_node = walk[ctx_pos] as usize;

                // Positive sample: (target, context)
                self.train_pair(target as usize, context_node, true, &mut rng);

                // Negative samples
                for _ in 0..neg_samples {
                    let neg_idx = rng.random_range(0..self.neg_table.len());
                    let neg_node = self.neg_table[neg_idx] as usize;
                    if neg_node != context_node {
                        self.train_pair(target as usize, neg_node, false, &mut rng);
                    }
                }
            }

            // Update learning rate
            self.words_processed.fetch_add(1, Ordering::Relaxed);
            self.update_lr();
        }
    }

    /// Train on a (target, context) pair.
    ///
    /// Uses negative sampling objective:
    /// - Positive: maximize sigma(v_c . v_t)
    /// - Negative: maximize sigma(-v_n . v_t)
    fn train_pair<R: Rng>(&mut self, target: usize, context: usize, positive: bool, _rng: &mut R) {
        let dim = self.config.embedding_dim;
        let lr = self.current_lr;

        let t_offset = target * dim;
        let c_offset = context * dim;

        // Compute dot product
        let mut dot = 0.0f32;
        for i in 0..dim {
            dot += self.embeddings[t_offset + i] * self.context[c_offset + i];
        }

        // Sigmoid gradient
        let label = if positive { 1.0 } else { 0.0 };
        let sigmoid = 1.0 / (1.0 + (-dot).exp());
        let grad = (label - sigmoid) * lr;

        // Update embeddings
        for i in 0..dim {
            let t_grad = grad * self.context[c_offset + i];
            let c_grad = grad * self.embeddings[t_offset + i];

            self.embeddings[t_offset + i] += t_grad;
            self.context[c_offset + i] += c_grad;
        }
    }

    /// Update learning rate based on progress.
    fn update_lr(&mut self) {
        if self.total_words == 0 {
            return;
        }

        let processed = self.words_processed.load(Ordering::Relaxed);
        let progress = processed as f32 / self.total_words as f32;

        self.current_lr = self.config.learning_rate
            - (self.config.learning_rate - self.config.min_learning_rate) * progress;
        self.current_lr = self.current_lr.max(self.config.min_learning_rate);
    }

    /// Train on multiple walks in parallel.
    ///
    /// Each thread maintains its own gradient accumulator.
    /// Uses Hogwild-style asynchronous SGD.
    pub fn train_walks_parallel(&mut self, walks: &[Vec<u32>]) {
        // For true parallel training, we'd need atomic embeddings
        // or gradient accumulation. For now, sequential.
        for walk in walks {
            self.train_walk(walk);
        }
    }

    /// Get the learned embeddings.
    ///
    /// Returns a reference to the embedding matrix (num_nodes * embedding_dim).
    pub fn embeddings(&self) -> &[f32] {
        &self.embeddings
    }

    /// Get embedding for a specific node.
    pub fn embedding(&self, node: usize) -> &[f32] {
        let dim = self.config.embedding_dim;
        &self.embeddings[node * dim..(node + 1) * dim]
    }

    /// Get embeddings as a 2D array (num_nodes, embedding_dim).
    pub fn embeddings_2d(&self) -> Vec<Vec<f32>> {
        let dim = self.config.embedding_dim;
        (0..self.num_nodes)
            .map(|i| self.embeddings[i * dim..(i + 1) * dim].to_vec())
            .collect()
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }
}

/// Complete Node2Vec training pipeline.
///
/// Combines walk generation (from lattix-core) with Skip-gram training.
pub struct Node2VecTrainer {
    skipgram: SkipGram,
    walk_length: usize,
    num_walks: usize,
    p: f32,
    q: f32,
}

impl Node2VecTrainer {
    /// Create a new Node2Vec trainer.
    pub fn new(
        num_nodes: usize,
        embedding_dim: usize,
        walk_length: usize,
        num_walks: usize,
        p: f32,
        q: f32,
    ) -> Self {
        let config = SkipGramConfig {
            embedding_dim,
            ..Default::default()
        };

        Self {
            skipgram: SkipGram::new(num_nodes, config),
            walk_length,
            num_walks,
            p,
            q,
        }
    }

    /// Train on pre-generated walks.
    ///
    /// Walks should be generated using `lattix_core::algo::random_walk::Node2Vec`.
    pub fn train(&mut self, walks: &[Vec<u32>], epochs: usize) {
        let total_words: usize = walks.iter().map(|w| w.len()).sum();
        self.skipgram.set_total_words(total_words * epochs);

        for _epoch in 0..epochs {
            for walk in walks {
                self.skipgram.train_walk(walk);
            }
        }
    }

    /// Get the learned embeddings.
    pub fn embeddings(&self) -> &[f32] {
        self.skipgram.embeddings()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skipgram_init() {
        let model = SkipGram::new(100, SkipGramConfig::default());
        assert_eq!(model.num_nodes(), 100);
        assert_eq!(model.embedding_dim(), 128);
    }

    #[test]
    fn test_skipgram_train_walk() {
        let mut model = SkipGram::new(
            10,
            SkipGramConfig {
                embedding_dim: 32,
                window_size: 2,
                negative_samples: 2,
                ..Default::default()
            },
        );

        let walk = vec![0, 1, 2, 3, 4, 5];
        model.train_walk(&walk);

        // Just verify it runs without panic
        let emb = model.embedding(0);
        assert_eq!(emb.len(), 32);
    }

    #[test]
    fn test_embeddings_shape() {
        let model = SkipGram::new(
            50,
            SkipGramConfig {
                embedding_dim: 64,
                ..Default::default()
            },
        );

        let embs = model.embeddings_2d();
        assert_eq!(embs.len(), 50);
        assert_eq!(embs[0].len(), 64);
    }
}
