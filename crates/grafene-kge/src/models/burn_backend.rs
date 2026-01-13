//! Burn-based KGE models (GPU-accelerated).
//!
//! This module provides GPU-accelerated implementations of KGE models using
//! the [Burn](https://burn.dev) deep learning framework.
//!
//! # Backends
//!
//! Burn supports multiple backends:
//! - `NdArray`: Pure Rust, no dependencies (default)
//! - `Wgpu`: WebGPU, portable GPU acceleration
//! - `Tch`: PyTorch backend (requires libtorch)
//! - `Cuda`: Direct CUDA (via rust-gpu)
//!
//! # Example
//!
//! ```rust,ignore
//! use grafene_kge::models::TransEBurn;
//! use grafene_kge::{KGEModel, Fact, TrainingConfig};
//! use burn::backend::Wgpu;
//!
//! // GPU-accelerated TransE
//! let mut model: TransEBurn<Wgpu> = TransEBurn::new(128);
//! model.train(&triples, &config)?;
//! ```
//!
//! # When to Use
//!
//! Use Burn backends when:
//! - Training on >100K triples
//! - Need batch inference at scale
//! - Have GPU available
//!
//! For small datasets (<10K triples), ndarray backends are often faster
//! due to lower overhead.
//!
//! # Implementation Status
//!
//! This is a reference implementation showing how to structure Burn-based models.
//! For production use with GPU, add the burn dependency:
//!
//! ```toml
//! [dependencies.burn]
//! version = "0.16"
//! optional = true
//! features = ["wgpu", "autodiff"]
//! ```

#![cfg(feature = "burn")]

use crate::error::{Error, Result};
use crate::model::{EpochMetrics, Fact, GpuCapable, KGEModel, ProgressCallback};
use crate::training::TrainingConfig;
use std::collections::HashMap;
use std::marker::PhantomData;

// =============================================================================
// Backend-Agnostic TransE Implementation
// =============================================================================
//
// This demonstrates the pattern for Burn integration. When the actual `burn`
// dependency is added, replace the placeholder types with real Burn types.

/// TransE model using Burn backend for GPU acceleration.
///
/// Generic over Burn backend `B` (Wgpu, Tch, NdArray, etc.).
///
/// # Scoring Function
///
/// TransE embeds entities and relations as vectors and scores triples by:
///
/// ```text
/// score(h, r, t) = -||h + r - t||
/// ```
///
/// Higher scores indicate more plausible triples.
///
/// # Implementation Notes
///
/// When implementing with actual Burn tensors:
///
/// ```rust,ignore
/// use burn::prelude::*;
/// use burn::tensor::Tensor;
///
/// struct TransEBurn<B: Backend> {
///     entity_embeddings: Tensor<B, 2>,   // [num_entities, dim]
///     relation_embeddings: Tensor<B, 2>, // [num_relations, dim]
///     entity_to_idx: HashMap<String, usize>,
///     relation_to_idx: HashMap<String, usize>,
/// }
/// ```
#[derive(Debug)]
pub struct TransEBurn<B> {
    dim: usize,
    /// Entity embeddings as Vec<f32> (would be Tensor<B, 2> with real Burn)
    entity_embeddings: HashMap<String, Vec<f32>>,
    /// Relation embeddings
    relation_embeddings: HashMap<String, Vec<f32>>,
    /// Whether the model has been trained
    trained: bool,
    /// Phantom for backend type
    _backend: PhantomData<B>,
}

impl<B> TransEBurn<B> {
    /// Create a new GPU-accelerated TransE model.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            trained: false,
            _backend: PhantomData,
        }
    }

    /// Create from pre-trained embeddings.
    pub fn from_embeddings(
        entity_embeddings: HashMap<String, Vec<f32>>,
        relation_embeddings: HashMap<String, Vec<f32>>,
    ) -> Self {
        let dim = entity_embeddings
            .values()
            .next()
            .map(|e| e.len())
            .unwrap_or(128);

        Self {
            dim,
            entity_embeddings,
            relation_embeddings,
            trained: true,
            _backend: PhantomData,
        }
    }

    /// Compute TransE score: -||h + r - t||
    fn transe_score(&self, h: &[f32], r: &[f32], t: &[f32]) -> f32 {
        let mut sum_sq = 0.0f32;
        for i in 0..self.dim {
            let diff = h[i] + r[i] - t[i];
            sum_sq += diff * diff;
        }
        -sum_sq.sqrt()
    }
}

impl<B: Send + Sync + 'static> KGEModel for TransEBurn<B> {
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let h = self
            .entity_embeddings
            .get(head)
            .ok_or_else(|| Error::NotFound(format!("Entity not found: {}", head)))?;
        let r = self
            .relation_embeddings
            .get(relation)
            .ok_or_else(|| Error::NotFound(format!("Relation not found: {}", relation)))?;
        let t = self
            .entity_embeddings
            .get(tail)
            .ok_or_else(|| Error::NotFound(format!("Entity not found: {}", tail)))?;

        Ok(self.transe_score(h, r, t))
    }

    fn train(&mut self, triples: &[Fact<String>], config: &TrainingConfig) -> Result<f32> {
        self.train_with_callback(triples, config, Box::new(|_, _| {}))
    }

    fn train_with_callback(
        &mut self,
        triples: &[Fact<String>],
        config: &TrainingConfig,
        callback: ProgressCallback,
    ) -> Result<f32> {
        use std::collections::HashSet;

        if triples.is_empty() {
            return Err(Error::Validation("No training triples provided".into()));
        }

        // Extract vocabulary
        let mut entities: HashSet<String> = HashSet::new();
        let mut relations: HashSet<String> = HashSet::new();

        for t in triples {
            entities.insert(t.head.clone());
            entities.insert(t.tail.clone());
            relations.insert(t.relation.clone());
        }

        // Initialize embeddings uniformly in [-init_range, init_range]
        let init_range = 6.0 / (config.embedding_dim as f32).sqrt();
        let seed = config.seed;

        for (i, entity) in entities.iter().enumerate() {
            let embedding: Vec<f32> = (0..config.embedding_dim)
                .map(|j| {
                    let hash = ((seed as usize * 31 + i) * 17 + j) % 10000;
                    (hash as f32 / 10000.0 - 0.5) * 2.0 * init_range
                })
                .collect();
            self.entity_embeddings.insert(entity.clone(), embedding);
        }

        for (i, relation) in relations.iter().enumerate() {
            let embedding: Vec<f32> = (0..config.embedding_dim)
                .map(|j| {
                    let hash = ((seed as usize * 37 + i) * 13 + j) % 10000;
                    (hash as f32 / 10000.0 - 0.5) * 2.0 * init_range
                })
                .collect();
            self.relation_embeddings.insert(relation.clone(), embedding);
        }

        self.dim = config.embedding_dim;
        let entities_vec: Vec<&String> = entities.iter().collect();
        let mut final_loss = 0.0;

        // Training loop (simplified SGD without actual Burn tensors)
        // In real implementation, this would use Burn's autodiff
        for epoch in 0..config.epochs {
            let mut epoch_loss = 0.0;
            let mut num_updates = 0;

            for (batch_idx, batch) in triples.chunks(config.batch_size).enumerate() {
                for triple in batch {
                    let h = self.entity_embeddings.get(&triple.head).unwrap().clone();
                    let r = self.relation_embeddings.get(&triple.relation).unwrap().clone();
                    let t = self.entity_embeddings.get(&triple.tail).unwrap().clone();

                    let pos_score = self.transe_score(&h, &r, &t);

                    // Negative sampling
                    for ns in 0..config.negative_samples {
                        let neg_idx = (epoch * 1000 + batch_idx * 100 + ns) % entities_vec.len();
                        let neg_tail = entities_vec[neg_idx];

                        if neg_tail == &triple.tail {
                            continue;
                        }

                        let t_neg = self.entity_embeddings.get(neg_tail).unwrap();
                        let neg_score = self.transe_score(&h, &r, t_neg);

                        // Margin ranking loss
                        let loss = (config.margin + pos_score - neg_score).max(0.0);
                        epoch_loss += loss;

                        if loss > 0.0 {
                            let lr = config.learning_rate;

                            // Gradient update (simplified)
                            let h_mut = self.entity_embeddings.get_mut(&triple.head).unwrap();
                            let t_mut = self.entity_embeddings.get_mut(&triple.tail).unwrap();
                            let r_mut = self
                                .relation_embeddings
                                .get_mut(&triple.relation)
                                .unwrap();

                            for i in 0..self.dim {
                                let grad = (h[i] + r[i] - t[i])
                                    / (pos_score.abs() + 1e-8)
                                    * config.margin.signum();

                                h_mut[i] -= lr * grad;
                                r_mut[i] -= lr * grad;
                                t_mut[i] += lr * grad;
                            }

                            // Normalize embeddings
                            let h_norm: f32 = h_mut.iter().map(|x| x * x).sum::<f32>().sqrt();
                            let t_norm: f32 = t_mut.iter().map(|x| x * x).sum::<f32>().sqrt();

                            if h_norm > 1.0 {
                                h_mut.iter_mut().for_each(|x| *x /= h_norm);
                            }
                            if t_norm > 1.0 {
                                t_mut.iter_mut().for_each(|x| *x /= t_norm);
                            }

                            num_updates += 1;
                        }
                    }
                }
            }

            let avg_loss = if num_updates > 0 {
                epoch_loss / num_updates as f32
            } else {
                0.0
            };

            let metrics = EpochMetrics {
                loss: avg_loss,
                val_mrr: None,
                val_hits_at_10: None,
            };
            callback(epoch, &metrics);

            final_loss = avg_loss;
        }

        self.trained = true;
        Ok(final_loss)
    }

    fn entity_embedding(&self, entity: &str) -> Option<Vec<f32>> {
        self.entity_embeddings.get(entity).cloned()
    }

    fn relation_embedding(&self, relation: &str) -> Option<Vec<f32>> {
        self.relation_embeddings.get(relation).cloned()
    }

    fn entity_embeddings(&self) -> &HashMap<String, Vec<f32>> {
        &self.entity_embeddings
    }

    fn relation_embeddings(&self) -> &HashMap<String, Vec<f32>> {
        &self.relation_embeddings
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn num_entities(&self) -> usize {
        self.entity_embeddings.len()
    }

    fn num_relations(&self) -> usize {
        self.relation_embeddings.len()
    }

    fn name(&self) -> &'static str {
        "TransE-Burn"
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

impl<B: Send + Sync + 'static> GpuCapable for TransEBurn<B> {
    fn is_gpu_active(&self) -> bool {
        // In real implementation, would check backend type
        false
    }

    fn device(&self) -> &str {
        "cpu" // Would return actual device
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_triples() -> Vec<Fact<String>> {
        vec![
            Fact::from_strs("Einstein", "won", "NobelPrize"),
            Fact::from_strs("Einstein", "bornIn", "Germany"),
            Fact::from_strs("Curie", "won", "NobelPrize"),
            Fact::from_strs("Curie", "bornIn", "Poland"),
            Fact::from_strs("Paris", "capitalOf", "France"),
            Fact::from_strs("Berlin", "capitalOf", "Germany"),
        ]
    }

    // Use a concrete type for B since we can't actually use burn::backend::NdArray
    // In tests, we use () as placeholder backend
    type TestBackend = ();

    #[test]
    fn test_transebrun_creation() {
        let model: TransEBurn<TestBackend> = TransEBurn::new(64);
        assert_eq!(model.embedding_dim(), 64);
        assert!(!model.is_trained());
        assert_eq!(model.name(), "TransE-Burn");
    }

    #[test]
    fn test_transebrun_training() {
        let mut model: TransEBurn<TestBackend> = TransEBurn::new(32);
        let triples = sample_triples();

        let config = TrainingConfig::default()
            .with_embedding_dim(32)
            .with_epochs(10)
            .with_learning_rate(0.01);

        let loss = model.train(&triples, &config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.num_entities(), 8);
        assert_eq!(model.num_relations(), 3);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_transebrun_scoring() {
        let mut model: TransEBurn<TestBackend> = TransEBurn::new(32);
        let triples = sample_triples();

        let config = TrainingConfig::default()
            .with_embedding_dim(32)
            .with_epochs(20);

        model.train(&triples, &config).unwrap();

        let score = model.score("Einstein", "won", "NobelPrize").unwrap();
        assert!(score.is_finite());
        assert!(score <= 0.0); // TransE scores are negative distances
    }

    #[test]
    fn test_transebrun_from_embeddings() {
        let mut entities = HashMap::new();
        entities.insert("a".to_string(), vec![0.1, 0.2, 0.3]);
        entities.insert("b".to_string(), vec![0.4, 0.5, 0.6]);

        let mut relations = HashMap::new();
        relations.insert("r".to_string(), vec![0.1, 0.1, 0.1]);

        let model: TransEBurn<TestBackend> = TransEBurn::from_embeddings(entities, relations);

        assert!(model.is_trained());
        assert_eq!(model.num_entities(), 2);
        assert_eq!(model.num_relations(), 1);
        assert_eq!(model.embedding_dim(), 3);

        let score = model.score("a", "r", "b").unwrap();
        assert!(score.is_finite());
    }
}

// =============================================================================
// Implementation Notes for Full Burn Integration
// =============================================================================
//
// To fully implement with actual Burn tensors:
//
// 1. Add burn dependency to Cargo.toml:
//    ```toml
//    [dependencies.burn]
//    version = "0.16"
//    optional = true
//    features = ["wgpu", "autodiff", "ndarray"]
//    ```
//
// 2. Replace HashMap<String, Vec<f32>> with proper Tensor storage:
//    ```rust
//    use burn::prelude::*;
//    use burn::tensor::Tensor;
//
//    struct TransEBurn<B: Backend> {
//        entity_emb: Tensor<B, 2>,      // [num_entities, dim]
//        relation_emb: Tensor<B, 2>,    // [num_relations, dim]
//        entity_to_idx: HashMap<String, usize>,
//        relation_to_idx: HashMap<String, usize>,
//        idx_to_entity: Vec<String>,
//    }
//    ```
//
// 3. Implement batched scoring:
//    ```rust
//    fn score_batch(
//        &self,
//        h_idx: Tensor<B, 1, Int>,
//        r_idx: Tensor<B, 1, Int>,
//        t_idx: Tensor<B, 1, Int>,
//    ) -> Tensor<B, 1> {
//        let h = self.entity_emb.select(0, h_idx);
//        let r = self.relation_emb.select(0, r_idx);
//        let t = self.entity_emb.select(0, t_idx);
//        -(h + r - t).powf_scalar(2.0).sum_dim(1).sqrt()
//    }
//    ```
//
// 4. Training with autodiff:
//    ```rust
//    use burn::optim::{AdamConfig, GradientsParams, Optimizer};
//
//    fn train<B: AutodiffBackend>(
//        &mut self,
//        triples: &[Triple],
//        config: &Config,
//    ) {
//        let mut optimizer = AdamConfig::new().init();
//
//        for epoch in 0..config.epochs {
//            let loss = self.compute_loss(batch);
//            let grads = loss.backward();
//            let grads = GradientsParams::from_grads(grads, self);
//            *self = optimizer.step(config.lr, self.clone(), grads);
//        }
//    }
//    ```
//
// Key advantages when using actual Burn:
// - GPU acceleration via WGPU (works on macOS Metal, Windows DX12, Linux Vulkan)
// - Automatic differentiation for training
// - ~97% PyTorch performance (as of 2025)
// - Cross-platform without CUDA dependency
