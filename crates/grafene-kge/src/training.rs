//! Training loop for Knowledge Graph Embeddings.
//!
//! Provides a unified training interface for different KGE models:
//! - TransE, DistMult, RotatE, ComplEx: Point embedding models
//! - BoxE: Box embedding model (via `boxe` feature)
//!
//! # BoxE Training
//!
//! BoxE ([Abboud et al. 2020](https://arxiv.org/abs/2007.06267)) represents
//! entities as points and relations as boxes. Training follows the standard
//! KGE paradigm: maximize positive triple scores, minimize negative triple scores.
//!
//! The training loop:
//! 1. For each positive triple (h, r, t):
//! 2.   Generate negative samples by corrupting h or t
//! 3.   Compute scores for positive and negative triples
//! 4.   Update embeddings to minimize margin-based ranking loss
//!
//! # Example
//!
//! ```rust,ignore
//! use grafene_kge::training::{KGETrainer, TrainingConfig};
//!
//! // Load dataset (triples as (head, relation, tail))
//! let triples = vec![
//!     ("Einstein", "won", "NobelPrize"),
//!     ("Paris", "capitalOf", "France"),
//! ];
//!
//! // Configure training
//! let config = TrainingConfig::default()
//!     .with_embedding_dim(128)
//!     .with_learning_rate(0.001)
//!     .with_epochs(100);
//!
//! // Train TransE
//! let trainer = KGETrainer::new(config);
//! let embeddings = trainer.train_transe(&triples)?;
//! ```

use crate::error::{Error, Result};
use std::collections::{HashMap, HashSet};

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Embedding dimension (default: 128).
    pub embedding_dim: usize,
    /// Learning rate (default: 0.001).
    pub learning_rate: f32,
    /// Number of training epochs (default: 100).
    pub epochs: usize,
    /// Batch size (default: 512).
    pub batch_size: usize,
    /// Negative samples per positive (default: 5).
    pub negative_samples: usize,
    /// Margin for ranking loss (default: 1.0).
    pub margin: f32,
    /// Random seed (default: 42).
    pub seed: u64,
    /// Regularization weight (default: 0.01).
    pub regularization: f32,
    /// Early stopping patience (None = no early stopping).
    pub early_stopping: Option<usize>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 512,
            negative_samples: 5,
            margin: 1.0,
            seed: 42,
            regularization: 0.01,
            early_stopping: Some(10),
        }
    }
}

impl TrainingConfig {
    pub fn with_embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_negative_samples(mut self, n: usize) -> Self {
        self.negative_samples = n;
        self
    }

    pub fn with_margin(mut self, margin: f32) -> Self {
        self.margin = margin;
        self
    }
}

/// A triple in a knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub head: String,
    pub relation: String,
    pub tail: String,
}

impl Triple {
    pub fn new(head: impl Into<String>, relation: impl Into<String>, tail: impl Into<String>) -> Self {
        Self {
            head: head.into(),
            relation: relation.into(),
            tail: tail.into(),
        }
    }
}

/// Training results.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Entity embeddings (entity_id -> embedding).
    pub entity_embeddings: HashMap<String, Vec<f32>>,
    /// Relation embeddings (relation_id -> embedding).
    pub relation_embeddings: HashMap<String, Vec<f32>>,
    /// Training loss per epoch.
    pub loss_history: Vec<f32>,
    /// Validation MRR per epoch (if validation set provided).
    pub validation_mrr: Vec<f32>,
    /// Best epoch based on validation.
    pub best_epoch: usize,
}

/// Knowledge Graph Embedding trainer.
pub struct KGETrainer {
    config: TrainingConfig,
}

impl KGETrainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Extract entities and relations from triples.
    fn extract_vocab(&self, triples: &[Triple]) -> (HashSet<String>, HashSet<String>) {
        let mut entities = HashSet::new();
        let mut relations = HashSet::new();

        for t in triples {
            entities.insert(t.head.clone());
            entities.insert(t.tail.clone());
            relations.insert(t.relation.clone());
        }

        (entities, relations)
    }

    /// Initialize random embeddings.
    fn init_embeddings(
        &self,
        vocab: &HashSet<String>,
        dim: usize,
        seed: u64,
    ) -> HashMap<String, Vec<f32>> {
        use std::hash::{Hash, Hasher};

        let mut embeddings = HashMap::new();

        for (i, item) in vocab.iter().enumerate() {
            // Simple deterministic initialization based on seed + index
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();

            let mut embedding = Vec::with_capacity(dim);
            for j in 0..dim {
                // Use hash to generate pseudo-random values
                let mut h = std::collections::hash_map::DefaultHasher::new();
                hash.hash(&mut h);
                j.hash(&mut h);
                let val = h.finish();
                // Map to [-0.5, 0.5]
                let f = (val as f64 / u64::MAX as f64) - 0.5;
                embedding.push(f as f32);
            }

            // Normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in &mut embedding {
                    *x /= norm;
                }
            }

            embeddings.insert(item.clone(), embedding);
        }

        embeddings
    }

    /// Train TransE model.
    ///
    /// TransE interprets relations as translations: h + r ≈ t.
    /// Loss = max(0, margin + ||h + r - t|| - ||h' + r - t'||)
    pub fn train_transe(&self, triples: &[Triple]) -> Result<TrainingResult> {
        let (entities, relations) = self.extract_vocab(triples);

        let mut entity_emb = self.init_embeddings(&entities, self.config.embedding_dim, self.config.seed);
        let mut relation_emb = self.init_embeddings(&relations, self.config.embedding_dim, self.config.seed + 1);

        let entities_vec: Vec<&String> = entities.iter().collect();
        let mut loss_history = Vec::with_capacity(self.config.epochs);

        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Process in batches
            for batch in triples.chunks(self.config.batch_size) {
                let mut batch_loss = 0.0;

                for triple in batch {
                    // Clone embeddings to avoid borrow conflicts
                    let h = entity_emb.get(&triple.head).unwrap().clone();
                    let r = relation_emb.get(&triple.relation).unwrap().clone();
                    let t = entity_emb.get(&triple.tail).unwrap().clone();

                    // Positive score (distance)
                    let pos_dist = transe_distance(&h, &r, &t);

                    // Generate negative samples
                    for _ in 0..self.config.negative_samples {
                        // Corrupt tail (most common strategy)
                        let neg_idx = (epoch * 1000 + num_batches) % entities_vec.len();
                        let neg_tail = entities_vec[neg_idx];

                        if neg_tail == &triple.tail {
                            continue;
                        }

                        let t_neg = entity_emb.get(neg_tail).unwrap();
                        let neg_dist = transe_distance(&h, &r, t_neg);

                        // Margin-based loss
                        let loss = (self.config.margin + pos_dist - neg_dist).max(0.0);
                        batch_loss += loss;

                        if loss > 0.0 {
                            // Gradient descent update (simplified)
                            // In a full implementation, this would use proper optimizers
                            let lr = self.config.learning_rate;

                            // Update embeddings in the direction that reduces loss
                            let h_mut = entity_emb.get_mut(&triple.head).unwrap();
                            for i in 0..self.config.embedding_dim {
                                let grad = 2.0 * (h[i] + r[i] - t[i]);
                                h_mut[i] -= lr * grad;
                            }

                            let t_mut = entity_emb.get_mut(&triple.tail).unwrap();
                            for i in 0..self.config.embedding_dim {
                                let grad = -2.0 * (h[i] + r[i] - t[i]);
                                t_mut[i] -= lr * grad;
                            }

                            let r_mut = relation_emb.get_mut(&triple.relation).unwrap();
                            for i in 0..self.config.embedding_dim {
                                let grad = 2.0 * (h[i] + r[i] - t[i]);
                                r_mut[i] -= lr * grad;
                            }
                        }
                    }
                }

                epoch_loss += batch_loss;
                num_batches += 1;
            }

            let avg_loss = if num_batches > 0 {
                epoch_loss / (num_batches * self.config.batch_size) as f32
            } else {
                0.0
            };

            loss_history.push(avg_loss);

            if epoch % 10 == 0 {
                eprintln!("Epoch {}: loss = {:.4}", epoch, avg_loss);
            }

            // Early stopping check
            if let Some(patience) = self.config.early_stopping {
                if loss_history.len() > patience {
                    let recent = &loss_history[loss_history.len() - patience..];
                    let first = recent[0];
                    let last = recent[recent.len() - 1];
                    if last >= first {
                        eprintln!("Early stopping at epoch {}", epoch);
                        break;
                    }
                }
            }
        }

        Ok(TrainingResult {
            entity_embeddings: entity_emb,
            relation_embeddings: relation_emb,
            loss_history,
            validation_mrr: Vec::new(),
            best_epoch: self.config.epochs.saturating_sub(1),
        })
    }
}

/// TransE distance: ||h + r - t||_2
fn transe_distance(h: &[f32], r: &[f32], t: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..h.len() {
        let diff = h[i] + r[i] - t[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

// ============================================================================
// BoxE Training (with `boxe` feature)
// ============================================================================

#[cfg(feature = "boxe")]
pub mod boxe {
    //! BoxE training for knowledge graph completion.
    //!
    //! BoxE represents entities as points and relations as boxes with
    //! translational bumps. A triple (h, r, t) is plausible if the
    //! translated head falls within the relation's box near the tail.
    //!
    //! # Key Concepts
    //!
    //! - **Entity embedding**: A point in R^d
    //! - **Relation box**: Center + offset defining a box
    //! - **Bump**: Translation applied to head before containment check
    //!
    //! The scoring function computes how well the translated head
    //! aligns with the tail within the relation's box constraints.

    use super::*;
    use subsume_core::trainer::{TrainingConfig as SubsumeConfig, NegativeSamplingStrategy};
    use subsume_core::dataset::Triple as SubsumeTriple;

    /// Convert grafene triples to subsume format.
    pub fn to_subsume_triples(triples: &[Triple]) -> Vec<SubsumeTriple> {
        triples
            .iter()
            .map(|t| SubsumeTriple {
                head: t.head.clone(),
                relation: t.relation.clone(),
                tail: t.tail.clone(),
            })
            .collect()
    }

    /// BoxE training configuration.
    #[derive(Debug, Clone)]
    pub struct BoxEConfig {
        /// Base embedding dimension.
        pub embedding_dim: usize,
        /// Learning rate.
        pub learning_rate: f32,
        /// Training epochs.
        pub epochs: usize,
        /// Batch size.
        pub batch_size: usize,
        /// Negative samples per positive.
        pub negative_samples: usize,
        /// Margin for ranking loss.
        pub margin: f32,
        /// Temperature for softbox (default: 1.0).
        pub temperature: f32,
        /// Box regularization weight.
        pub box_regularization: f32,
    }

    impl Default for BoxEConfig {
        fn default() -> Self {
            Self {
                embedding_dim: 128,
                learning_rate: 0.001,
                epochs: 100,
                batch_size: 512,
                negative_samples: 5,
                margin: 1.0,
                temperature: 1.0,
                box_regularization: 0.01,
            }
        }
    }

    impl From<BoxEConfig> for SubsumeConfig {
        fn from(c: BoxEConfig) -> Self {
            SubsumeConfig {
                learning_rate: c.learning_rate,
                epochs: c.epochs,
                batch_size: c.batch_size,
                negative_samples: c.negative_samples,
                negative_strategy: NegativeSamplingStrategy::CorruptTail,
                regularization_weight: c.box_regularization,
                temperature: c.temperature,
                weight_decay: 1e-5,
                margin: c.margin,
                early_stopping_patience: Some(10),
            }
        }
    }

    /// BoxE training result.
    #[derive(Debug, Clone)]
    pub struct BoxEResult {
        /// Entity embeddings (entity -> point coordinates).
        pub entity_points: HashMap<String, Vec<f32>>,
        /// Relation boxes (relation -> (center, offset)).
        pub relation_boxes: HashMap<String, (Vec<f32>, Vec<f32>)>,
        /// Relation bumps (relation -> translation vector).
        pub relation_bumps: HashMap<String, Vec<f32>>,
        /// Training loss history.
        pub loss_history: Vec<f32>,
    }

    /// Train BoxE model on knowledge graph triples.
    ///
    /// # Arguments
    ///
    /// * `triples` - Training triples (head, relation, tail)
    /// * `config` - BoxE training configuration
    ///
    /// # Returns
    ///
    /// BoxE result with entity points and relation boxes.
    pub fn train_boxe(triples: &[Triple], config: BoxEConfig) -> Result<BoxEResult> {
        // Extract vocabulary
        let mut entities = HashSet::new();
        let mut relations = HashSet::new();

        for t in triples {
            entities.insert(t.head.clone());
            entities.insert(t.tail.clone());
            relations.insert(t.relation.clone());
        }

        let dim = config.embedding_dim;
        let _subsume_config: SubsumeConfig = config.clone().into();

        // Initialize entity points (random in [-0.5, 0.5])
        let mut entity_points = HashMap::new();
        for (i, entity) in entities.iter().enumerate() {
            let point: Vec<f32> = (0..dim)
                .map(|j| {
                    let seed = (i * dim + j) as f64;
                    ((seed * 0.618033988749895) % 1.0 - 0.5) as f32
                })
                .collect();
            entity_points.insert(entity.clone(), point);
        }

        // Initialize relation boxes (center + offset)
        let mut relation_boxes = HashMap::new();
        let mut relation_bumps = HashMap::new();

        for (i, relation) in relations.iter().enumerate() {
            // Center at origin, offset = 0.5 (unit box)
            let center: Vec<f32> = (0..dim)
                .map(|j| {
                    let seed = ((i + 1000) * dim + j) as f64;
                    ((seed * 0.618033988749895) % 1.0 - 0.5) as f32 * 0.1
                })
                .collect();
            let offset: Vec<f32> = vec![0.5; dim];

            // Small random bump
            let bump: Vec<f32> = (0..dim)
                .map(|j| {
                    let seed = ((i + 2000) * dim + j) as f64;
                    ((seed * 0.618033988749895) % 1.0 - 0.5) as f32 * 0.1
                })
                .collect();

            relation_boxes.insert(relation.clone(), (center, offset));
            relation_bumps.insert(relation.clone(), bump);
        }

        // Training loop
        let lr = config.learning_rate;
        let margin = config.margin;
        let mut loss_history = Vec::new();
        let entities_vec: Vec<String> = entities.into_iter().collect();

        for epoch in 0..config.epochs {
            let mut epoch_loss = 0.0;

            for (batch_idx, batch) in triples.chunks(config.batch_size).enumerate() {
                for triple in batch {
                    // Clone to avoid borrow conflicts
                    let h = entity_points.get(&triple.head).unwrap().clone();
                    let t = entity_points.get(&triple.tail).unwrap().clone();
                    let (center, offset) = relation_boxes.get(&triple.relation).unwrap().clone();
                    let bump = relation_bumps.get(&triple.relation).unwrap().clone();

                    // Positive score: distance from (h + bump) to t, penalized by box violation
                    let pos_score = boxe_score_simple(&h, &t, &center, &offset, &bump);

                    // Negative sample
                    let neg_idx = (epoch * 1000 + batch_idx) % entities_vec.len();
                    let neg_tail = &entities_vec[neg_idx];
                    if neg_tail == &triple.tail {
                        continue;
                    }
                    let t_neg = entity_points.get(neg_tail).unwrap();
                    let neg_score = boxe_score_simple(&h, t_neg, &center, &offset, &bump);

                    // Margin-based loss: want pos_score > neg_score + margin
                    // But BoxE uses distance-like scores (lower = better)
                    // So loss = max(0, margin + pos_score - neg_score)
                    let loss = (margin + pos_score - neg_score).max(0.0);
                    epoch_loss += loss;

                    if loss > 0.0 {
                        // Simplified gradient updates
                        // In production, use proper optimizers (Adam, AdamW)

                        // Update entity points
                        let h_mut = entity_points.get_mut(&triple.head).unwrap();
                        for i in 0..dim {
                            let grad = 2.0 * ((h[i] + bump[i]) - t[i]);
                            h_mut[i] -= lr * grad * 0.1;
                        }

                        let t_mut = entity_points.get_mut(&triple.tail).unwrap();
                        for i in 0..dim {
                            let grad = -2.0 * ((h[i] + bump[i]) - t[i]);
                            t_mut[i] -= lr * grad * 0.1;
                        }

                        // Update relation bump
                        let bump_mut = relation_bumps.get_mut(&triple.relation).unwrap();
                        for i in 0..dim {
                            let grad = 2.0 * ((h[i] + bump[i]) - t[i]);
                            bump_mut[i] -= lr * grad * 0.1;
                        }
                    }
                }
            }

            let avg_loss = epoch_loss / triples.len() as f32;
            loss_history.push(avg_loss);

            if epoch % 20 == 0 {
                eprintln!("BoxE Epoch {}: loss = {:.4}", epoch, avg_loss);
            }
        }

        Ok(BoxEResult {
            entity_points,
            relation_boxes,
            relation_bumps,
            loss_history,
        })
    }

    /// Simplified BoxE score (distance-based).
    ///
    /// Computes ||h + bump - t||^2 plus box violation penalty.
    fn boxe_score_simple(
        head: &[f32],
        tail: &[f32],
        center: &[f32],
        offset: &[f32],
        bump: &[f32],
    ) -> f32 {
        let dim = head.len();
        let mut dist_sq = 0.0;
        let mut violation = 0.0;

        for i in 0..dim {
            // Translated head
            let h_bumped = head[i] + bump[i];

            // Distance to tail
            let diff = h_bumped - tail[i];
            dist_sq += diff * diff;

            // Box violation: how far is h_bumped from the relation box?
            let box_min = center[i] - offset[i];
            let box_max = center[i] + offset[i];

            if h_bumped < box_min {
                violation += (box_min - h_bumped).powi(2);
            } else if h_bumped > box_max {
                violation += (h_bumped - box_max).powi(2);
            }
        }

        dist_sq.sqrt() + violation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_triples() -> Vec<Triple> {
        vec![
            Triple::new("Einstein", "won", "NobelPrize"),
            Triple::new("Curie", "won", "NobelPrize"),
            Triple::new("Paris", "capitalOf", "France"),
            Triple::new("Berlin", "capitalOf", "Germany"),
            Triple::new("France", "locatedIn", "Europe"),
            Triple::new("Germany", "locatedIn", "Europe"),
        ]
    }

    #[test]
    fn test_training_config_builder() {
        let config = TrainingConfig::default()
            .with_embedding_dim(64)
            .with_learning_rate(0.01)
            .with_epochs(50);

        assert_eq!(config.embedding_dim, 64);
        assert!((config.learning_rate - 0.01).abs() < 1e-6);
        assert_eq!(config.epochs, 50);
    }

    #[test]
    fn test_transe_training() {
        let triples = sample_triples();
        let config = TrainingConfig::default()
            .with_embedding_dim(32)
            .with_epochs(10)
            .with_batch_size(4);

        let trainer = KGETrainer::new(config);
        let result = trainer.train_transe(&triples).unwrap();

        // Check all entities have embeddings
        assert!(result.entity_embeddings.contains_key("Einstein"));
        assert!(result.entity_embeddings.contains_key("NobelPrize"));
        assert!(result.entity_embeddings.contains_key("Paris"));
        assert!(result.entity_embeddings.contains_key("France"));

        // Check all relations have embeddings
        assert!(result.relation_embeddings.contains_key("won"));
        assert!(result.relation_embeddings.contains_key("capitalOf"));
        assert!(result.relation_embeddings.contains_key("locatedIn"));

        // Check embedding dimensions
        assert_eq!(result.entity_embeddings["Einstein"].len(), 32);
        assert_eq!(result.relation_embeddings["won"].len(), 32);

        // Check loss history exists
        assert!(!result.loss_history.is_empty());
    }

    #[test]
    fn test_transe_distance() {
        let h = vec![1.0, 0.0, 0.0];
        let r = vec![0.0, 1.0, 0.0];
        let t = vec![1.0, 1.0, 0.0];

        let dist = transe_distance(&h, &r, &t);
        assert!(dist.abs() < 1e-6); // h + r = t exactly
    }

    #[cfg(feature = "boxe")]
    mod boxe_tests {
        use super::super::boxe::*;
        use super::*;

        #[test]
        fn test_boxe_config() {
            let config = BoxEConfig::default();
            assert_eq!(config.embedding_dim, 128);
            assert!((config.temperature - 1.0).abs() < 1e-6);
        }

        #[test]
        fn test_boxe_training() {
            let triples = sample_triples();
            let config = BoxEConfig {
                embedding_dim: 16,
                epochs: 5,
                batch_size: 4,
                ..Default::default()
            };

            let result = train_boxe(&triples, config).unwrap();

            // Check entity points exist
            assert!(result.entity_points.contains_key("Einstein"));
            assert!(result.entity_points.contains_key("France"));

            // Check relation boxes exist
            assert!(result.relation_boxes.contains_key("won"));
            assert!(result.relation_boxes.contains_key("capitalOf"));

            // Check relation bumps exist
            assert!(result.relation_bumps.contains_key("won"));

            // Check dimensions
            assert_eq!(result.entity_points["Einstein"].len(), 16);

            let (center, offset) = &result.relation_boxes["won"];
            assert_eq!(center.len(), 16);
            assert_eq!(offset.len(), 16);
        }
    }
}
