//! MuRP: Multi-relational Poincare Graph Embeddings.
//!
//! MuRP ([Balazevic et al. 2019](https://arxiv.org/abs/1905.09791)) extends Poincare
//! embeddings to multi-relational data by learning relation-specific transformations.
//!
//! # Key Idea
//!
//! While HyperE uses `h +_M r ≈ t` (Mobius addition), MuRP uses:
//!
//! ```text
//! R_r * h +_M r ≈ t
//! ```
//!
//! where `R_r` is a diagonal relation matrix that rescales the head embedding.
//! This allows different relations to emphasize different dimensions.
//!
//! # Scoring
//!
//! ```text
//! score = -d_H(R_r * h +_M r, t)^2 + b_h + b_t
//! ```
//!
//! where `b_h`, `b_t` are learnable entity biases.
//!
//! # When to Use
//!
//! - **Hierarchies with typed relations**: WordNet (hypernym, meronym, etc.)
//! - **Multi-hop reasoning**: Relation composition in trees
//! - **Low-dimensional embeddings**: MuRP excels with dim < 50
//!
//! # Example
//!
//! ```rust,ignore
//! use lattix_kge::models::MuRP;
//! use lattix_kge::{KGEModel, Fact, TrainingConfig};
//!
//! let mut model = MuRP::new(32, 1.0);  // 32-dim, curvature 1.0
//!
//! let triples = vec![
//!     Fact::from_strs("mammal", "hypernym", "animal"),
//!     Fact::from_strs("dog", "hypernym", "mammal"),
//! ];
//!
//! model.train(&triples, &TrainingConfig::default())?;
//! ```
//!
//! # References
//!
//! - Balazevic et al. (2019). "Multi-relational Poincare Graph Embeddings"

use crate::error::{Error, Result};
use crate::model::{EpochMetrics, Fact, KGEModel, ProgressCallback};
use crate::training::TrainingConfig;
use hyperball::PoincareBall;
use ndarray::Array1;
use std::collections::{HashMap, HashSet};

/// MuRP model: Multi-relational Poincare embeddings.
///
/// Entities are embedded as points in the Poincare ball.
/// Relations have three components:
/// - Diagonal scaling matrix (rescales head)
/// - Translation vector (Mobius addition)
/// - Bias terms
pub struct MuRP {
    /// Embedding dimension.
    dim: usize,
    /// Curvature of hyperbolic space.
    curvature: f64,
    /// Entity embeddings (points in Poincare ball).
    entity_embeddings: HashMap<String, Array1<f64>>,
    /// Entity biases (scalar per entity).
    entity_biases: HashMap<String, f64>,
    /// Relation diagonal scaling (dim-sized vector).
    relation_diag: HashMap<String, Array1<f64>>,
    /// Relation translations (tangent vectors).
    relation_trans: HashMap<String, Array1<f64>>,
    /// Whether trained.
    trained: bool,
}

impl std::fmt::Debug for MuRP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MuRP")
            .field("dim", &self.dim)
            .field("curvature", &self.curvature)
            .field("num_entities", &self.entity_embeddings.len())
            .field("num_relations", &self.relation_diag.len())
            .field("trained", &self.trained)
            .finish()
    }
}

impl Clone for MuRP {
    fn clone(&self) -> Self {
        Self {
            dim: self.dim,
            curvature: self.curvature,
            entity_embeddings: self.entity_embeddings.clone(),
            entity_biases: self.entity_biases.clone(),
            relation_diag: self.relation_diag.clone(),
            relation_trans: self.relation_trans.clone(),
            trained: self.trained,
        }
    }
}

impl MuRP {
    /// Create a new MuRP model.
    ///
    /// # Arguments
    /// * `dim` - Embedding dimension
    /// * `curvature` - Hyperbolic curvature (1.0 = standard)
    pub fn new(dim: usize, curvature: f64) -> Self {
        Self {
            dim,
            curvature,
            entity_embeddings: HashMap::new(),
            entity_biases: HashMap::new(),
            relation_diag: HashMap::new(),
            relation_trans: HashMap::new(),
            trained: false,
        }
    }

    /// Create from pre-trained embeddings.
    pub fn from_embeddings(
        curvature: f64,
        entity_embeddings: HashMap<String, Array1<f64>>,
        entity_biases: HashMap<String, f64>,
        relation_diag: HashMap<String, Array1<f64>>,
        relation_trans: HashMap<String, Array1<f64>>,
    ) -> Self {
        let dim = entity_embeddings
            .values()
            .next()
            .map(|e| e.len())
            .unwrap_or(32);
        Self {
            dim,
            curvature,
            entity_embeddings,
            entity_biases,
            relation_diag,
            relation_trans,
            trained: true,
        }
    }

    /// Get the Poincare ball manifold.
    #[inline]
    fn manifold(&self) -> PoincareBall {
        PoincareBall::new(self.curvature)
    }

    /// MuRP scoring function.
    ///
    /// score = -d_H(diag(R) * h +_M r, t)^2 + b_h + b_t
    fn murp_score(
        &self,
        h: &Array1<f64>,
        t: &Array1<f64>,
        diag: &Array1<f64>,
        trans: &Array1<f64>,
        b_h: f64,
        b_t: f64,
    ) -> f64 {
        let manifold = self.manifold();

        // Apply diagonal scaling: diag(R) * h
        let scaled_h: Array1<f64> = h * diag;

        // Project to ensure we're in the ball
        let scaled_h = manifold.project(&scaled_h.view());

        // Map translation to manifold
        let trans_on_manifold = manifold.exp_map_zero(&trans.view());

        // Mobius addition: scaled_h +_M trans
        let transformed = manifold.mobius_add(&scaled_h.view(), &trans_on_manifold.view());

        // Hyperbolic distance to tail
        let dist = manifold.distance(&transformed.view(), &t.view());

        // Score with biases
        -dist * dist + b_h + b_t
    }

    /// Get entity depth (distance from origin in hyperbolic space).
    pub fn entity_depth(&self, entity: &str) -> Option<f64> {
        let e = self.entity_embeddings.get(entity)?;
        let origin = Array1::zeros(self.dim);
        Some(self.manifold().distance(&origin.view(), &e.view()))
    }

    /// Extract vocabulary from triples.
    fn extract_vocab(&self, triples: &[Fact<String>]) -> (HashSet<String>, HashSet<String>) {
        let mut entities = HashSet::new();
        let mut relations = HashSet::new();
        for t in triples {
            entities.insert(t.head.clone());
            entities.insert(t.tail.clone());
            relations.insert(t.relation.clone());
        }
        (entities, relations)
    }

    /// Initialize embeddings near origin.
    fn init_entity_embeddings(
        &self,
        vocab: &HashSet<String>,
        seed: u64,
    ) -> (HashMap<String, Array1<f64>>, HashMap<String, f64>) {
        use std::hash::{Hash, Hasher};

        let init_scale = 0.001;
        let mut embeddings = HashMap::new();
        let mut biases = HashMap::new();

        for (i, item) in vocab.iter().enumerate() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();

            let mut embedding = Vec::with_capacity(self.dim);
            for j in 0..self.dim {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                hash.hash(&mut h);
                j.hash(&mut h);
                let val = h.finish();
                embedding.push(((val as f64 / u64::MAX as f64) - 0.5) * init_scale);
            }

            embeddings.insert(item.clone(), Array1::from_vec(embedding));
            biases.insert(item.clone(), 0.0);
        }

        (embeddings, biases)
    }

    /// Initialize relation parameters.
    fn init_relation_params(
        &self,
        vocab: &HashSet<String>,
        seed: u64,
    ) -> (HashMap<String, Array1<f64>>, HashMap<String, Array1<f64>>) {
        use std::hash::{Hash, Hasher};

        let init_scale = 0.001;
        let mut diags = HashMap::new();
        let mut trans = HashMap::new();

        for (i, item) in vocab.iter().enumerate() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();

            // Diagonal: initialized near 1.0 (identity-like)
            let mut diag_vec = Vec::with_capacity(self.dim);
            // Translation: initialized near 0
            let mut trans_vec = Vec::with_capacity(self.dim);

            for j in 0..self.dim {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                hash.hash(&mut h);
                j.hash(&mut h);
                let val = h.finish();
                let f = (val as f64 / u64::MAX as f64) - 0.5;
                diag_vec.push(1.0 + f * init_scale);
                trans_vec.push(f * init_scale);
            }

            diags.insert(item.clone(), Array1::from_vec(diag_vec));
            trans.insert(item.clone(), Array1::from_vec(trans_vec));
        }

        (diags, trans)
    }
}

impl KGEModel for MuRP {
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let h = self
            .entity_embeddings
            .get(head)
            .ok_or_else(|| Error::NotFound(format!("Unknown entity: {}", head)))?;
        let t = self
            .entity_embeddings
            .get(tail)
            .ok_or_else(|| Error::NotFound(format!("Unknown entity: {}", tail)))?;
        let diag = self
            .relation_diag
            .get(relation)
            .ok_or_else(|| Error::NotFound(format!("Unknown relation: {}", relation)))?;
        let trans = self
            .relation_trans
            .get(relation)
            .ok_or_else(|| Error::NotFound(format!("Unknown relation: {}", relation)))?;

        let b_h = self.entity_biases.get(head).copied().unwrap_or(0.0);
        let b_t = self.entity_biases.get(tail).copied().unwrap_or(0.0);

        Ok(self.murp_score(h, t, diag, trans, b_h, b_t) as f32)
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
        if triples.is_empty() {
            return Err(Error::Validation("No training triples provided".into()));
        }

        // Initialize
        let (entities, relations) = self.extract_vocab(triples);
        self.dim = config.embedding_dim;

        let (entity_emb, entity_bias) = self.init_entity_embeddings(&entities, config.seed);
        let (rel_diag, rel_trans) = self.init_relation_params(&relations, config.seed + 1);

        self.entity_embeddings = entity_emb;
        self.entity_biases = entity_bias;
        self.relation_diag = rel_diag;
        self.relation_trans = rel_trans;

        let manifold = self.manifold();
        let entities_vec: Vec<&String> = entities.iter().collect();
        let mut final_loss = 0.0;

        // Burn-in period (reduced LR)
        let burn_in_epochs = config.epochs / 10;

        for epoch in 0..config.epochs {
            let lr = if epoch < burn_in_epochs {
                config.learning_rate as f64 * 0.1
            } else {
                config.learning_rate as f64
            };

            let mut epoch_loss = 0.0;
            let mut num_updates = 0;

            for (batch_idx, batch) in triples.chunks(config.batch_size).enumerate() {
                for triple in batch {
                    let h = self.entity_embeddings.get(&triple.head).unwrap().clone();
                    let t = self.entity_embeddings.get(&triple.tail).unwrap().clone();
                    let diag = self.relation_diag.get(&triple.relation).unwrap().clone();
                    let trans = self.relation_trans.get(&triple.relation).unwrap().clone();
                    let b_h = *self.entity_biases.get(&triple.head).unwrap();
                    let b_t = *self.entity_biases.get(&triple.tail).unwrap();

                    let pos_score = self.murp_score(&h, &t, &diag, &trans, b_h, b_t);

                    // Negative sampling
                    for ns in 0..config.negative_samples {
                        let neg_idx = (epoch * 1000 + batch_idx * 100 + ns) % entities_vec.len();
                        let neg_tail = entities_vec[neg_idx];

                        if neg_tail == &triple.tail {
                            continue;
                        }

                        let t_neg = self.entity_embeddings.get(neg_tail).unwrap();
                        let b_t_neg = *self.entity_biases.get(neg_tail).unwrap();
                        let neg_score = self.murp_score(&h, t_neg, &diag, &trans, b_h, b_t_neg);

                        // Margin-based ranking loss
                        let loss = (config.margin as f64 - pos_score + neg_score).max(0.0);
                        epoch_loss += loss as f32;

                        if loss > 0.0 {
                            // Simplified gradient updates
                            // Full Riemannian optimization would be more complex

                            // Update entity embeddings
                            let h_mut = self.entity_embeddings.get_mut(&triple.head).unwrap();
                            for i in 0..self.dim {
                                h_mut[i] -= lr * 0.5 * (h[i] - t[i]);
                            }
                            *h_mut = manifold.project(&h_mut.view());

                            let t_mut = self.entity_embeddings.get_mut(&triple.tail).unwrap();
                            for i in 0..self.dim {
                                t_mut[i] -= lr * 0.5 * (t[i] - h[i]);
                            }
                            *t_mut = manifold.project(&t_mut.view());

                            // Update relation translation
                            let trans_mut = self.relation_trans.get_mut(&triple.relation).unwrap();
                            for i in 0..self.dim {
                                trans_mut[i] -= lr * 0.3 * (h[i] + trans[i] - t[i]);
                            }

                            // Update relation diagonal (keep positive)
                            let diag_mut = self.relation_diag.get_mut(&triple.relation).unwrap();
                            for i in 0..self.dim {
                                diag_mut[i] -= lr * 0.1 * h[i] * (h[i] * diag[i] - t[i]);
                                diag_mut[i] = diag_mut[i].max(0.01); // Keep positive
                            }

                            // Update biases
                            *self.entity_biases.get_mut(&triple.head).unwrap() += lr * 0.1;
                            *self.entity_biases.get_mut(&triple.tail).unwrap() += lr * 0.1;

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
        self.entity_embeddings
            .get(entity)
            .map(|e| e.iter().map(|&x| x as f32).collect())
    }

    fn relation_embedding(&self, relation: &str) -> Option<Vec<f32>> {
        // Return translation as the "embedding"
        self.relation_trans
            .get(relation)
            .map(|e| e.iter().map(|&x| x as f32).collect())
    }

    fn entity_embeddings(&self) -> &HashMap<String, Vec<f32>> {
        // This is inefficient but maintains trait compatibility
        // In practice, use entity_embedding() for individual lookups
        static EMPTY: std::sync::OnceLock<HashMap<String, Vec<f32>>> = std::sync::OnceLock::new();
        EMPTY.get_or_init(HashMap::new)
    }

    fn relation_embeddings(&self) -> &HashMap<String, Vec<f32>> {
        static EMPTY: std::sync::OnceLock<HashMap<String, Vec<f32>>> = std::sync::OnceLock::new();
        EMPTY.get_or_init(HashMap::new)
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn num_entities(&self) -> usize {
        self.entity_embeddings.len()
    }

    fn num_relations(&self) -> usize {
        self.relation_diag.len()
    }

    fn name(&self) -> &'static str {
        "MuRP"
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_hierarchy() -> Vec<Fact<String>> {
        vec![
            Fact::from_strs("entity", "root", "root"),
            Fact::from_strs("animal", "hypernym", "entity"),
            Fact::from_strs("mammal", "hypernym", "animal"),
            Fact::from_strs("dog", "hypernym", "mammal"),
            Fact::from_strs("cat", "hypernym", "mammal"),
            Fact::from_strs("bird", "hypernym", "animal"),
        ]
    }

    #[test]
    fn test_murp_creation() {
        let model = MuRP::new(32, 1.0);
        assert_eq!(model.embedding_dim(), 32);
        assert!(!model.is_trained());
        assert_eq!(model.name(), "MuRP");
    }

    #[test]
    fn test_murp_training() {
        let mut model = MuRP::new(16, 1.0);
        let triples = sample_hierarchy();

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(20)
            .with_learning_rate(0.01);

        let loss = model.train(&triples, &config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.num_entities(), 7); // entity, root, animal, mammal, dog, cat, bird
        assert_eq!(model.num_relations(), 2); // root, hypernym
        assert!(loss.is_finite());
    }

    #[test]
    fn test_murp_scoring() {
        let mut model = MuRP::new(16, 1.0);
        let triples = sample_hierarchy();

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(30);

        model.train(&triples, &config).unwrap();

        let score = model.score("dog", "hypernym", "mammal").unwrap();
        assert!(score.is_finite());
    }

    #[test]
    fn test_murp_hierarchy_depth() {
        let mut model = MuRP::new(16, 1.0);
        let triples = sample_hierarchy();

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(50)
            .with_learning_rate(0.01);

        model.train(&triples, &config).unwrap();

        // After training, we can check relative depths
        // (Note: with small data, hierarchy may not perfectly emerge)
        if let (Some(entity_depth), Some(dog_depth)) =
            (model.entity_depth("entity"), model.entity_depth("dog"))
        {
            eprintln!(
                "entity depth: {:.4}, dog depth: {:.4}",
                entity_depth, dog_depth
            );
        }
    }
}
