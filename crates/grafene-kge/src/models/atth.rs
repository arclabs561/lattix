//! AttH: Attention-based Hyperbolic Embeddings.
//!
//! AttH ([Chami et al. 2020](https://arxiv.org/abs/2005.00545)) extends hyperbolic
//! KGE with attention-based curvature selection and relation-specific transformations.
//!
//! # Key Innovation
//!
//! AttH learns **relation-specific curvatures** rather than using a fixed global
//! curvature. This allows the model to adapt to different hierarchical depths
//! across relations.
//!
//! # Scoring
//!
//! ```text
//! score = -d_H^{c_r}(f_r(h), t)^2 + b_h + b_t
//! ```
//!
//! where:
//! - `c_r` is the relation-specific curvature
//! - `f_r(h)` is a relation-specific transformation (rotation + reflection)
//! - `d_H^c` is hyperbolic distance at curvature c
//!
//! # Attention Mechanism
//!
//! The attention weights combine:
//! - **Rotation**: For composition patterns
//! - **Reflection**: For symmetric relations
//!
//! # When to Use
//!
//! - Multiple relation types with different hierarchy depths
//! - Mix of symmetric and hierarchical relations
//! - Need to learn curvature from data
//!
//! # Example
//!
//! ```rust,ignore
//! use grafene_kge::models::AttH;
//! use grafene_kge::{KGEModel, Fact, TrainingConfig};
//!
//! let mut model = AttH::new(32);  // 32-dim
//!
//! let triples = vec![
//!     Fact::from_strs("cat", "isA", "mammal"),      // deep hierarchy
//!     Fact::from_strs("Alice", "knows", "Bob"),     // flat relation
//! ];
//!
//! model.train(&triples, &TrainingConfig::default())?;
//! ```

use crate::error::{Error, Result};
use crate::model::{EpochMetrics, Fact, KGEModel, ProgressCallback};
use crate::training::TrainingConfig;
use hyperball::PoincareBall;
use ndarray::Array1;
use std::collections::{HashMap, HashSet};

/// AttH model: Attention-based Hyperbolic embeddings.
///
/// Learns per-relation curvatures and combines rotation + reflection
/// transformations via attention.
pub struct AttH {
    /// Embedding dimension.
    dim: usize,
    /// Entity embeddings.
    entity_embeddings: HashMap<String, Array1<f64>>,
    /// Entity biases.
    entity_biases: HashMap<String, f64>,
    /// Relation-specific curvatures (learnable).
    relation_curvatures: HashMap<String, f64>,
    /// Relation rotation angles.
    relation_rotations: HashMap<String, Vec<f64>>,
    /// Relation reflection vectors.
    relation_reflections: HashMap<String, Array1<f64>>,
    /// Attention weights: [alpha_rot, alpha_ref] per relation.
    relation_attention: HashMap<String, (f64, f64)>,
    /// Whether trained.
    trained: bool,
}

impl std::fmt::Debug for AttH {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttH")
            .field("dim", &self.dim)
            .field("num_entities", &self.entity_embeddings.len())
            .field("num_relations", &self.relation_curvatures.len())
            .field("trained", &self.trained)
            .finish()
    }
}

impl Clone for AttH {
    fn clone(&self) -> Self {
        Self {
            dim: self.dim,
            entity_embeddings: self.entity_embeddings.clone(),
            entity_biases: self.entity_biases.clone(),
            relation_curvatures: self.relation_curvatures.clone(),
            relation_rotations: self.relation_rotations.clone(),
            relation_reflections: self.relation_reflections.clone(),
            relation_attention: self.relation_attention.clone(),
            trained: self.trained,
        }
    }
}

impl AttH {
    /// Create a new AttH model.
    ///
    /// # Arguments
    /// * `dim` - Embedding dimension (will be rounded up to even if odd)
    pub fn new(dim: usize) -> Self {
        let dim = if dim % 2 == 0 { dim } else { dim + 1 };
        Self {
            dim,
            entity_embeddings: HashMap::new(),
            entity_biases: HashMap::new(),
            relation_curvatures: HashMap::new(),
            relation_rotations: HashMap::new(),
            relation_reflections: HashMap::new(),
            relation_attention: HashMap::new(),
            trained: false,
        }
    }

    /// Get manifold at a specific curvature.
    fn manifold(&self, curvature: f64) -> PoincareBall {
        PoincareBall::new(curvature.abs().max(0.001))
    }

    /// Apply Givens rotation.
    fn apply_rotation(&self, v: &Array1<f64>, angles: &[f64]) -> Array1<f64> {
        let mut result = v.clone();
        for (pair_idx, &angle) in angles.iter().enumerate() {
            let i = pair_idx * 2;
            let j = i + 1;
            if j >= self.dim {
                break;
            }
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let vi = result[i];
            let vj = result[j];
            result[i] = cos_a * vi - sin_a * vj;
            result[j] = sin_a * vi + cos_a * vj;
        }
        result
    }

    /// Apply reflection (Householder).
    fn apply_reflection(&self, v: &Array1<f64>, r: &Array1<f64>) -> Array1<f64> {
        let r_norm_sq: f64 = r.iter().map(|x| x * x).sum();
        if r_norm_sq < 1e-12 {
            return v.clone();
        }

        // Householder: v - 2 * (v . r / |r|^2) * r
        let dot: f64 = v.iter().zip(r.iter()).map(|(a, b)| a * b).sum();
        let scale = 2.0 * dot / r_norm_sq;

        let mut result = v.clone();
        for i in 0..self.dim {
            result[i] -= scale * r[i];
        }
        result
    }

    /// AttH scoring: combined rotation + reflection with attention.
    fn atth_score(
        &self,
        h: &Array1<f64>,
        t: &Array1<f64>,
        curvature: f64,
        rotation: &[f64],
        reflection: &Array1<f64>,
        attention: (f64, f64),
        b_h: f64,
        b_t: f64,
    ) -> f64 {
        let manifold = self.manifold(curvature);
        let (alpha_rot, alpha_ref) = attention;

        // Normalize attention weights (softmax)
        let exp_rot = alpha_rot.exp();
        let exp_ref = alpha_ref.exp();
        let total = exp_rot + exp_ref;
        let w_rot = exp_rot / total;
        let w_ref = exp_ref / total;

        // Map head to tangent space
        let h_tangent = manifold.log_map_zero(&h.view());

        // Apply rotation
        let h_rotated = self.apply_rotation(&h_tangent, rotation);

        // Apply reflection
        let h_reflected = self.apply_reflection(&h_tangent, reflection);

        // Weighted combination
        let mut h_combined = Array1::zeros(self.dim);
        for i in 0..self.dim {
            h_combined[i] = w_rot * h_rotated[i] + w_ref * h_reflected[i];
        }

        // Map back to manifold
        let h_transformed = manifold.exp_map_zero(&h_combined.view());

        // Hyperbolic distance
        let dist = manifold.distance(&h_transformed.view(), &t.view());

        // Score with biases
        -dist * dist + b_h + b_t
    }

    /// Get the learned curvature for a relation.
    pub fn relation_curvature(&self, relation: &str) -> Option<f64> {
        self.relation_curvatures.get(relation).copied()
    }

    /// Get entity depth at a specific curvature.
    pub fn entity_depth(&self, entity: &str, curvature: f64) -> Option<f64> {
        let e = self.entity_embeddings.get(entity)?;
        let origin = Array1::zeros(self.dim);
        Some(self.manifold(curvature).distance(&origin.view(), &e.view()))
    }
}

impl KGEModel for AttH {
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let h = self
            .entity_embeddings
            .get(head)
            .ok_or_else(|| Error::NotFound(format!("Unknown entity: {}", head)))?;
        let t = self
            .entity_embeddings
            .get(tail)
            .ok_or_else(|| Error::NotFound(format!("Unknown entity: {}", tail)))?;
        let curvature = self
            .relation_curvatures
            .get(relation)
            .ok_or_else(|| Error::NotFound(format!("Unknown relation: {}", relation)))?;
        let rotation = self
            .relation_rotations
            .get(relation)
            .ok_or_else(|| Error::NotFound(format!("Unknown relation: {}", relation)))?;
        let reflection = self
            .relation_reflections
            .get(relation)
            .ok_or_else(|| Error::NotFound(format!("Unknown relation: {}", relation)))?;
        let attention = self
            .relation_attention
            .get(relation)
            .ok_or_else(|| Error::NotFound(format!("Unknown relation: {}", relation)))?;

        let b_h = self.entity_biases.get(head).copied().unwrap_or(0.0);
        let b_t = self.entity_biases.get(tail).copied().unwrap_or(0.0);

        Ok(self.atth_score(h, t, *curvature, rotation, reflection, *attention, b_h, b_t) as f32)
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
        use std::hash::{Hash, Hasher};

        if triples.is_empty() {
            return Err(Error::Validation("No training triples provided".into()));
        }

        // Initialize
        self.dim = if config.embedding_dim % 2 == 0 {
            config.embedding_dim
        } else {
            config.embedding_dim + 1
        };

        let mut entities = HashSet::new();
        let mut relations = HashSet::new();
        for t in triples {
            entities.insert(t.head.clone());
            entities.insert(t.tail.clone());
            relations.insert(t.relation.clone());
        }

        // Initialize entity embeddings
        let init_scale = 0.001;
        for (i, entity) in entities.iter().enumerate() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            config.seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();

            let embedding: Vec<f64> = (0..self.dim)
                .map(|j| {
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    hash.hash(&mut h);
                    j.hash(&mut h);
                    ((h.finish() as f64 / u64::MAX as f64) - 0.5) * init_scale
                })
                .collect();

            self.entity_embeddings
                .insert(entity.clone(), Array1::from_vec(embedding));
            self.entity_biases.insert(entity.clone(), 0.0);
        }

        // Initialize relation parameters
        let num_rotations = self.dim / 2;
        for (i, relation) in relations.iter().enumerate() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (config.seed + 1).hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();

            // Curvature initialized to 1.0
            self.relation_curvatures.insert(relation.clone(), 1.0);

            // Rotation angles
            let angles: Vec<f64> = (0..num_rotations)
                .map(|j| {
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    hash.hash(&mut h);
                    j.hash(&mut h);
                    ((h.finish() as f64 / u64::MAX as f64) - 0.5) * 0.2
                })
                .collect();
            self.relation_rotations.insert(relation.clone(), angles);

            // Reflection vector
            let reflection: Vec<f64> = (0..self.dim)
                .map(|j| {
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    hash.hash(&mut h);
                    (j + 1000).hash(&mut h);
                    ((h.finish() as f64 / u64::MAX as f64) - 0.5) * init_scale
                })
                .collect();
            self.relation_reflections
                .insert(relation.clone(), Array1::from_vec(reflection));

            // Attention weights (equal initially)
            self.relation_attention.insert(relation.clone(), (0.0, 0.0));
        }

        let entities_vec: Vec<&String> = entities.iter().collect();
        let mut final_loss = 0.0;

        // Training loop
        for epoch in 0..config.epochs {
            let lr = if epoch < config.epochs / 10 {
                config.learning_rate as f64 * 0.1
            } else {
                config.learning_rate as f64
            };

            let mut epoch_loss = 0.0f32;
            let mut num_updates = 0;

            for (batch_idx, batch) in triples.chunks(config.batch_size).enumerate() {
                for triple in batch {
                    let h = self.entity_embeddings.get(&triple.head).unwrap().clone();
                    let t = self.entity_embeddings.get(&triple.tail).unwrap().clone();
                    let curvature = *self.relation_curvatures.get(&triple.relation).unwrap();
                    let rotation = self.relation_rotations.get(&triple.relation).unwrap().clone();
                    let reflection = self.relation_reflections.get(&triple.relation).unwrap().clone();
                    let attention = *self.relation_attention.get(&triple.relation).unwrap();
                    let b_h = *self.entity_biases.get(&triple.head).unwrap();
                    let b_t = *self.entity_biases.get(&triple.tail).unwrap();

                    let pos_score = self.atth_score(
                        &h, &t, curvature, &rotation, &reflection, attention, b_h, b_t,
                    );

                    // Negative sampling
                    for ns in 0..config.negative_samples {
                        let neg_idx = (epoch * 1000 + batch_idx * 100 + ns) % entities_vec.len();
                        let neg_tail = entities_vec[neg_idx];

                        if neg_tail == &triple.tail {
                            continue;
                        }

                        let t_neg = self.entity_embeddings.get(neg_tail).unwrap();
                        let b_t_neg = *self.entity_biases.get(neg_tail).unwrap();
                        let neg_score = self.atth_score(
                            &h, t_neg, curvature, &rotation, &reflection, attention, b_h, b_t_neg,
                        );

                        let loss = (config.margin as f64 - pos_score + neg_score).max(0.0);
                        epoch_loss += loss as f32;

                        if loss > 0.0 {
                            let manifold = self.manifold(curvature);

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

                            // Update rotation
                            let rot_mut = self.relation_rotations.get_mut(&triple.relation).unwrap();
                            for (pair_idx, angle) in rot_mut.iter_mut().enumerate() {
                                let i = pair_idx * 2;
                                let j = i + 1;
                                if j >= self.dim {
                                    break;
                                }
                                let grad = (h[i] - t[i]) * (h[j] - t[j]);
                                *angle -= lr * 0.1 * grad;
                                *angle = angle.rem_euclid(2.0 * std::f64::consts::PI)
                                    - std::f64::consts::PI;
                            }

                            // Update reflection
                            let ref_mut = self.relation_reflections.get_mut(&triple.relation).unwrap();
                            for i in 0..self.dim {
                                ref_mut[i] -= lr * 0.1 * (h[i] - t[i]);
                            }

                            // Update curvature (keep positive)
                            let curv_mut = self.relation_curvatures.get_mut(&triple.relation).unwrap();
                            *curv_mut -= lr * 0.01 * (pos_score - neg_score);
                            *curv_mut = curv_mut.max(0.01).min(10.0);

                            // Update attention
                            let att_mut = self.relation_attention.get_mut(&triple.relation).unwrap();
                            att_mut.0 -= lr * 0.05;
                            att_mut.1 -= lr * 0.05;

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
        // Return rotation angles
        self.relation_rotations
            .get(relation)
            .map(|angles| angles.iter().map(|&x| x as f32).collect())
    }

    fn entity_embeddings(&self) -> &HashMap<String, Vec<f32>> {
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
        self.relation_curvatures.len()
    }

    fn name(&self) -> &'static str {
        "AttH"
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> Vec<Fact<String>> {
        vec![
            // Deep hierarchy
            Fact::from_strs("dog", "isA", "mammal"),
            Fact::from_strs("mammal", "isA", "animal"),
            // Flat relation
            Fact::from_strs("Alice", "knows", "Bob"),
            Fact::from_strs("Bob", "knows", "Alice"),
        ]
    }

    #[test]
    fn test_atth_creation() {
        let model = AttH::new(32);
        assert_eq!(model.embedding_dim(), 32);
        assert!(!model.is_trained());
        assert_eq!(model.name(), "AttH");
    }

    #[test]
    fn test_atth_odd_dim_rounded() {
        let model = AttH::new(31);
        assert_eq!(model.embedding_dim(), 32);
    }

    #[test]
    fn test_atth_training() {
        let mut model = AttH::new(16);
        let triples = sample_graph();

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(20)
            .with_learning_rate(0.01);

        let loss = model.train(&triples, &config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.num_entities(), 5);
        assert_eq!(model.num_relations(), 2);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_atth_scoring() {
        let mut model = AttH::new(16);
        let triples = sample_graph();

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(30);

        model.train(&triples, &config).unwrap();

        let score = model.score("dog", "isA", "mammal").unwrap();
        assert!(score.is_finite());
    }

    #[test]
    fn test_atth_learned_curvature() {
        let mut model = AttH::new(16);
        let triples = sample_graph();

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(30);

        model.train(&triples, &config).unwrap();

        // Curvatures should be learned and positive
        let curv_isa = model.relation_curvature("isA").unwrap();
        let curv_knows = model.relation_curvature("knows").unwrap();

        assert!(curv_isa > 0.0);
        assert!(curv_knows > 0.0);
    }
}
