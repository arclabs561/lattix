//! RotH: Rotation in Hyperbolic space.
//!
//! RotH ([Chami et al. 2020](https://arxiv.org/abs/2005.00545)) models relations
//! as rotations in hyperbolic space, enabling representation of hierarchical
//! and logical patterns simultaneously.
//!
//! # Key Insight
//!
//! Different relation patterns require different geometric transformations:
//! - **Hierarchies**: Hyperbolic geometry captures tree-like structures
//! - **Symmetry/antisymmetry**: Rotations can model symmetric relations
//! - **Composition**: Rotation composition follows relation composition
//!
//! # Scoring
//!
//! ```text
//! score = -d_H(R_r * h, t)^2 + b_h + b_t
//! ```
//!
//! where:
//! - `d_H` is hyperbolic distance
//! - `R_r` is the relation-specific rotation (applied in tangent space)
//! - `b_h, b_t` are learnable entity biases
//!
//! # Rotation Implementation
//!
//! Rotations are applied via:
//! 1. Map head to tangent space at origin (log map)
//! 2. Apply Givens rotation matrix
//! 3. Map back to manifold (exp map)
//!
//! This preserves the hyperbolic metric since rotations in tangent space
//! correspond to isometries on the manifold.
//!
//! # When to Use
//!
//! - **Hierarchies with symmetric relations**: Family trees, taxonomies
//! - **Relation composition**: Multi-hop reasoning
//! - **Low dimensions**: RotH excels with dim < 50
//!
//! # Comparison with MuRP
//!
//! | Model | Transformation | Best for |
//! |-------|---------------|----------|
//! | MuRP | Diagonal scaling + translation | Pure hierarchies |
//! | RotH | Rotation | Hierarchies + logical patterns |
//!
//! # Example
//!
//! ```rust,ignore
//! use lattix_kge::models::RotH;
//! use lattix_kge::{KGEModel, Fact, TrainingConfig};
//!
//! let mut model = RotH::new(32, 1.0);
//!
//! let triples = vec![
//!     Fact::from_strs("Alice", "friendOf", "Bob"),  // symmetric
//!     Fact::from_strs("cat", "isA", "mammal"),      // hierarchical
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

/// RotH model: Rotation in Hyperbolic space.
///
/// Relations are modeled as rotations, applied in the tangent space
/// of the Poincare ball at the origin.
pub struct RotH {
    /// Embedding dimension.
    dim: usize,
    /// Curvature of hyperbolic space.
    curvature: f64,
    /// Entity embeddings (points in Poincare ball).
    entity_embeddings: HashMap<String, Array1<f64>>,
    /// Entity biases.
    entity_biases: HashMap<String, f64>,
    /// Relation rotation angles (Givens rotations).
    /// Each relation has (dim/2) rotation angles for pairs of dimensions.
    relation_rotations: HashMap<String, Vec<f64>>,
    /// Whether trained.
    trained: bool,
}

impl std::fmt::Debug for RotH {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RotH")
            .field("dim", &self.dim)
            .field("curvature", &self.curvature)
            .field("num_entities", &self.entity_embeddings.len())
            .field("num_relations", &self.relation_rotations.len())
            .field("trained", &self.trained)
            .finish()
    }
}

impl Clone for RotH {
    fn clone(&self) -> Self {
        Self {
            dim: self.dim,
            curvature: self.curvature,
            entity_embeddings: self.entity_embeddings.clone(),
            entity_biases: self.entity_biases.clone(),
            relation_rotations: self.relation_rotations.clone(),
            trained: self.trained,
        }
    }
}

impl RotH {
    /// Create a new RotH model.
    ///
    /// # Arguments
    /// * `dim` - Embedding dimension (should be even for Givens rotations)
    /// * `curvature` - Hyperbolic curvature (1.0 = standard)
    pub fn new(dim: usize, curvature: f64) -> Self {
        // Round up to even dimension for Givens rotations
        let dim = if dim % 2 == 0 { dim } else { dim + 1 };
        Self {
            dim,
            curvature,
            entity_embeddings: HashMap::new(),
            entity_biases: HashMap::new(),
            relation_rotations: HashMap::new(),
            trained: false,
        }
    }

    /// Get the Poincare ball manifold.
    #[inline]
    fn manifold(&self) -> PoincareBall {
        PoincareBall::new(self.curvature)
    }

    /// Apply Givens rotation to a vector in tangent space.
    ///
    /// Givens rotations are applied to pairs of dimensions (0,1), (2,3), ...
    /// This gives (dim/2) independent rotation angles.
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

    /// RotH scoring function.
    ///
    /// score = -d_H(R_r * h, t)^2 + b_h + b_t
    fn roth_score(
        &self,
        h: &Array1<f64>,
        t: &Array1<f64>,
        rotation: &[f64],
        b_h: f64,
        b_t: f64,
    ) -> f64 {
        let manifold = self.manifold();

        // Map head to tangent space at origin
        let h_tangent = manifold.log_map_zero(&h.view());

        // Apply rotation in tangent space
        let h_rotated_tangent = self.apply_rotation(&h_tangent, rotation);

        // Map back to manifold
        let h_rotated = manifold.exp_map_zero(&h_rotated_tangent.view());

        // Hyperbolic distance squared
        let dist = manifold.distance(&h_rotated.view(), &t.view());

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

    /// Initialize relation rotation angles.
    fn init_relation_rotations(
        &self,
        vocab: &HashSet<String>,
        seed: u64,
    ) -> HashMap<String, Vec<f64>> {
        use std::hash::{Hash, Hasher};

        let num_rotations = self.dim / 2;
        let mut rotations = HashMap::new();

        for (i, item) in vocab.iter().enumerate() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();

            // Initialize angles near 0 (small initial rotations)
            let mut angles = Vec::with_capacity(num_rotations);
            for j in 0..num_rotations {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                hash.hash(&mut h);
                j.hash(&mut h);
                let val = h.finish();
                // Small random angle in [-0.1, 0.1]
                angles.push(((val as f64 / u64::MAX as f64) - 0.5) * 0.2);
            }

            rotations.insert(item.clone(), angles);
        }

        rotations
    }
}

impl KGEModel for RotH {
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let h = self
            .entity_embeddings
            .get(head)
            .ok_or_else(|| Error::NotFound(format!("Unknown entity: {}", head)))?;
        let t = self
            .entity_embeddings
            .get(tail)
            .ok_or_else(|| Error::NotFound(format!("Unknown entity: {}", tail)))?;
        let rotation = self
            .relation_rotations
            .get(relation)
            .ok_or_else(|| Error::NotFound(format!("Unknown relation: {}", relation)))?;

        let b_h = self.entity_biases.get(head).copied().unwrap_or(0.0);
        let b_t = self.entity_biases.get(tail).copied().unwrap_or(0.0);

        Ok(self.roth_score(h, t, rotation, b_h, b_t) as f32)
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
        self.dim = if config.embedding_dim % 2 == 0 {
            config.embedding_dim
        } else {
            config.embedding_dim + 1
        };

        let (entity_emb, entity_bias) = self.init_entity_embeddings(&entities, config.seed);
        let rel_rot = self.init_relation_rotations(&relations, config.seed + 1);

        self.entity_embeddings = entity_emb;
        self.entity_biases = entity_bias;
        self.relation_rotations = rel_rot;

        let manifold = self.manifold();
        let entities_vec: Vec<&String> = entities.iter().collect();
        let mut final_loss = 0.0;

        // Burn-in period
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
                    let rotation = self
                        .relation_rotations
                        .get(&triple.relation)
                        .unwrap()
                        .clone();
                    let b_h = *self.entity_biases.get(&triple.head).unwrap();
                    let b_t = *self.entity_biases.get(&triple.tail).unwrap();

                    let pos_score = self.roth_score(&h, &t, &rotation, b_h, b_t);

                    // Negative sampling
                    for ns in 0..config.negative_samples {
                        let neg_idx = (epoch * 1000 + batch_idx * 100 + ns) % entities_vec.len();
                        let neg_tail = entities_vec[neg_idx];

                        if neg_tail == &triple.tail {
                            continue;
                        }

                        let t_neg = self.entity_embeddings.get(neg_tail).unwrap();
                        let b_t_neg = *self.entity_biases.get(neg_tail).unwrap();
                        let neg_score = self.roth_score(&h, t_neg, &rotation, b_h, b_t_neg);

                        // Margin-based ranking loss
                        let loss = (config.margin as f64 - pos_score + neg_score).max(0.0);
                        epoch_loss += loss as f32;

                        if loss > 0.0 {
                            // Gradient updates (simplified)
                            // In practice, use Riemannian SGD for proper hyperbolic gradients

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

                            // Update rotation angles (gradient through rotation is complex,
                            // simplified here)
                            let rot_mut =
                                self.relation_rotations.get_mut(&triple.relation).unwrap();
                            for (pair_idx, angle) in rot_mut.iter_mut().enumerate() {
                                let i = pair_idx * 2;
                                let j = i + 1;
                                if j >= self.dim {
                                    break;
                                }
                                // Push angle toward value that brings h closer to t
                                let grad = (h[i] - t[i]) * (h[j] - t[j]);
                                *angle -= lr * 0.1 * grad;
                                // Keep angle in [-pi, pi]
                                *angle = angle.rem_euclid(2.0 * std::f64::consts::PI)
                                    - std::f64::consts::PI;
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
        // Return rotation angles as the "embedding"
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
        self.relation_rotations.len()
    }

    fn name(&self) -> &'static str {
        "RotH"
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
            // Symmetric relation
            Fact::from_strs("Alice", "friendOf", "Bob"),
            Fact::from_strs("Bob", "friendOf", "Alice"),
            // Hierarchical relation
            Fact::from_strs("cat", "isA", "mammal"),
            Fact::from_strs("mammal", "isA", "animal"),
            Fact::from_strs("dog", "isA", "mammal"),
        ]
    }

    #[test]
    fn test_roth_creation() {
        let model = RotH::new(32, 1.0);
        assert_eq!(model.embedding_dim(), 32);
        assert!(!model.is_trained());
        assert_eq!(model.name(), "RotH");
    }

    #[test]
    fn test_roth_odd_dim_rounded() {
        let model = RotH::new(31, 1.0);
        // Should round up to even dimension
        assert_eq!(model.embedding_dim(), 32);
    }

    #[test]
    fn test_roth_training() {
        let mut model = RotH::new(16, 1.0);
        let triples = sample_graph();

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(20)
            .with_learning_rate(0.01);

        let loss = model.train(&triples, &config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.num_entities(), 6); // Alice, Bob, cat, dog, mammal, animal
        assert_eq!(model.num_relations(), 2); // friendOf, isA
        assert!(loss.is_finite());
    }

    #[test]
    fn test_roth_scoring() {
        let mut model = RotH::new(16, 1.0);
        let triples = sample_graph();

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(30);

        model.train(&triples, &config).unwrap();

        let score = model.score("cat", "isA", "mammal").unwrap();
        assert!(score.is_finite());
    }

    #[test]
    fn test_roth_givens_rotation() {
        let model = RotH::new(4, 1.0);

        // Test rotation by pi/2 on first pair of dims
        let v = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let angles = vec![std::f64::consts::FRAC_PI_2, 0.0];

        let rotated = model.apply_rotation(&v, &angles);

        // After 90-degree rotation: (1,0) -> (0,1)
        assert!((rotated[0] - 0.0).abs() < 1e-10);
        assert!((rotated[1] - 1.0).abs() < 1e-10);
        assert!((rotated[2] - 0.0).abs() < 1e-10);
        assert!((rotated[3] - 0.0).abs() < 1e-10);
    }
}
