//! BoxE: Relations as Boxes.
//!
//! BoxE ([Abboud et al. 2020](https://arxiv.org/abs/2007.06267)) represents entities
//! as points and relations as axis-aligned boxes. A triple (h, r, t) is plausible if
//! the head (translated by relation bump) falls inside the relation's box and is
//! close to the tail.
//!
//! # Key Advantages Over TransE
//!
//! - **Fully expressive**: Can model any knowledge graph
//! - **Handles all relation patterns**: Symmetric, antisymmetric, inverse, composition
//! - **Natural containment**: Box intersection/containment for hierarchies
//!
//! # Scoring
//!
//! BoxE uses distance from translated head to tail, penalized by box violation:
//!
//! ```text
//! score = -||h + bump - t||₂ - λ * box_violation(h + bump, box)
//! ```
//!
//! where `box_violation` measures how far the point is outside the box.
//!
//! # Example
//!
//! ```rust,ignore
//! use lattix_kge::models::BoxE;
//! use lattix_kge::{KGEModel, Fact, TrainingConfig};
//!
//! let mut model = BoxE::new(128);
//!
//! let triples = vec![
//!     Fact::from_strs("Cat", "isA", "Animal"),
//!     Fact::from_strs("Dog", "isA", "Animal"),
//! ];
//!
//! model.train(&triples, &TrainingConfig::default())?;
//! ```

use crate::error::{Error, Result};
use crate::model::{EpochMetrics, Fact, KGEModel, ProgressCallback};
use crate::training::TrainingConfig;
use std::collections::{HashMap, HashSet};

/// BoxE model: entities as points, relations as boxes + bumps.
///
/// This is the ndarray-based implementation.
#[derive(Debug, Clone)]
pub struct BoxE {
    /// Embedding dimension.
    dim: usize,
    /// Entity embeddings (point representations).
    entity_embeddings: HashMap<String, Vec<f32>>,
    /// Relation boxes: (center, offset) where box = [center - offset, center + offset].
    relation_boxes: HashMap<String, (Vec<f32>, Vec<f32>)>,
    /// Relation bumps (translation vectors).
    relation_bumps: HashMap<String, Vec<f32>>,
    /// Box violation penalty weight.
    box_penalty: f32,
    /// Whether the model has been trained.
    trained: bool,
}

impl Default for BoxE {
    fn default() -> Self {
        Self::new(128)
    }
}

impl BoxE {
    /// Create a new BoxE model with the given embedding dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entity_embeddings: HashMap::new(),
            relation_boxes: HashMap::new(),
            relation_bumps: HashMap::new(),
            box_penalty: 0.5,
            trained: false,
        }
    }

    /// Set box violation penalty weight.
    pub fn with_box_penalty(mut self, penalty: f32) -> Self {
        self.box_penalty = penalty;
        self
    }

    /// Compute distance from point to tail, plus box violation.
    fn boxe_score(&self, h: &[f32], bump: &[f32], center: &[f32], offset: &[f32], t: &[f32]) -> f32 {
        let mut dist_sq = 0.0;
        let mut violation = 0.0;

        for i in 0..self.dim {
            let translated = h[i] + bump[i];
            let diff = translated - t[i];
            dist_sq += diff * diff;

            // Box violation: how far outside [center - offset, center + offset]
            let lower = center[i] - offset[i].abs();
            let upper = center[i] + offset[i].abs();
            if translated < lower {
                violation += (lower - translated).powi(2);
            } else if translated > upper {
                violation += (translated - upper).powi(2);
            }
        }

        // Negative distance + penalty (higher = more plausible)
        -(dist_sq.sqrt() + self.box_penalty * violation.sqrt())
    }

    /// Initialize point embeddings.
    fn init_points(&self, vocab: &HashSet<String>, seed: u64) -> HashMap<String, Vec<f32>> {
        use std::hash::{Hash, Hasher};

        let mut embeddings = HashMap::new();

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
                let f = (val as f64 / u64::MAX as f64) - 0.5;
                embedding.push(f as f32);
            }

            embeddings.insert(item.clone(), embedding);
        }

        embeddings
    }

    /// Initialize relation boxes (center, offset) and bumps.
    fn init_relations(
        &self,
        vocab: &HashSet<String>,
        seed: u64,
    ) -> (
        HashMap<String, (Vec<f32>, Vec<f32>)>,
        HashMap<String, Vec<f32>>,
    ) {
        use std::hash::{Hash, Hasher};

        let mut boxes = HashMap::new();
        let mut bumps = HashMap::new();

        for (i, item) in vocab.iter().enumerate() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();

            let mut center = Vec::with_capacity(self.dim);
            let mut offset = Vec::with_capacity(self.dim);
            let mut bump = Vec::with_capacity(self.dim);

            for j in 0..self.dim {
                // Center
                let mut h = std::collections::hash_map::DefaultHasher::new();
                hash.hash(&mut h);
                j.hash(&mut h);
                let val = h.finish();
                center.push(((val as f64 / u64::MAX as f64) - 0.5) as f32);

                // Offset (always positive)
                let mut h2 = std::collections::hash_map::DefaultHasher::new();
                hash.hash(&mut h2);
                (j + self.dim).hash(&mut h2);
                let val2 = h2.finish();
                offset.push((val2 as f64 / u64::MAX as f64 * 0.5 + 0.1) as f32);

                // Bump
                let mut h3 = std::collections::hash_map::DefaultHasher::new();
                hash.hash(&mut h3);
                (j + 2 * self.dim).hash(&mut h3);
                let val3 = h3.finish();
                bump.push(((val3 as f64 / u64::MAX as f64) - 0.5) as f32 * 0.1);
            }

            boxes.insert(item.clone(), (center, offset));
            bumps.insert(item.clone(), bump);
        }

        (boxes, bumps)
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
}

impl KGEModel for BoxE {
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let h = self
            .entity_embeddings
            .get(head)
            .ok_or_else(|| Error::NotFound(format!("Unknown entity: {}", head)))?;
        let (center, offset) = self
            .relation_boxes
            .get(relation)
            .ok_or_else(|| Error::NotFound(format!("Unknown relation: {}", relation)))?;
        let bump = self
            .relation_bumps
            .get(relation)
            .ok_or_else(|| Error::NotFound(format!("Unknown relation: {}", relation)))?;
        let t = self
            .entity_embeddings
            .get(tail)
            .ok_or_else(|| Error::NotFound(format!("Unknown entity: {}", tail)))?;

        Ok(self.boxe_score(h, bump, center, offset, t))
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

        let (entities, relations) = self.extract_vocab(triples);
        self.dim = config.embedding_dim;
        self.entity_embeddings = self.init_points(&entities, config.seed);
        let (boxes, bumps) = self.init_relations(&relations, config.seed + 1);
        self.relation_boxes = boxes;
        self.relation_bumps = bumps;

        let entities_vec: Vec<&String> = entities.iter().collect();
        let mut final_loss = 0.0;

        for epoch in 0..config.epochs {
            let mut epoch_loss = 0.0;
            let mut num_updates = 0;

            for (batch_idx, batch) in triples.chunks(config.batch_size).enumerate() {
                for triple in batch {
                    let h = self.entity_embeddings.get(&triple.head).unwrap().clone();
                    let t = self.entity_embeddings.get(&triple.tail).unwrap().clone();
                    let (center, offset) = self.relation_boxes.get(&triple.relation).unwrap().clone();
                    let bump = self.relation_bumps.get(&triple.relation).unwrap().clone();

                    let pos_score = self.boxe_score(&h, &bump, &center, &offset, &t);

                    // Negative sampling
                    for ns in 0..config.negative_samples {
                        let neg_idx = (epoch * 1000 + batch_idx * 100 + ns) % entities_vec.len();
                        let neg_tail = entities_vec[neg_idx];

                        if neg_tail == &triple.tail {
                            continue;
                        }

                        let t_neg = self.entity_embeddings.get(neg_tail).unwrap();
                        let neg_score = self.boxe_score(&h, &bump, &center, &offset, t_neg);

                        // Margin-based ranking loss: want pos_score > neg_score
                        let loss = (config.margin - pos_score + neg_score).max(0.0);
                        epoch_loss += loss;

                        if loss > 0.0 {
                            let lr = config.learning_rate;

                            // Simplified gradient update for entity points
                            let h_mut = self.entity_embeddings.get_mut(&triple.head).unwrap();
                            for i in 0..self.dim {
                                let grad = 2.0 * ((h[i] + bump[i]) - t[i]);
                                h_mut[i] -= lr * grad;
                            }

                            let t_mut = self.entity_embeddings.get_mut(&triple.tail).unwrap();
                            for i in 0..self.dim {
                                let grad = -2.0 * ((h[i] + bump[i]) - t[i]);
                                t_mut[i] -= lr * grad;
                            }

                            // Update bump
                            let bump_mut = self.relation_bumps.get_mut(&triple.relation).unwrap();
                            for i in 0..self.dim {
                                let grad = 2.0 * ((h[i] + bump[i]) - t[i]);
                                bump_mut[i] -= lr * grad * 0.1; // Slower update for bumps
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
        // Return the bump as the "embedding" for compatibility
        self.relation_bumps.get(relation).cloned()
    }

    fn entity_embeddings(&self) -> &HashMap<String, Vec<f32>> {
        &self.entity_embeddings
    }

    fn relation_embeddings(&self) -> &HashMap<String, Vec<f32>> {
        &self.relation_bumps
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn num_entities(&self) -> usize {
        self.entity_embeddings.len()
    }

    fn num_relations(&self) -> usize {
        self.relation_boxes.len()
    }

    fn name(&self) -> &'static str {
        "BoxE"
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_triples() -> Vec<Fact<String>> {
        vec![
            Fact::from_strs("Cat", "isA", "Animal"),
            Fact::from_strs("Dog", "isA", "Animal"),
            Fact::from_strs("Animal", "isA", "LivingThing"),
            Fact::from_strs("Cat", "has", "Fur"),
            Fact::from_strs("Dog", "has", "Fur"),
        ]
    }

    #[test]
    fn test_boxe_creation() {
        let model = BoxE::new(64);
        assert_eq!(model.embedding_dim(), 64);
        assert!(!model.is_trained());
        assert_eq!(model.name(), "BoxE");
    }

    #[test]
    fn test_boxe_training() {
        let mut model = BoxE::new(32);
        let triples = sample_triples();

        let config = TrainingConfig::default()
            .with_embedding_dim(32)
            .with_epochs(10)
            .with_learning_rate(0.01);

        let loss = model.train(&triples, &config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.num_entities(), 5); // Cat, Dog, Animal, LivingThing, Fur
        assert_eq!(model.num_relations(), 2); // isA, has
        assert!(loss.is_finite());
    }

    #[test]
    fn test_boxe_scoring() {
        let mut model = BoxE::new(32);
        let triples = sample_triples();

        let config = TrainingConfig::default()
            .with_embedding_dim(32)
            .with_epochs(20);

        model.train(&triples, &config).unwrap();

        let score = model.score("Cat", "isA", "Animal").unwrap();
        assert!(score.is_finite());
    }

    #[test]
    fn test_boxe_with_penalty() {
        let model = BoxE::new(32).with_box_penalty(1.0);
        assert!((model.box_penalty - 1.0).abs() < f32::EPSILON);
    }
}
