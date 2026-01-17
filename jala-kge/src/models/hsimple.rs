//! HSimplE: Hypergraph SimplE via Tucker Decomposition.
//!
//! HSimplE extends SimplE to hypergraphs using Tucker decomposition to model
//! n-ary relations. Unlike binary KGE models, HSimplE can directly represent
//! relations involving more than two entities.
//!
//! # Key Idea
//!
//! For an n-ary relation r with entities (e_1, ..., e_n), the score is computed
//! via Tucker decomposition:
//!
//! ```text
//! score = W_r ×_1 e_1 ×_2 e_2 ... ×_n e_n
//! ```
//!
//! where W_r is a relation-specific core tensor and ×_i denotes mode-i product.
//!
//! # Simplified Implementation
//!
//! For efficiency, we use a factorized approximation:
//!
//! ```text
//! score = sum_k( r_k * prod_i(e_i[k]) )
//! ```
//!
//! This is equivalent to CP (CANDECOMP/PARAFAC) decomposition, a special case
//! of Tucker where the core tensor is superdiagonal.
//!
//! # When to Use
//!
//! - N-ary relations (e.g., "Alice bought Book from Store on Monday")
//! - Drug-drug-gene interactions
//! - Multi-entity events
//!
//! # Example
//!
//! ```rust,ignore
//! use lattix_kge::models::HSimplE;
//! use lattix_kge::{KGEModel, Fact, TrainingConfig};
//! use lattix_kge::models::HyperFact;
//!
//! let mut model = HSimplE::new(32, 4);  // 32-dim, max 4 entities
//!
//! // N-ary fact: (Alice, bought, Book, Store)
//! let fact = HyperFact::new("bought")
//!     .with_entity(0, "Alice")
//!     .with_entity(1, "Book")
//!     .with_entity(2, "Store");
//!
//! // Train on hyperedges
//! model.train_hyperedges(&[fact], &TrainingConfig::default())?;
//! ```

use crate::error::{Error, Result};
use crate::model::{EpochMetrics, Fact, KGEModel, ProgressCallback};
use crate::models::HyperFact;
use crate::training::TrainingConfig;
use std::collections::{HashMap, HashSet};

/// HSimplE model: Hypergraph embedding via Tucker/CP decomposition.
///
/// Uses a factorized scoring function for n-ary relations.
#[derive(Debug, Clone)]
pub struct HSimplE {
    /// Embedding dimension.
    dim: usize,
    /// Maximum arity (number of positions).
    max_arity: usize,
    /// Entity embeddings.
    entity_embeddings: HashMap<String, Vec<f32>>,
    /// Relation embeddings (diagonal of core tensor).
    relation_embeddings: HashMap<String, Vec<f32>>,
    /// Position-specific projection matrices (flattened).
    /// Each position has a dim x dim matrix stored as Vec<f32>.
    position_projections: Vec<Vec<f32>>,
    /// Whether trained.
    trained: bool,
}

impl HSimplE {
    /// Create a new HSimplE model.
    ///
    /// # Arguments
    /// * `dim` - Embedding dimension
    /// * `max_arity` - Maximum number of entities in a hyperedge
    pub fn new(dim: usize, max_arity: usize) -> Self {
        Self {
            dim,
            max_arity,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            position_projections: Vec::new(),
            trained: false,
        }
    }

    /// Score a hyperfact using CP decomposition.
    ///
    /// score = sum_k( r[k] * prod_{i in positions}( P_i @ e_i )[k] )
    pub fn score_hyperfact(&self, fact: &HyperFact) -> Result<f32> {
        let r = self
            .relation_embeddings
            .get(&fact.relation)
            .ok_or_else(|| Error::NotFound(format!("Unknown relation: {}", fact.relation)))?;

        // Get projected entity embeddings for each position
        let mut projected: Vec<Vec<f32>> = Vec::new();

        for (pos, entity) in &fact.entities {
            if *pos >= self.max_arity {
                return Err(Error::Validation(format!(
                    "Position {} exceeds max_arity {}",
                    pos, self.max_arity
                )));
            }

            let e = self
                .entity_embeddings
                .get(entity)
                .ok_or_else(|| Error::NotFound(format!("Unknown entity: {}", entity)))?;

            // Apply position-specific projection
            let proj = if *pos < self.position_projections.len() {
                self.apply_projection(e, &self.position_projections[*pos])
            } else {
                e.clone() // Identity if no projection
            };

            projected.push(proj);
        }

        // CP score: sum_k( r[k] * prod_i(proj_i[k]) )
        let mut score = 0.0f32;
        for k in 0..self.dim {
            let mut prod = r[k];
            for proj in &projected {
                prod *= proj[k];
            }
            score += prod;
        }

        Ok(score)
    }

    /// Apply a projection matrix (stored as flattened dim x dim).
    fn apply_projection(&self, v: &[f32], proj: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                result[i] += proj[i * self.dim + j] * v[j];
            }
        }
        result
    }

    /// Train on hyperedges.
    pub fn train_hyperedges(
        &mut self,
        facts: &[HyperFact],
        config: &TrainingConfig,
    ) -> Result<f32> {
        self.train_hyperedges_with_callback(facts, config, Box::new(|_, _| {}))
    }

    /// Train on hyperedges with progress callback.
    pub fn train_hyperedges_with_callback(
        &mut self,
        facts: &[HyperFact],
        config: &TrainingConfig,
        callback: ProgressCallback,
    ) -> Result<f32> {
        use std::hash::{Hash, Hasher};

        if facts.is_empty() {
            return Err(Error::Validation("No training facts provided".into()));
        }

        self.dim = config.embedding_dim;

        // Extract vocabulary
        let mut entities = HashSet::new();
        let mut relations = HashSet::new();
        let mut max_pos = 0usize;

        for fact in facts {
            relations.insert(fact.relation.clone());
            for (pos, entity) in &fact.entities {
                entities.insert(entity.clone());
                max_pos = max_pos.max(*pos);
            }
        }

        self.max_arity = self.max_arity.max(max_pos + 1);

        // Initialize embeddings
        let init_scale = 0.1;
        for (i, entity) in entities.iter().enumerate() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            config.seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();

            let embedding: Vec<f32> = (0..self.dim)
                .map(|j| {
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    hash.hash(&mut h);
                    j.hash(&mut h);
                    ((h.finish() as f64 / u64::MAX as f64) - 0.5) as f32 * init_scale
                })
                .collect();

            self.entity_embeddings.insert(entity.clone(), embedding);
        }

        for (i, relation) in relations.iter().enumerate() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (config.seed + 1).hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();

            let embedding: Vec<f32> = (0..self.dim)
                .map(|j| {
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    hash.hash(&mut h);
                    j.hash(&mut h);
                    ((h.finish() as f64 / u64::MAX as f64) - 0.5) as f32 * init_scale
                })
                .collect();

            self.relation_embeddings.insert(relation.clone(), embedding);
        }

        // Initialize position projections (identity + noise)
        self.position_projections = (0..self.max_arity)
            .map(|pos| {
                let mut proj = vec![0.0f32; self.dim * self.dim];
                // Start near identity
                for i in 0..self.dim {
                    proj[i * self.dim + i] = 1.0;
                }
                // Add small noise
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                (config.seed + 2).hash(&mut hasher);
                pos.hash(&mut hasher);
                let hash = hasher.finish();

                for (k, v) in proj.iter_mut().enumerate() {
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    hash.hash(&mut h);
                    k.hash(&mut h);
                    *v += ((h.finish() as f64 / u64::MAX as f64) - 0.5) as f32 * 0.01;
                }
                proj
            })
            .collect();

        let entities_vec: Vec<&String> = entities.iter().collect();
        let mut final_loss = 0.0;

        // Training loop
        for epoch in 0..config.epochs {
            let lr = config.learning_rate;
            let mut epoch_loss = 0.0f32;
            let mut num_updates = 0;

            for (batch_idx, batch) in facts.chunks(config.batch_size).enumerate() {
                for fact in batch {
                    let pos_score = self.score_hyperfact(fact)?;

                    // Negative sampling: corrupt one entity
                    for ns in 0..config.negative_samples {
                        let neg_idx = (epoch * 1000 + batch_idx * 100 + ns) % entities_vec.len();
                        let neg_entity = entities_vec[neg_idx];

                        // Corrupt a random position
                        let corrupt_pos = ns % fact.entities.len();
                        let mut neg_fact = fact.clone();
                        let positions: Vec<_> = neg_fact.entities.keys().copied().collect();
                        if let Some(&pos) = positions.get(corrupt_pos) {
                            neg_fact.entities.insert(pos, neg_entity.clone());
                        }

                        let neg_score = self.score_hyperfact(&neg_fact)?;

                        // Margin loss
                        let loss = (config.margin - pos_score + neg_score).max(0.0);
                        epoch_loss += loss;

                        if loss > 0.0 {
                            // Gradient updates (simplified)
                            // Update relation embedding
                            let r = self
                                .relation_embeddings
                                .get_mut(&fact.relation)
                                .ok_or_else(|| Error::RelationNotFound(fact.relation.clone()))?;
                            for v in r.iter_mut() {
                                *v += lr * 0.1;
                            }

                            // Update entity embeddings
                            for entity in fact.entities.values() {
                                let e = self
                                    .entity_embeddings
                                    .get_mut(entity)
                                    .ok_or_else(|| Error::EntityNotFound(entity.clone()))?;
                                for v in e.iter_mut() {
                                    *v += lr * 0.1;
                                }
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
}

impl KGEModel for HSimplE {
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        // Convert triple to 2-ary hyperfact
        let fact = HyperFact::from_triple(head, relation, tail);
        self.score_hyperfact(&fact)
    }

    fn train(&mut self, triples: &[Fact<String>], config: &TrainingConfig) -> Result<f32> {
        // Convert to hyperfacts
        let facts: Vec<HyperFact> = triples
            .iter()
            .map(|t| HyperFact::from_triple(&t.head, &t.relation, &t.tail))
            .collect();
        self.train_hyperedges(&facts, config)
    }

    fn train_with_callback(
        &mut self,
        triples: &[Fact<String>],
        config: &TrainingConfig,
        callback: ProgressCallback,
    ) -> Result<f32> {
        let facts: Vec<HyperFact> = triples
            .iter()
            .map(|t| HyperFact::from_triple(&t.head, &t.relation, &t.tail))
            .collect();
        self.train_hyperedges_with_callback(&facts, config, callback)
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
        "HSimplE"
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn test_hsimple_creation() {
        let model = HSimplE::new(32, 4);
        assert_eq!(model.embedding_dim(), 32);
        assert_eq!(model.max_arity, 4);
        assert!(!model.is_trained());
        assert_eq!(model.name(), "HSimplE");
    }

    #[test]
    fn test_hsimple_triple_training() {
        let mut model = HSimplE::new(16, 2);
        let triples = vec![
            Fact::from_strs("Alice", "knows", "Bob"),
            Fact::from_strs("Bob", "knows", "Carol"),
        ];

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(20);

        let loss = model.train(&triples, &config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.num_entities(), 3);
        assert_eq!(model.num_relations(), 1);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_hsimple_hyperedge_training() {
        let mut model = HSimplE::new(16, 4);

        // N-ary facts: (buyer, item, seller)
        let facts = vec![
            HyperFact::new("bought")
                .with_entity(0, "Alice")
                .with_entity(1, "Book")
                .with_entity(2, "Store"),
            HyperFact::new("bought")
                .with_entity(0, "Bob")
                .with_entity(1, "Laptop")
                .with_entity(2, "Amazon"),
        ];

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(30);

        let loss = model.train_hyperedges(&facts, &config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.num_entities(), 6);
        assert_eq!(model.num_relations(), 1);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_hsimple_scoring() {
        let mut model = HSimplE::new(16, 4);

        let facts = vec![HyperFact::new("transaction")
            .with_entity(0, "Alice")
            .with_entity(1, "Item")
            .with_entity(2, "Bob")];

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(20);

        model.train_hyperedges(&facts, &config).unwrap();

        let score = model.score_hyperfact(&facts[0]).unwrap();
        assert!(score.is_finite());
    }

    #[test]
    fn test_hsimple_position_projections() {
        let mut model = HSimplE::new(8, 3);

        let facts = vec![HyperFact::new("rel")
            .with_entity(0, "A")
            .with_entity(1, "B")
            .with_entity(2, "C")];

        let config = TrainingConfig::default()
            .with_embedding_dim(8)
            .with_epochs(10);

        model.train_hyperedges(&facts, &config).unwrap();

        // Should have 3 position projections
        assert_eq!(model.position_projections.len(), 3);
        // Each projection is dim x dim
        assert_eq!(model.position_projections[0].len(), 64);
    }
}
