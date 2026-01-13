//! HypE: Knowledge Hypergraph Embedding via Position-Specific Convolution.
//!
//! HypE ([Fatemi et al. 2019](https://arxiv.org/abs/1906.00137)) extends knowledge
//! graph embedding to hypergraphs where relations can connect any number of entities.
//!
//! # Key Insight
//!
//! Traditional KGE models are limited to triples (binary relations).
//! Real-world knowledge often involves n-ary relations:
//!
//! - `purchase(buyer, seller, item, price, date)` - 5-ary
//! - `acted_in(actor, movie, role)` - 3-ary (ternary)
//! - `played_for(player, team, position, start_year, end_year)` - 5-ary
//!
//! HypE handles this via position-specific convolutional filters.
//!
//! # Scoring Function
//!
//! For a hyperedge `(e_1, e_2, ..., e_n, r)`:
//!
//! ```text
//! score = sum_i(conv(e_i, F_r^i))
//! ```
//!
//! where `F_r^i` is the position-i filter for relation r.
//!
//! # Implementation Note
//!
//! This implementation supports hyperedges stored as [`HyperFact`] structures.
//! For compatibility with triple-based APIs, triples are treated as 2-ary
//! hyperedges (head at position 0, tail at position 1).
//!
//! # Example
//!
//! ```rust,ignore
//! use lattix_kge::models::HypE;
//! use lattix_kge::{KGEModel, HyperFact, TrainingConfig};
//!
//! let mut model = HypE::new(32, 5);  // 32-dim, max 5 positions
//!
//! // N-ary fact
//! let fact = HyperFact::new("purchase")
//!     .with_entity(0, "Alice")
//!     .with_entity(1, "Amazon")
//!     .with_entity(2, "Rust Book");
//!
//! model.train_hyperedges(&[fact], &TrainingConfig::default())?;
//! ```
//!
//! # References
//!
//! - Fatemi et al. (2019). "Knowledge Hypergraphs: Prediction Beyond Binary Relations"

use crate::error::{Error, Result};
use crate::model::{EpochMetrics, Fact, KGEModel, ProgressCallback};
use crate::training::TrainingConfig;
use std::collections::{HashMap, HashSet};

/// A hyperedge fact: relation with entities at numbered positions.
///
/// Position 0 is typically the "subject", but all positions are symmetric
/// in HypE's scoring function.
#[derive(Debug, Clone)]
pub struct HyperFact {
    /// Relation name.
    pub relation: String,
    /// Entities at each position (sparse: not all positions need entities).
    pub entities: HashMap<usize, String>,
}

impl HyperFact {
    /// Create a new hyperfact with given relation.
    pub fn new(relation: impl Into<String>) -> Self {
        Self {
            relation: relation.into(),
            entities: HashMap::new(),
        }
    }

    /// Add an entity at a specific position.
    pub fn with_entity(mut self, position: usize, entity: impl Into<String>) -> Self {
        self.entities.insert(position, entity.into());
        self
    }

    /// Create from a triple (binary relation).
    pub fn from_triple(head: impl Into<String>, relation: impl Into<String>, tail: impl Into<String>) -> Self {
        Self {
            relation: relation.into(),
            entities: [(0, head.into()), (1, tail.into())].into_iter().collect(),
        }
    }

    /// Get arity (number of entity positions filled).
    pub fn arity(&self) -> usize {
        self.entities.len()
    }

    /// Maximum position index.
    pub fn max_position(&self) -> Option<usize> {
        self.entities.keys().max().copied()
    }
}

/// HypE model: Hypergraph embeddings via position-specific convolution.
///
/// # Components
///
/// - Entity embeddings: `E[entity] -> R^dim`
/// - Position-specific filters: `F[relation][position] -> R^dim`
///
/// The score for a hyperedge is computed by summing position-weighted
/// contributions from each entity.
pub struct HypE {
    /// Embedding dimension.
    dim: usize,
    /// Maximum number of positions (arity).
    max_positions: usize,
    /// Entity embeddings.
    entity_embeddings: HashMap<String, Vec<f32>>,
    /// Position filters per relation: relation -> [position -> filter]
    relation_filters: HashMap<String, Vec<Vec<f32>>>,
    /// Whether trained.
    trained: bool,
}

impl std::fmt::Debug for HypE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HypE")
            .field("dim", &self.dim)
            .field("max_positions", &self.max_positions)
            .field("num_entities", &self.entity_embeddings.len())
            .field("num_relations", &self.relation_filters.len())
            .field("trained", &self.trained)
            .finish()
    }
}

impl Clone for HypE {
    fn clone(&self) -> Self {
        Self {
            dim: self.dim,
            max_positions: self.max_positions,
            entity_embeddings: self.entity_embeddings.clone(),
            relation_filters: self.relation_filters.clone(),
            trained: self.trained,
        }
    }
}

impl HypE {
    /// Create a new HypE model.
    ///
    /// # Arguments
    /// * `dim` - Embedding dimension
    /// * `max_positions` - Maximum number of entity positions per hyperedge
    pub fn new(dim: usize, max_positions: usize) -> Self {
        Self {
            dim,
            max_positions,
            entity_embeddings: HashMap::new(),
            relation_filters: HashMap::new(),
            trained: false,
        }
    }

    /// Score a hyperfact.
    ///
    /// ```text
    /// score = sum_{i in positions} dot(entity_i, filter_r^i)
    /// ```
    pub fn score_hyperfact(&self, fact: &HyperFact) -> Result<f32> {
        let filters = self
            .relation_filters
            .get(&fact.relation)
            .ok_or_else(|| Error::NotFound(format!("Unknown relation: {}", fact.relation)))?;

        let mut score = 0.0f32;

        for (&position, entity) in &fact.entities {
            if position >= self.max_positions {
                continue; // Skip positions beyond max
            }

            let entity_emb = self
                .entity_embeddings
                .get(entity)
                .ok_or_else(|| Error::NotFound(format!("Unknown entity: {}", entity)))?;

            let filter = &filters[position];

            // Dot product (simplified convolution)
            for i in 0..self.dim {
                score += entity_emb[i] * filter[i];
            }
        }

        Ok(score)
    }

    /// Train on hyperedges.
    pub fn train_hyperedges(
        &mut self,
        hyperedges: &[HyperFact],
        config: &TrainingConfig,
    ) -> Result<f32> {
        use std::hash::{Hash, Hasher};

        if hyperedges.is_empty() {
            return Err(Error::Validation("No hyperedges provided".into()));
        }

        self.dim = config.embedding_dim;

        // Extract vocabulary
        let mut entities: HashSet<String> = HashSet::new();
        let mut relations: HashSet<String> = HashSet::new();

        for he in hyperedges {
            relations.insert(he.relation.clone());
            for entity in he.entities.values() {
                entities.insert(entity.clone());
            }
        }

        // Initialize entity embeddings
        let init_scale = 0.1f32;
        for (i, entity) in entities.iter().enumerate() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            config.seed.hash(&mut hasher);
            i.hash(&mut hasher);

            let embedding: Vec<f32> = (0..self.dim)
                .map(|j| {
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    hasher.finish().hash(&mut h);
                    j.hash(&mut h);
                    (h.finish() as f32 / u64::MAX as f32 - 0.5) * init_scale
                })
                .collect();

            self.entity_embeddings.insert(entity.clone(), embedding);
        }

        // Initialize position filters
        for (i, relation) in relations.iter().enumerate() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (config.seed + 1).hash(&mut hasher);
            i.hash(&mut hasher);

            let filters: Vec<Vec<f32>> = (0..self.max_positions)
                .map(|pos| {
                    (0..self.dim)
                        .map(|j| {
                            let mut h = std::collections::hash_map::DefaultHasher::new();
                            hasher.finish().hash(&mut h);
                            pos.hash(&mut h);
                            j.hash(&mut h);
                            (h.finish() as f32 / u64::MAX as f32 - 0.5) * init_scale
                        })
                        .collect()
                })
                .collect();

            self.relation_filters.insert(relation.clone(), filters);
        }

        let entities_vec: Vec<&String> = entities.iter().collect();
        let mut final_loss = 0.0f32;

        // Training loop
        for epoch in 0..config.epochs {
            let mut epoch_loss = 0.0f32;
            let mut num_updates = 0;

            for (batch_idx, hyperedge) in hyperedges.iter().enumerate() {
                let pos_score = self.score_hyperfact(hyperedge).unwrap_or(0.0);

                // Negative sampling: corrupt random position
                for ns in 0..config.negative_samples {
                    // Choose position to corrupt
                    let positions: Vec<usize> = hyperedge.entities.keys().copied().collect();
                    if positions.is_empty() {
                        continue;
                    }

                    let pos_idx = (epoch + batch_idx + ns) % positions.len();
                    let corrupt_pos = positions[pos_idx];

                    // Choose random entity
                    let neg_idx = (epoch * 1000 + batch_idx * 100 + ns) % entities_vec.len();
                    let neg_entity = entities_vec[neg_idx];

                    // Create corrupted hyperedge
                    let mut corrupted = hyperedge.clone();
                    if corrupted.entities.get(&corrupt_pos) == Some(neg_entity) {
                        continue;
                    }
                    corrupted.entities.insert(corrupt_pos, neg_entity.clone());

                    let neg_score = self.score_hyperfact(&corrupted).unwrap_or(0.0);

                    // Margin loss
                    let loss = (config.margin - pos_score + neg_score).max(0.0);
                    epoch_loss += loss;

                    if loss > 0.0 {
                        let lr = config.learning_rate;

                        // Update entity embeddings and filters
                        for (&position, entity) in &hyperedge.entities {
                            if position >= self.max_positions {
                                continue;
                            }

                            // Get filter gradient direction
                            let filter = &self.relation_filters[&hyperedge.relation][position];
                            let entity_emb = self.entity_embeddings.get_mut(entity).unwrap();

                            for i in 0..self.dim {
                                entity_emb[i] += lr * filter[i] * 0.5;
                            }

                            // Update filter
                            let entity_emb = &self.entity_embeddings[entity];
                            let filters = self.relation_filters.get_mut(&hyperedge.relation).unwrap();
                            for i in 0..self.dim {
                                filters[position][i] += lr * entity_emb[i] * 0.5;
                            }
                        }

                        num_updates += 1;
                    }
                }
            }

            let avg_loss = if num_updates > 0 {
                epoch_loss / num_updates as f32
            } else {
                0.0
            };

            final_loss = avg_loss;
        }

        self.trained = true;
        Ok(final_loss)
    }
}

// Implement KGEModel for HypE (treating triples as 2-ary hyperedges)
impl KGEModel for HypE {
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let fact = HyperFact::from_triple(head, relation, tail);
        self.score_hyperfact(&fact)
    }

    fn train(&mut self, triples: &[Fact<String>], config: &TrainingConfig) -> Result<f32> {
        // Convert triples to hyperedges
        let hyperedges: Vec<HyperFact> = triples
            .iter()
            .map(|t| HyperFact::from_triple(&t.head, &t.relation, &t.tail))
            .collect();

        self.train_hyperedges(&hyperedges, config)
    }

    fn train_with_callback(
        &mut self,
        triples: &[Fact<String>],
        config: &TrainingConfig,
        callback: ProgressCallback,
    ) -> Result<f32> {
        // Simple implementation without per-epoch callback
        let result = self.train(triples, config)?;
        let metrics = EpochMetrics {
            loss: result,
            val_mrr: None,
            val_hits_at_10: None,
        };
        callback(config.epochs.saturating_sub(1), &metrics);
        Ok(result)
    }

    fn entity_embedding(&self, entity: &str) -> Option<Vec<f32>> {
        self.entity_embeddings.get(entity).cloned()
    }

    fn relation_embedding(&self, relation: &str) -> Option<Vec<f32>> {
        // Return flattened filters as "embedding"
        self.relation_filters.get(relation).map(|filters| {
            filters.iter().flat_map(|f| f.iter().copied()).collect()
        })
    }

    fn entity_embeddings(&self) -> &HashMap<String, Vec<f32>> {
        &self.entity_embeddings
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
        self.relation_filters.len()
    }

    fn name(&self) -> &'static str {
        "HypE"
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperfact_creation() {
        let fact = HyperFact::new("purchase")
            .with_entity(0, "Alice")
            .with_entity(1, "Amazon")
            .with_entity(2, "Book");

        assert_eq!(fact.relation, "purchase");
        assert_eq!(fact.arity(), 3);
        assert_eq!(fact.max_position(), Some(2));
    }

    #[test]
    fn test_hyperfact_from_triple() {
        let fact = HyperFact::from_triple("Einstein", "won", "NobelPrize");

        assert_eq!(fact.relation, "won");
        assert_eq!(fact.arity(), 2);
        assert_eq!(fact.entities.get(&0), Some(&"Einstein".to_string()));
        assert_eq!(fact.entities.get(&1), Some(&"NobelPrize".to_string()));
    }

    #[test]
    fn test_hype_creation() {
        let model = HypE::new(32, 5);
        assert_eq!(model.embedding_dim(), 32);
        assert!(!model.is_trained());
        assert_eq!(model.name(), "HypE");
    }

    #[test]
    fn test_hype_training_hyperedges() {
        let mut model = HypE::new(16, 5);

        let facts = vec![
            HyperFact::new("purchase")
                .with_entity(0, "Alice")
                .with_entity(1, "Amazon")
                .with_entity(2, "Book"),
            HyperFact::new("purchase")
                .with_entity(0, "Bob")
                .with_entity(1, "eBay")
                .with_entity(2, "Phone"),
        ];

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(10);

        let loss = model.train_hyperedges(&facts, &config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.num_entities(), 6); // Alice, Amazon, Book, Bob, eBay, Phone
        assert_eq!(model.num_relations(), 1); // purchase
        assert!(loss.is_finite());
    }

    #[test]
    fn test_hype_training_triples() {
        let mut model = HypE::new(16, 3);

        let triples = vec![
            Fact::from_strs("Einstein", "won", "NobelPrize"),
            Fact::from_strs("Curie", "won", "NobelPrize"),
        ];

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(10);

        let loss = model.train(&triples, &config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.num_entities(), 3);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_hype_scoring() {
        let mut model = HypE::new(16, 3);

        let triples = vec![
            Fact::from_strs("Einstein", "won", "NobelPrize"),
            Fact::from_strs("Curie", "won", "NobelPrize"),
        ];

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(20);

        model.train(&triples, &config).unwrap();

        let score = model.score("Einstein", "won", "NobelPrize").unwrap();
        assert!(score.is_finite());
    }
}
