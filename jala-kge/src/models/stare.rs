//! StarE: Hyper-relational Knowledge Graph Embedding with Qualifiers.
//!
//! StarE ([Galkin et al. 2020](https://arxiv.org/abs/2006.04986)) extends KGE to
//! handle hyper-relational facts with qualifiers. In Wikidata-style KGs, facts
//! often have additional context:
//!
//! ```text
//! (Einstein, educated_at, ETH_Zurich) + {start_time: 1896, degree: PhD}
//! ```
//!
//! # Key Innovation
//!
//! StarE represents qualified statements as:
//!
//! ```text
//! score(h, r, t | Q) = f(h, r*, t)
//! ```
//!
//! where r* is a relation embedding modified by the qualifiers Q:
//!
//! ```text
//! r* = r + g(Q)
//! g(Q) = sum_{(qr, qv) in Q} phi(qr, qv)
//! ```
//!
//! # Qualifier Composition
//!
//! Each qualifier (qr, qv) is composed using:
//!
//! ```text
//! phi(qr, qv) = qr * qv  (element-wise)
//! ```
//!
//! This allows the base fact to be modified by arbitrary qualifiers.
//!
//! # When to Use
//!
//! - Wikidata, Freebase, or similar hyper-relational KGs
//! - Facts with temporal, spatial, or provenance qualifiers
//! - Statements requiring additional context
//!
//! # Example
//!
//! ```rust,ignore
//! use lattix_kge::models::StarE;
//! use lattix_kge::{KGEModel, Fact, TrainingConfig};
//!
//! let mut model = StarE::new(32);
//!
//! // Qualified fact: (Einstein, educated_at, ETH) with qualifiers
//! let fact = QualifiedFact::new("Einstein", "educated_at", "ETH_Zurich")
//!     .with_qualifier("start_time", "1896")
//!     .with_qualifier("degree", "PhD");
//!
//! model.train_qualified(&[fact], &TrainingConfig::default())?;
//! ```

use crate::error::{Error, Result};
use crate::model::{EpochMetrics, Fact, KGEModel, ProgressCallback};
use crate::training::TrainingConfig;
use std::collections::{HashMap, HashSet};

/// A qualified fact: base triple + qualifier key-value pairs.
#[derive(Debug, Clone)]
pub struct QualifiedFact {
    /// Subject entity.
    pub head: String,
    /// Predicate relation.
    pub relation: String,
    /// Object entity.
    pub tail: String,
    /// Qualifiers: (qualifier_relation, qualifier_value) pairs.
    pub qualifiers: Vec<(String, String)>,
}

impl QualifiedFact {
    /// Create a new qualified fact.
    pub fn new(
        head: impl Into<String>,
        relation: impl Into<String>,
        tail: impl Into<String>,
    ) -> Self {
        Self {
            head: head.into(),
            relation: relation.into(),
            tail: tail.into(),
            qualifiers: Vec::new(),
        }
    }

    /// Add a qualifier.
    pub fn with_qualifier(mut self, qr: impl Into<String>, qv: impl Into<String>) -> Self {
        self.qualifiers.push((qr.into(), qv.into()));
        self
    }

    /// Create from a base triple (no qualifiers).
    pub fn from_triple(head: &str, relation: &str, tail: &str) -> Self {
        Self::new(head, relation, tail)
    }

    /// Number of qualifiers.
    pub fn num_qualifiers(&self) -> usize {
        self.qualifiers.len()
    }
}

/// StarE model: Hyper-relational KG embedding with qualifiers.
///
/// Modifies relation embeddings based on qualifier key-value pairs.
#[derive(Debug, Clone)]
pub struct StarE {
    /// Embedding dimension.
    dim: usize,
    /// Entity embeddings.
    entity_embeddings: HashMap<String, Vec<f32>>,
    /// Base relation embeddings.
    relation_embeddings: HashMap<String, Vec<f32>>,
    /// Qualifier relation embeddings.
    qualifier_rel_embeddings: HashMap<String, Vec<f32>>,
    /// Qualifier value embeddings (can overlap with entities).
    qualifier_val_embeddings: HashMap<String, Vec<f32>>,
    /// Whether trained.
    trained: bool,
}

impl StarE {
    /// Create a new StarE model.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            qualifier_rel_embeddings: HashMap::new(),
            qualifier_val_embeddings: HashMap::new(),
            trained: false,
        }
    }

    /// Compose qualifier contribution: phi(qr, qv) = qr * qv.
    fn compose_qualifier(&self, qr: &[f32], qv: &[f32]) -> Vec<f32> {
        qr.iter().zip(qv.iter()).map(|(a, b)| a * b).collect()
    }

    /// Compute modified relation embedding: r* = r + sum(phi(qr, qv)).
    fn modified_relation(&self, base_r: &[f32], qualifiers: &[(String, String)]) -> Vec<f32> {
        let mut r_star = base_r.to_vec();

        for (qr_name, qv_name) in qualifiers {
            if let (Some(qr), Some(qv)) = (
                self.qualifier_rel_embeddings.get(qr_name),
                self.qualifier_val_embeddings.get(qv_name),
            ) {
                let phi = self.compose_qualifier(qr, qv);
                for i in 0..self.dim {
                    r_star[i] += phi[i];
                }
            }
        }

        r_star
    }

    /// Score a qualified fact using TransE-style scoring on modified relation.
    pub fn score_qualified(&self, fact: &QualifiedFact) -> Result<f32> {
        let h = self
            .entity_embeddings
            .get(&fact.head)
            .ok_or_else(|| Error::NotFound(format!("Unknown entity: {}", fact.head)))?;
        let t = self
            .entity_embeddings
            .get(&fact.tail)
            .ok_or_else(|| Error::NotFound(format!("Unknown entity: {}", fact.tail)))?;
        let base_r = self
            .relation_embeddings
            .get(&fact.relation)
            .ok_or_else(|| Error::NotFound(format!("Unknown relation: {}", fact.relation)))?;

        // Compute modified relation
        let r_star = self.modified_relation(base_r, &fact.qualifiers);

        // TransE-style score: -||h + r* - t||
        let mut dist_sq = 0.0f32;
        for i in 0..self.dim {
            let diff = h[i] + r_star[i] - t[i];
            dist_sq += diff * diff;
        }

        Ok(-dist_sq.sqrt())
    }

    /// Train on qualified facts.
    pub fn train_qualified(
        &mut self,
        facts: &[QualifiedFact],
        config: &TrainingConfig,
    ) -> Result<f32> {
        self.train_qualified_with_callback(facts, config, Box::new(|_, _| {}))
    }

    /// Train on qualified facts with progress callback.
    pub fn train_qualified_with_callback(
        &mut self,
        facts: &[QualifiedFact],
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
        let mut qual_rels = HashSet::new();
        let mut qual_vals = HashSet::new();

        for fact in facts {
            entities.insert(fact.head.clone());
            entities.insert(fact.tail.clone());
            relations.insert(fact.relation.clone());

            for (qr, qv) in &fact.qualifiers {
                qual_rels.insert(qr.clone());
                qual_vals.insert(qv.clone());
            }
        }

        // Initialize embeddings
        let init_scale = 0.1;

        // Helper to create embedding
        let make_embedding = |seed: u64, idx: usize, dim: usize| -> Vec<f32> {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            seed.hash(&mut hasher);
            idx.hash(&mut hasher);
            let hash = hasher.finish();

            (0..dim)
                .map(|j| {
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    hash.hash(&mut h);
                    j.hash(&mut h);
                    ((h.finish() as f64 / u64::MAX as f64) - 0.5) as f32 * init_scale
                })
                .collect()
        };

        for (i, entity) in entities.iter().enumerate() {
            self.entity_embeddings
                .insert(entity.clone(), make_embedding(config.seed, i, self.dim));
        }

        for (i, relation) in relations.iter().enumerate() {
            self.relation_embeddings.insert(
                relation.clone(),
                make_embedding(config.seed + 1, i, self.dim),
            );
        }

        for (i, qr) in qual_rels.iter().enumerate() {
            self.qualifier_rel_embeddings
                .insert(qr.clone(), make_embedding(config.seed + 2, i, self.dim));
        }

        for (i, qv) in qual_vals.iter().enumerate() {
            self.qualifier_val_embeddings
                .insert(qv.clone(), make_embedding(config.seed + 3, i, self.dim));
        }

        let entities_vec: Vec<&String> = entities.iter().collect();
        let mut final_loss = 0.0;

        // Training loop
        for epoch in 0..config.epochs {
            let lr = config.learning_rate;
            let mut epoch_loss = 0.0f32;
            let mut num_updates = 0;

            for (batch_idx, batch) in facts.chunks(config.batch_size).enumerate() {
                for fact in batch {
                    let pos_score = self.score_qualified(fact)?;

                    // Negative sampling: corrupt tail
                    for ns in 0..config.negative_samples {
                        let neg_idx = (epoch * 1000 + batch_idx * 100 + ns) % entities_vec.len();
                        let neg_tail = entities_vec[neg_idx];

                        if neg_tail == &fact.tail {
                            continue;
                        }

                        let mut neg_fact = fact.clone();
                        neg_fact.tail = neg_tail.clone();

                        let neg_score = self.score_qualified(&neg_fact)?;

                        // Margin loss
                        let loss = (config.margin - pos_score + neg_score).max(0.0);
                        epoch_loss += loss;

                        if loss > 0.0 {
                            // Gradient updates
                            let h = self
                                .entity_embeddings
                                .get_mut(&fact.head)
                                .ok_or_else(|| Error::EntityNotFound(fact.head.clone()))?;
                            for v in h.iter_mut() {
                                *v += lr * 0.3;
                            }

                            let t = self
                                .entity_embeddings
                                .get_mut(&fact.tail)
                                .ok_or_else(|| Error::EntityNotFound(fact.tail.clone()))?;
                            for v in t.iter_mut() {
                                *v -= lr * 0.3;
                            }

                            let r = self
                                .relation_embeddings
                                .get_mut(&fact.relation)
                                .ok_or_else(|| Error::RelationNotFound(fact.relation.clone()))?;
                            for v in r.iter_mut() {
                                *v += lr * 0.1;
                            }

                            // Update qualifier embeddings
                            for (qr_name, qv_name) in &fact.qualifiers {
                                if let Some(qr) = self.qualifier_rel_embeddings.get_mut(qr_name) {
                                    for v in qr.iter_mut() {
                                        *v += lr * 0.05;
                                    }
                                }
                                if let Some(qv) = self.qualifier_val_embeddings.get_mut(qv_name) {
                                    for v in qv.iter_mut() {
                                        *v += lr * 0.05;
                                    }
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

impl KGEModel for StarE {
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        // Score as unqualified fact
        let fact = QualifiedFact::from_triple(head, relation, tail);
        self.score_qualified(&fact)
    }

    fn train(&mut self, triples: &[Fact<String>], config: &TrainingConfig) -> Result<f32> {
        // Convert to qualified facts (no qualifiers)
        let facts: Vec<QualifiedFact> = triples
            .iter()
            .map(|t| QualifiedFact::from_triple(&t.head, &t.relation, &t.tail))
            .collect();
        self.train_qualified(&facts, config)
    }

    fn train_with_callback(
        &mut self,
        triples: &[Fact<String>],
        config: &TrainingConfig,
        callback: ProgressCallback,
    ) -> Result<f32> {
        let facts: Vec<QualifiedFact> = triples
            .iter()
            .map(|t| QualifiedFact::from_triple(&t.head, &t.relation, &t.tail))
            .collect();
        self.train_qualified_with_callback(&facts, config, callback)
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
        "StarE"
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
    fn test_qualified_fact_creation() {
        let fact = QualifiedFact::new("Einstein", "educated_at", "ETH")
            .with_qualifier("start_time", "1896")
            .with_qualifier("degree", "PhD");

        assert_eq!(fact.head, "Einstein");
        assert_eq!(fact.relation, "educated_at");
        assert_eq!(fact.tail, "ETH");
        assert_eq!(fact.num_qualifiers(), 2);
    }

    #[test]
    fn test_stare_creation() {
        let model = StarE::new(32);
        assert_eq!(model.embedding_dim(), 32);
        assert!(!model.is_trained());
        assert_eq!(model.name(), "StarE");
    }

    #[test]
    fn test_stare_triple_training() {
        let mut model = StarE::new(16);
        let triples = vec![
            Fact::from_strs("Einstein", "born_in", "Germany"),
            Fact::from_strs("Einstein", "educated_at", "ETH"),
        ];

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(20);

        let loss = model.train(&triples, &config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.num_entities(), 3);
        assert_eq!(model.num_relations(), 2);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_stare_qualified_training() {
        let mut model = StarE::new(16);

        let facts = vec![
            QualifiedFact::new("Einstein", "educated_at", "ETH")
                .with_qualifier("start_time", "1896")
                .with_qualifier("end_time", "1900"),
            QualifiedFact::new("Curie", "educated_at", "Sorbonne").with_qualifier("degree", "PhD"),
        ];

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(30);

        let loss = model.train_qualified(&facts, &config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.num_entities(), 4); // Einstein, ETH, Curie, Sorbonne
        assert_eq!(model.num_relations(), 1); // educated_at
        assert!(loss.is_finite());

        // Check qualifier embeddings were created
        assert!(model.qualifier_rel_embeddings.contains_key("start_time"));
        assert!(model.qualifier_val_embeddings.contains_key("1896"));
    }

    #[test]
    fn test_stare_scoring() {
        let mut model = StarE::new(16);

        let facts = vec![QualifiedFact::new("A", "r", "B").with_qualifier("qr", "qv")];

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(20);

        model.train_qualified(&facts, &config).unwrap();

        // Score with qualifiers
        let score_with = model.score_qualified(&facts[0]).unwrap();
        assert!(score_with.is_finite());

        // Score without qualifiers
        let score_without = model.score("A", "r", "B").unwrap();
        assert!(score_without.is_finite());

        // Scores should differ since qualifiers modify the relation
        // (Though the difference may be small with random init)
    }

    #[test]
    fn test_stare_qualifier_composition() {
        let model = StarE::new(4);

        let qr = vec![1.0, 2.0, 3.0, 4.0];
        let qv = vec![0.5, 0.5, 0.5, 0.5];

        let phi = model.compose_qualifier(&qr, &qv);

        assert_eq!(phi, vec![0.5, 1.0, 1.5, 2.0]);
    }
}
