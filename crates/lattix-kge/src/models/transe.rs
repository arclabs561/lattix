//! TransE: Relations as Translations.
//!
//! TransE ([Bordes et al. 2013](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html))
//! is the foundational KGE model. It interprets relations as translations in embedding space:
//!
//! ```text
//! h + r ≈ t  (if the triple is true)
//! ```
//!
//! # Scoring
//!
//! Score = -||h + r - t||₂ (negative L2 distance)
//!
//! Higher scores indicate more plausible triples.
//!
//! # Training
//!
//! Uses margin-based ranking loss with negative sampling:
//!
//! ```text
//! L = max(0, margin + d(h+r, t) - d(h+r, t'))
//! ```
//!
//! where t' is a corrupted (negative) tail entity.
//!
//! # Example
//!
//! ```rust,ignore
//! use lattix_kge::models::TransE;
//! use lattix_kge::{KGEModel, Fact, TrainingConfig};
//!
//! let mut model = TransE::new(128);  // 128-dim embeddings
//!
//! let triples = vec![
//!     Fact::from_strs("Einstein", "won", "NobelPrize"),
//!     Fact::from_strs("Paris", "capitalOf", "France"),
//! ];
//!
//! model.train(&triples, &TrainingConfig::default())?;
//!
//! let score = model.score("Einstein", "won", "NobelPrize")?;
//! ```

use crate::error::{Error, Result};
use crate::model::{EpochMetrics, Fact, KGEModel, ProgressCallback};
use crate::training::TrainingConfig;
use std::collections::{HashMap, HashSet};

/// TransE model: relations as translations in embedding space.
///
/// This is the ndarray-based implementation. For GPU acceleration,
/// use `TransEBurn` (requires `burn` feature).
#[derive(Debug, Clone)]
pub struct TransE {
    /// Embedding dimension.
    dim: usize,
    /// Entity embeddings (entity_id -> embedding vector).
    entity_embeddings: HashMap<String, Vec<f32>>,
    /// Relation embeddings (relation_id -> embedding vector).
    relation_embeddings: HashMap<String, Vec<f32>>,
    /// Whether the model has been trained.
    trained: bool,
}

impl Default for TransE {
    fn default() -> Self {
        Self::new(128)
    }
}

impl TransE {
    /// Create a new TransE model with the given embedding dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            trained: false,
        }
    }

    /// Create from pre-trained embeddings.
    pub fn from_embeddings(
        entity_embeddings: HashMap<String, Vec<f32>>,
        relation_embeddings: HashMap<String, Vec<f32>>,
    ) -> Result<Self> {
        // Validate dimensions match
        let dim = entity_embeddings
            .values()
            .next()
            .map(|v| v.len())
            .unwrap_or(128);

        for (k, v) in &entity_embeddings {
            if v.len() != dim {
                return Err(Error::Validation(format!(
                    "Entity '{}' has dimension {} but expected {}",
                    k,
                    v.len(),
                    dim
                )));
            }
        }

        for (k, v) in &relation_embeddings {
            if v.len() != dim {
                return Err(Error::Validation(format!(
                    "Relation '{}' has dimension {} but expected {}",
                    k,
                    v.len(),
                    dim
                )));
            }
        }

        Ok(Self {
            dim,
            entity_embeddings,
            relation_embeddings,
            trained: true,
        })
    }

    /// TransE distance: ||h + r - t||₂
    #[inline]
    fn distance(&self, h: &[f32], r: &[f32], t: &[f32]) -> f32 {
        let mut sum = 0.0;
        for i in 0..self.dim {
            let diff = h[i] + r[i] - t[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// Initialize embeddings for a vocabulary.
    fn init_embeddings(&self, vocab: &HashSet<String>, seed: u64) -> HashMap<String, Vec<f32>> {
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
                // Map to [-0.5, 0.5]
                let f = (val as f64 / u64::MAX as f64) - 0.5;
                embedding.push(f as f32);
            }

            // Normalize to unit ball
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

impl KGEModel for TransE {
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let h = self
            .entity_embeddings
            .get(head)
            .ok_or_else(|| Error::NotFound(format!("Unknown entity: {}", head)))?;
        let r = self
            .relation_embeddings
            .get(relation)
            .ok_or_else(|| Error::NotFound(format!("Unknown relation: {}", relation)))?;
        let t = self
            .entity_embeddings
            .get(tail)
            .ok_or_else(|| Error::NotFound(format!("Unknown entity: {}", tail)))?;

        // TransE score: negative distance (higher = more plausible)
        Ok(-self.distance(h, r, t))
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

        // Initialize embeddings
        let (entities, relations) = self.extract_vocab(triples);
        self.dim = config.embedding_dim;
        self.entity_embeddings = self.init_embeddings(&entities, config.seed);
        self.relation_embeddings = self.init_embeddings(&relations, config.seed + 1);

        let entities_vec: Vec<&String> = entities.iter().collect();
        let mut final_loss = 0.0;

        for epoch in 0..config.epochs {
            let mut epoch_loss = 0.0;
            let mut num_updates = 0;

            for (batch_idx, batch) in triples.chunks(config.batch_size).enumerate() {
                for triple in batch {
                    let h = self.entity_embeddings.get(&triple.head).unwrap().clone();
                    let r = self
                        .relation_embeddings
                        .get(&triple.relation)
                        .unwrap()
                        .clone();
                    let t = self.entity_embeddings.get(&triple.tail).unwrap().clone();

                    let pos_dist = self.distance(&h, &r, &t);

                    // Negative sampling
                    for ns in 0..config.negative_samples {
                        let neg_idx = (epoch * 1000 + batch_idx * 100 + ns) % entities_vec.len();
                        let neg_tail = entities_vec[neg_idx];

                        if neg_tail == &triple.tail {
                            continue;
                        }

                        let t_neg = self.entity_embeddings.get(neg_tail).unwrap();
                        let neg_dist = self.distance(&h, &r, t_neg);

                        // Margin-based ranking loss
                        let loss = (config.margin + pos_dist - neg_dist).max(0.0);
                        epoch_loss += loss;

                        if loss > 0.0 {
                            let lr = config.learning_rate;

                            // Gradient: d/dh ||h+r-t||₂ = (h+r-t) / ||h+r-t||
                            // Simplified SGD update
                            let h_mut = self.entity_embeddings.get_mut(&triple.head).unwrap();
                            for i in 0..self.dim {
                                h_mut[i] -= lr * 2.0 * (h[i] + r[i] - t[i]);
                            }

                            let t_mut = self.entity_embeddings.get_mut(&triple.tail).unwrap();
                            for i in 0..self.dim {
                                t_mut[i] -= lr * -2.0 * (h[i] + r[i] - t[i]);
                            }

                            let r_mut = self.relation_embeddings.get_mut(&triple.relation).unwrap();
                            for i in 0..self.dim {
                                r_mut[i] -= lr * 2.0 * (h[i] + r[i] - t[i]);
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

            // Early stopping: if loss is very low, we can stop
            if avg_loss < 1e-6 && epoch > 10 {
                break;
            }
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
        "TransE"
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
            Fact::from_strs("Einstein", "won", "NobelPrize"),
            Fact::from_strs("Curie", "won", "NobelPrize"),
            Fact::from_strs("Paris", "capitalOf", "France"),
            Fact::from_strs("Berlin", "capitalOf", "Germany"),
            Fact::from_strs("Einstein", "bornIn", "Germany"),
            Fact::from_strs("Curie", "bornIn", "Poland"),
        ]
    }

    #[test]
    fn test_transe_creation() {
        let model = TransE::new(64);
        assert_eq!(model.embedding_dim(), 64);
        assert_eq!(model.num_entities(), 0);
        assert!(!model.is_trained());
        assert_eq!(model.name(), "TransE");
    }

    #[test]
    fn test_transe_training() {
        let mut model = TransE::new(32);
        let triples = sample_triples();

        let config = TrainingConfig::default()
            .with_embedding_dim(32)
            .with_epochs(10)
            .with_learning_rate(0.01);

        let loss = model.train(&triples, &config).unwrap();

        assert!(model.is_trained());
        assert_eq!(model.num_entities(), 8); // Einstein, Curie, NobelPrize, Paris, France, Berlin, Germany, Poland
        assert_eq!(model.num_relations(), 3); // won, capitalOf, bornIn
        assert!(loss.is_finite());
    }

    #[test]
    fn test_transe_scoring() {
        let mut model = TransE::new(32);
        let triples = sample_triples();

        let config = TrainingConfig::default()
            .with_embedding_dim(32)
            .with_epochs(50)
            .with_learning_rate(0.01);

        model.train(&triples, &config).unwrap();

        // Should be able to score any triple with known entities
        let score = model.score("Einstein", "won", "NobelPrize").unwrap();
        assert!(score.is_finite());

        // Unknown entity should error
        let result = model.score("Unknown", "won", "NobelPrize");
        assert!(result.is_err());
    }

    #[test]
    fn test_transe_embeddings() {
        let mut model = TransE::new(32);
        let triples = sample_triples();

        let config = TrainingConfig::default()
            .with_embedding_dim(32)
            .with_epochs(5);

        model.train(&triples, &config).unwrap();

        let emb = model.entity_embedding("Einstein").unwrap();
        assert_eq!(emb.len(), 32);

        let rel_emb = model.relation_embedding("won").unwrap();
        assert_eq!(rel_emb.len(), 32);

        assert!(model.entity_embedding("Unknown").is_none());
    }

    #[test]
    fn test_transe_prediction() {
        let mut model = TransE::new(32);
        let triples = sample_triples();

        let config = TrainingConfig::default()
            .with_embedding_dim(32)
            .with_epochs(20)
            .with_learning_rate(0.01);

        model.train(&triples, &config).unwrap();

        let predictions = model.predict_tail("Einstein", "won", 3).unwrap();
        assert!(!predictions.is_empty());
        assert!(predictions.len() <= 3);

        // Predictions should be sorted by score (descending)
        for i in 1..predictions.len() {
            assert!(predictions[i - 1].score >= predictions[i].score);
        }
    }

    #[test]
    fn test_transe_from_embeddings() {
        let mut entity_emb = HashMap::new();
        entity_emb.insert("A".to_string(), vec![1.0, 0.0, 0.0]);
        entity_emb.insert("B".to_string(), vec![0.0, 1.0, 0.0]);

        let mut rel_emb = HashMap::new();
        rel_emb.insert("r".to_string(), vec![0.0, 0.0, 1.0]);

        let model = TransE::from_embeddings(entity_emb, rel_emb).unwrap();
        assert!(model.is_trained());
        assert_eq!(model.embedding_dim(), 3);
        assert_eq!(model.num_entities(), 2);
        assert_eq!(model.num_relations(), 1);

        let score = model.score("A", "r", "B").unwrap();
        assert!(score.is_finite());
    }

    #[test]
    fn test_transe_callback() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let mut model = TransE::new(16);
        let triples = sample_triples();

        let config = TrainingConfig::default()
            .with_embedding_dim(16)
            .with_epochs(5);

        let call_count = Arc::new(AtomicUsize::new(0));
        let counter = call_count.clone();

        model
            .train_with_callback(
                &triples,
                &config,
                Box::new(move |_epoch, _metrics| {
                    counter.fetch_add(1, Ordering::SeqCst);
                }),
            )
            .unwrap();

        assert_eq!(call_count.load(Ordering::SeqCst), 5);
    }
}
