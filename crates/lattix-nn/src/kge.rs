//! Knowledge Graph Embedding scoring functions.
//!
//! Provides scoring functions for KG link prediction:
//! - TransE: Translation-based (Bordes et al., 2013)
//! - RotatE: Rotation-based (Sun et al., 2019)
//! - ComplEx: Complex embeddings (Trouillon et al., 2016)
//! - DistMult: Bilinear diagonal (Yang et al., 2015)
//!
//! # Usage
//!
//! These scorers work with pre-trained embeddings. Training is typically
//! done in Python (PyKEEN) and embeddings exported for inference.
//!
//! ```rust,ignore
//! use propago::kge::{TransE, KGEScorer};
//!
//! let scorer = TransE::new(embeddings, relations, margin);
//!
//! // Score a triple (head, relation, tail)
//! let score = scorer.score(head_id, rel_id, tail_id);
//!
//! // Link prediction: top-k tail entities for (head, rel, ?)
//! let predictions = scorer.predict_tail(head_id, rel_id, k);
//! ```
//!
//! # Scoring Functions
//!
//! | Model | Score Function | Properties |
//! |-------|---------------|------------|
//! | TransE | -||h + r - t|| | Simple, additive |
//! | RotatE | -||h * r - t|| | Rotation in complex space |
//! | ComplEx | Re(<h, r, conj(t)>) | Complex, handles symmetry |
//! | DistMult | <h, r, t> | Symmetric relations only |

use candle_core::{DType, Device, Result, Tensor};

/// Trait for KGE scoring functions.
pub trait KGEScorer {
    /// Score a single triple.
    ///
    /// Higher scores indicate more plausible triples.
    fn score(&self, head: usize, relation: usize, tail: usize) -> Result<f32>;

    /// Score multiple triples.
    fn score_batch(&self, triples: &[(usize, usize, usize)]) -> Result<Vec<f32>> {
        triples
            .iter()
            .map(|(h, r, t)| self.score(*h, *r, *t))
            .collect()
    }

    /// Predict top-k tail entities for (head, relation, ?).
    fn predict_tail(&self, head: usize, relation: usize, k: usize) -> Result<Vec<(usize, f32)>>;

    /// Predict top-k head entities for (?, relation, tail).
    fn predict_head(&self, relation: usize, tail: usize, k: usize) -> Result<Vec<(usize, f32)>>;

    /// Number of entities.
    fn num_entities(&self) -> usize;

    /// Number of relations.
    fn num_relations(&self) -> usize;

    /// Embedding dimension.
    fn embedding_dim(&self) -> usize;
}

/// TransE: Translation-based embeddings.
///
/// Score: -||h + r - t||
///
/// Interprets relations as translations: if (h, r, t) holds,
/// then h + r ≈ t in embedding space.
pub struct TransE {
    /// Entity embeddings (num_entities, dim)
    entities: Tensor,
    /// Relation embeddings (num_relations, dim)
    relations: Tensor,
    /// Margin for ranking loss
    margin: f32,
    /// L1 or L2 norm
    norm: usize,
    device: Device,
}

impl TransE {
    /// Create TransE scorer from embeddings.
    ///
    /// # Arguments
    /// - `entities`: Entity embeddings (num_entities x dim)
    /// - `relations`: Relation embeddings (num_relations x dim)
    /// - `margin`: Margin for ranking (typically 1.0)
    /// - `norm`: 1 for L1, 2 for L2
    pub fn new(entities: Tensor, relations: Tensor, margin: f32, norm: usize) -> Result<Self> {
        let device = entities.device().clone();
        Ok(Self {
            entities,
            relations,
            margin,
            norm,
            device,
        })
    }

    /// Create from flat vectors.
    pub fn from_vecs(
        entities: &[f32],
        relations: &[f32],
        num_entities: usize,
        num_relations: usize,
        dim: usize,
        margin: f32,
    ) -> Result<Self> {
        let device = Device::Cpu;
        let entities = Tensor::from_slice(entities, (num_entities, dim), &device)?;
        let relations = Tensor::from_slice(relations, (num_relations, dim), &device)?;
        Self::new(entities, relations, margin, 2)
    }
}

impl KGEScorer for TransE {
    fn score(&self, head: usize, relation: usize, tail: usize) -> Result<f32> {
        let h = self.entities.get(head)?;
        let r = self.relations.get(relation)?;
        let t = self.entities.get(tail)?;

        // h + r - t
        let diff = ((h + r)? - t)?;

        // Norm
        let score = if self.norm == 1 {
            diff.abs()?.sum_all()?.to_scalar::<f32>()?
        } else {
            diff.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?
        };

        // Negative distance (higher = better)
        Ok(-score)
    }

    fn predict_tail(&self, head: usize, relation: usize, k: usize) -> Result<Vec<(usize, f32)>> {
        let h = self.entities.get(head)?;
        let r = self.relations.get(relation)?;
        let hr = (h + r)?;

        // Score all entities as tails
        // diff = hr - entities  (broadcast)
        let hr = hr.unsqueeze(0)?; // (1, dim)
        let diff = hr.broadcast_sub(&self.entities)?; // (num_entities, dim)

        let scores = if self.norm == 1 {
            diff.abs()?.sum(1)?
        } else {
            diff.sqr()?.sum(1)?.sqrt()?
        };

        // Negate and get top-k
        let neg_scores = (scores * -1.0)?;
        let (_, indices) = neg_scores.sort_last_dim(false)?;

        let indices = indices.to_vec1::<u32>()?;
        let scores = neg_scores.to_vec1::<f32>()?;

        Ok(indices
            .iter()
            .take(k)
            .map(|&i| (i as usize, scores[i as usize]))
            .collect())
    }

    fn predict_head(&self, relation: usize, tail: usize, k: usize) -> Result<Vec<(usize, f32)>> {
        let r = self.relations.get(relation)?;
        let t = self.entities.get(tail)?;
        let target = (t - r)?; // h should be approximately t - r

        let target = target.unsqueeze(0)?;
        let diff = target.broadcast_sub(&self.entities)?;

        let scores = if self.norm == 1 {
            diff.abs()?.sum(1)?
        } else {
            diff.sqr()?.sum(1)?.sqrt()?
        };

        let neg_scores = (scores * -1.0)?;
        let (_, indices) = neg_scores.sort_last_dim(false)?;

        let indices = indices.to_vec1::<u32>()?;
        let scores = neg_scores.to_vec1::<f32>()?;

        Ok(indices
            .iter()
            .take(k)
            .map(|&i| (i as usize, scores[i as usize]))
            .collect())
    }

    fn num_entities(&self) -> usize {
        self.entities.dim(0).unwrap_or(0)
    }

    fn num_relations(&self) -> usize {
        self.relations.dim(0).unwrap_or(0)
    }

    fn embedding_dim(&self) -> usize {
        self.entities.dim(1).unwrap_or(0)
    }
}

/// DistMult: Bilinear diagonal model.
///
/// Score: <h, r, t> = sum_i(h_i * r_i * t_i)
///
/// Simple but effective for symmetric relations.
pub struct DistMult {
    entities: Tensor,
    relations: Tensor,
    device: Device,
}

impl DistMult {
    /// Create DistMult scorer.
    pub fn new(entities: Tensor, relations: Tensor) -> Result<Self> {
        let device = entities.device().clone();
        Ok(Self {
            entities,
            relations,
            device,
        })
    }

    /// Create from flat vectors.
    pub fn from_vecs(
        entities: &[f32],
        relations: &[f32],
        num_entities: usize,
        num_relations: usize,
        dim: usize,
    ) -> Result<Self> {
        let device = Device::Cpu;
        let entities = Tensor::from_slice(entities, (num_entities, dim), &device)?;
        let relations = Tensor::from_slice(relations, (num_relations, dim), &device)?;
        Self::new(entities, relations)
    }
}

impl KGEScorer for DistMult {
    fn score(&self, head: usize, relation: usize, tail: usize) -> Result<f32> {
        let h = self.entities.get(head)?;
        let r = self.relations.get(relation)?;
        let t = self.entities.get(tail)?;

        // Element-wise product, then sum
        let score = ((h * r)? * t)?.sum_all()?.to_scalar::<f32>()?;
        Ok(score)
    }

    fn predict_tail(&self, head: usize, relation: usize, k: usize) -> Result<Vec<(usize, f32)>> {
        let h = self.entities.get(head)?;
        let r = self.relations.get(relation)?;
        let hr = (h * r)?;

        // Score = sum(hr * t) for each t
        // = hr . entities^T
        let hr = hr.unsqueeze(0)?;
        let scores = hr.matmul(&self.entities.t()?)?.squeeze(0)?;

        let (_, indices) = scores.sort_last_dim(false)?;
        let indices = indices.to_vec1::<u32>()?;
        let scores = scores.to_vec1::<f32>()?;

        Ok(indices
            .iter()
            .take(k)
            .map(|&i| (i as usize, scores[i as usize]))
            .collect())
    }

    fn predict_head(&self, relation: usize, tail: usize, k: usize) -> Result<Vec<(usize, f32)>> {
        let r = self.relations.get(relation)?;
        let t = self.entities.get(tail)?;
        let rt = (r * t)?;

        let rt = rt.unsqueeze(0)?;
        let scores = rt.matmul(&self.entities.t()?)?.squeeze(0)?;

        let (_, indices) = scores.sort_last_dim(false)?;
        let indices = indices.to_vec1::<u32>()?;
        let scores = scores.to_vec1::<f32>()?;

        Ok(indices
            .iter()
            .take(k)
            .map(|&i| (i as usize, scores[i as usize]))
            .collect())
    }

    fn num_entities(&self) -> usize {
        self.entities.dim(0).unwrap_or(0)
    }

    fn num_relations(&self) -> usize {
        self.relations.dim(0).unwrap_or(0)
    }

    fn embedding_dim(&self) -> usize {
        self.entities.dim(1).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_embeddings(n: usize, dim: usize) -> Vec<f32> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        (0..n * dim).map(|_| rng.random::<f32>() - 0.5).collect()
    }

    #[test]
    fn test_transe_score() {
        let entities = random_embeddings(10, 32);
        let relations = random_embeddings(5, 32);

        let scorer = TransE::from_vecs(&entities, &relations, 10, 5, 32, 1.0).unwrap();

        let score = scorer.score(0, 0, 1).unwrap();
        // Score should be negative (distance)
        assert!(score <= 0.0);
    }

    #[test]
    fn test_transe_predict_tail() {
        let entities = random_embeddings(10, 32);
        let relations = random_embeddings(5, 32);

        let scorer = TransE::from_vecs(&entities, &relations, 10, 5, 32, 1.0).unwrap();

        let predictions = scorer.predict_tail(0, 0, 3).unwrap();
        assert_eq!(predictions.len(), 3);

        // Scores should be in descending order (higher = better = less negative)
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i].1 >= predictions[i + 1].1);
        }
    }

    #[test]
    fn test_distmult_score() {
        let entities = random_embeddings(10, 32);
        let relations = random_embeddings(5, 32);

        let scorer = DistMult::from_vecs(&entities, &relations, 10, 5, 32).unwrap();

        let score = scorer.score(0, 0, 1).unwrap();
        // DistMult scores can be positive or negative
        assert!(score.is_finite());
    }

    #[test]
    fn test_distmult_predict_tail() {
        let entities = random_embeddings(10, 32);
        let relations = random_embeddings(5, 32);

        let scorer = DistMult::from_vecs(&entities, &relations, 10, 5, 32).unwrap();

        let predictions = scorer.predict_tail(0, 0, 3).unwrap();
        assert_eq!(predictions.len(), 3);
    }
}
