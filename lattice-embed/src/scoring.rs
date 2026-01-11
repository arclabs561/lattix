//! Scoring functions for knowledge graph embeddings.

use serde::{Deserialize, Serialize};

/// Result of link prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkPredictionResult {
    /// Entity ID or label.
    pub entity: String,
    /// Score (higher = more plausible).
    pub score: f32,
    /// Rank (1 = best).
    pub rank: usize,
}

/// Score for a triple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleScore {
    /// Head entity.
    pub head: String,
    /// Relation.
    pub relation: String,
    /// Tail entity.
    pub tail: String,
    /// Plausibility score.
    pub score: f32,
}

/// Common scoring functions for KGE models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringFunction {
    /// TransE: ||h + r - t||
    TransE,
    /// RotatE: ||h ∘ r - t||
    RotatE,
    /// ComplEx: Re(<h, r, conj(t)>)
    ComplEx,
    /// DistMult: <h, r, t>
    DistMult,
}

impl ScoringFunction {
    /// Compute score for embeddings.
    ///
    /// All embeddings should have the same dimension.
    pub fn score(&self, head: &[f32], relation: &[f32], tail: &[f32]) -> f32 {
        match self {
            Self::TransE => score_transe(head, relation, tail),
            Self::DistMult => score_distmult(head, relation, tail),
            // RotatE and ComplEx need complex numbers, simplified here
            Self::RotatE => score_transe(head, relation, tail), // Simplified
            Self::ComplEx => score_distmult(head, relation, tail), // Simplified
        }
    }
}

/// TransE scoring: -||h + r - t||_2
fn score_transe(head: &[f32], relation: &[f32], tail: &[f32]) -> f32 {
    let mut sum_sq = 0.0;
    for i in 0..head.len() {
        let diff = head[i] + relation[i] - tail[i];
        sum_sq += diff * diff;
    }
    -sum_sq.sqrt()
}

/// DistMult scoring: <h, r, t>
fn score_distmult(head: &[f32], relation: &[f32], tail: &[f32]) -> f32 {
    let mut score = 0.0;
    for i in 0..head.len() {
        score += head[i] * relation[i] * tail[i];
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transe_scoring() {
        // h + r = t should have score close to 0 (best)
        let h = vec![1.0, 0.0, 0.0];
        let r = vec![0.0, 1.0, 0.0];
        let t = vec![1.0, 1.0, 0.0];

        let score = ScoringFunction::TransE.score(&h, &r, &t);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_distmult_scoring() {
        let h = vec![1.0, 0.5, 0.0];
        let r = vec![1.0, 1.0, 1.0];
        let t = vec![0.5, 1.0, 0.0];

        let score = ScoringFunction::DistMult.score(&h, &r, &t);
        assert!((score - 1.0).abs() < 1e-6); // 1*1*0.5 + 0.5*1*1 + 0*1*0 = 1.0
    }
}
