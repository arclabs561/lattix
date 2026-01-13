//! Scoring functions for knowledge graph embeddings.
//!
//! # Geometric Approaches
//!
//! KGE models differ in how they interpret relations geometrically:
//!
//! | Geometry | Models | Strengths |
//! |----------|--------|-----------|
//! | Translation | TransE | Simple, fast, good on hierarchies |
//! | Rotation | RotatE | Composition, symmetry patterns |
//! | Bilinear | DistMult, ComplEx | Asymmetric relations |
//! | Box | BoxE | Containment, hierarchy, all patterns |
//!
//! # Historical Note
//!
//! TransE (Bordes et al. 2013) started the translation distance paradigm.
//! RotatE (Sun et al. 2019) added rotation for composition.
//! BoxE (Abboud et al. 2020) unified all patterns via box containment.
//!
//! Enable `boxe` feature for BoxE support via `subsume` crate.

use lattix_core::{EntityId, RelationType};
use serde::{Deserialize, Serialize};

/// Result of link prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkPredictionResult {
    /// Predicted entity.
    pub entity: EntityId,
    /// Score (higher = more plausible).
    pub score: f32,
    /// Rank (1 = best).
    pub rank: usize,
}

/// Score for a triple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleScore {
    /// Head entity (subject).
    pub head: EntityId,
    /// Relation (predicate).
    pub relation: RelationType,
    /// Tail entity (object).
    pub tail: EntityId,
    /// Plausibility score.
    pub score: f32,
}

/// Common scoring functions for KGE models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringFunction {
    /// TransE: -||h + r - t||
    ///
    /// The original translation model (Bordes et al. 2013). Relations are
    /// translations: if (h, r, t) is true, then h + r ≈ t.
    ///
    /// **Strengths**: Simple, fast, works well on hierarchies.
    /// **Weakness**: Can't model symmetric or N-to-N relations.
    TransE,

    /// RotatE: -||h ∘ r - t|| (Complex space)
    ///
    /// Models relations as rotations (Sun et al. 2019). Each relation
    /// has an angle θ, and h is rotated by θ to match t.
    ///
    /// **Strengths**: Composition (angles add), symmetry (180° rotation).
    RotatE,

    /// ComplEx: Re(<h, r, conj(t)>)
    ///
    /// Complex embeddings for asymmetric relations (Trouillon et al. 2016).
    /// The conjugate breaks symmetry.
    ComplEx,

    /// DistMult: <h, r, t>
    ///
    /// Diagonal bilinear model (Yang et al. 2015).
    ///
    /// **Weakness**: Inherently symmetric - predicts (h,r,t) = (t,r,h).
    DistMult,
}

impl ScoringFunction {
    /// Compute score for embeddings.
    ///
    /// For complex models (RotatE, ComplEx), the input vectors are treated as
    /// concatenations of real and imaginary parts (size 2d).
    pub fn score(&self, head: &[f32], relation: &[f32], tail: &[f32]) -> f32 {
        match self {
            Self::TransE => score_transe(head, relation, tail),
            Self::DistMult => score_distmult(head, relation, tail),
            Self::RotatE => score_rotate(head, relation, tail),
            Self::ComplEx => score_complex(head, relation, tail),
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

/// RotatE scoring: -||h ∘ r - t|| in complex space.
/// Embedding size must be even (real, imag interleaved).
fn score_rotate(head: &[f32], relation: &[f32], tail: &[f32]) -> f32 {
    let dim = head.len() / 2;
    let mut sum_sq = 0.0;

    for i in 0..dim {
        let h_re = head[2 * i];
        let h_im = head[2 * i + 1];
        let r_re = relation[2 * i];
        let r_im = relation[2 * i + 1];
        let t_re = tail[2 * i];
        let t_im = tail[2 * i + 1];

        // h ∘ r (element-wise rotation)
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        let rot_re = h_re * r_re - h_im * r_im;
        let rot_im = h_re * r_im + h_im * r_re;

        let diff_re = rot_re - t_re;
        let diff_im = rot_im - t_im;

        sum_sq += diff_re * diff_re + diff_im * diff_im;
    }
    -sum_sq.sqrt()
}

/// ComplEx scoring: Re(<h, r, conj(t)>).
/// Embedding size must be even (real, imag interleaved).
fn score_complex(head: &[f32], relation: &[f32], tail: &[f32]) -> f32 {
    let dim = head.len() / 2;
    let mut score = 0.0;

    for i in 0..dim {
        let h_re = head[2 * i];
        let h_im = head[2 * i + 1];
        let r_re = relation[2 * i];
        let r_im = relation[2 * i + 1];
        let t_re = tail[2 * i];
        let t_im = tail[2 * i + 1];

        // <h, r, conj(t)> = h * r * conj(t)
        // (a+bi)(c+di)(e-fi)
        // Let (a+bi)(c+di) = (ac-bd) + (ad+bc)i = X + Yi
        // (X+Yi)(e-fi) = (Xe+Yf) + (Ye-Xf)i
        // Real part = Xe + Yf

        let x = h_re * r_re - h_im * r_im;
        let y = h_re * r_im + h_im * r_re;

        score += x * t_re + y * t_im;
    }
    score
}

// ============================================================================
// BoxE Scoring (with `boxe` feature)
// ============================================================================

/// BoxE scoring: box containment after relation translation.
///
/// Entities are boxes (min/max bounds), relations are translations.
/// Score = containment probability of tail in translated head.
///
/// # Mathematical Formulation
///
/// For triple (h, r, t):
/// 1. h' = h + bump_r (translate head by relation)
/// 2. score = P(t ⊆ h') = Vol(h' ∩ t) / Vol(t)
///
/// # Why BoxE?
///
/// Unlike point embeddings (TransE, RotatE), boxes naturally model:
/// - **Hierarchy**: Container boxes subsume contained ones
/// - **Symmetry**: Overlapping boxes of equal size
/// - **Composition**: Nested translations
/// - **Multiple patterns simultaneously**
///
/// # Reference
///
/// Abboud et al. (2020). "BoxE: A Box Embedding Model for Knowledge Base
/// Completion." NeurIPS 2020. <https://arxiv.org/abs/2007.06267>
#[cfg(feature = "boxe")]
pub fn boxe_score(
    head_min: &[f32],
    head_max: &[f32],
    tail_min: &[f32],
    tail_max: &[f32],
    relation_bump: &[f32],
    temperature: f32,
) -> crate::Result<f32> {
    use subsume_core::boxe;

    boxe::boxe_score(head_min, head_max, tail_min, tail_max, relation_bump, temperature)
        .map_err(|e| crate::Error::BoxEmbedding(format!("{:?}", e)))
}

/// BoxE margin-based ranking loss.
///
/// Encourages positive triples to score higher than negative triples
/// by at least the margin.
///
/// loss = max(0, margin - score_pos + score_neg)
#[cfg(feature = "boxe")]
pub fn boxe_loss(positive_score: f32, negative_score: f32, margin: f32) -> f32 {
    subsume_core::boxe::boxe_loss(positive_score, negative_score, margin)
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

    #[test]
    fn test_rotate_scoring() {
        // h = 1+0i, r = 0+1i (90 deg rotation), t = 0+1i
        // h * r = i, which matches t. Distance should be 0.
        let h = vec![1.0, 0.0];
        let r = vec![0.0, 1.0];
        let t = vec![0.0, 1.0];

        let score = ScoringFunction::RotatE.score(&h, &r, &t);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_complex_scoring() {
        // h = 1+0i, r = 1+0i, t = 1+0i
        // Re(1*1*1) = 1
        let h = vec![1.0, 0.0];
        let r = vec![1.0, 0.0];
        let t = vec![1.0, 0.0];

        let score = ScoringFunction::ComplEx.score(&h, &r, &t);
        assert!((score - 1.0).abs() < 1e-6);
    }
}
