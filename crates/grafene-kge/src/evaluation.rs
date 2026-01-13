//! Rank-based evaluation for knowledge graph embeddings.
//!
//! This module implements the standard KGE evaluation protocol used in
//! link prediction benchmarks (FB15k-237, WN18RR, etc.).
//!
//! # The Link Prediction Task
//!
//! Given a test triple (h, r, t), we evaluate by:
//! 1. **Tail prediction**: Score all (h, r, ?) candidates, rank true tail
//! 2. **Head prediction**: Score all (?, r, t) candidates, rank true head
//!
//! # Filtered vs Raw Metrics
//!
//! | Setting | Description | Use Case |
//! |---------|-------------|----------|
//! | Raw | All entities as negatives | Pessimistic estimate |
//! | Filtered | Remove known true triples | Standard benchmark |
//!
//! **Filtered is standard**: Raw penalizes correct predictions of triples
//! that happen to be in training/valid data.
//!
//! # Standard Metrics
//!
//! | Metric | Range | Description |
//! |--------|-------|-------------|
//! | MRR | (0, 1] | Mean Reciprocal Rank: average of 1/rank |
//! | Hits@1 | [0, 1] | Fraction with rank = 1 |
//! | Hits@3 | [0, 1] | Fraction with rank <= 3 |
//! | Hits@10 | [0, 1] | Fraction with rank <= 10 |
//!
//! # References
//!
//! - Bordes et al. (2013): Original TransE evaluation
//! - Ruffinelli et al. (2020): "You CAN Teach an Old Dog New Tricks"
//!   (analysis of evaluation pitfalls)
//! - Ali et al. (2020): "Bringing Light Into the Dark" (PyKEEN framework)

use std::collections::{HashMap, HashSet};

use grafene_core::{EntityId, RelationType, Triple};

use crate::scoring::ScoringFunction;

/// Rank-based evaluation results.
#[derive(Debug, Clone, Default)]
pub struct RankMetrics {
    /// Mean Reciprocal Rank: E\[1/rank\]
    pub mrr: f64,
    /// Mean Rank: E\[rank\]
    pub mr: f64,
    /// Hits@1: P(rank = 1)
    pub hits_at_1: f64,
    /// Hits@3: P(rank <= 3)
    pub hits_at_3: f64,
    /// Hits@10: P(rank <= 10)
    pub hits_at_10: f64,
    /// Number of test triples evaluated
    pub num_triples: usize,
}

impl RankMetrics {
    /// Compute metrics from a list of ranks.
    pub fn from_ranks(ranks: &[usize]) -> Self {
        if ranks.is_empty() {
            return Self::default();
        }

        let n = ranks.len() as f64;
        let mrr: f64 = ranks.iter().map(|&r| 1.0 / r as f64).sum::<f64>() / n;
        let mr: f64 = ranks.iter().map(|&r| r as f64).sum::<f64>() / n;
        let hits_at_1 = ranks.iter().filter(|&&r| r == 1).count() as f64 / n;
        let hits_at_3 = ranks.iter().filter(|&&r| r <= 3).count() as f64 / n;
        let hits_at_10 = ranks.iter().filter(|&&r| r <= 10).count() as f64 / n;

        Self {
            mrr,
            mr,
            hits_at_1,
            hits_at_3,
            hits_at_10,
            num_triples: ranks.len(),
        }
    }

    /// Merge metrics from multiple evaluation runs.
    pub fn merge(metrics: &[Self]) -> Self {
        if metrics.is_empty() {
            return Self::default();
        }

        let total_triples: usize = metrics.iter().map(|m| m.num_triples).sum();
        if total_triples == 0 {
            return Self::default();
        }

        let total_f = total_triples as f64;

        Self {
            mrr: metrics
                .iter()
                .map(|m| m.mrr * m.num_triples as f64)
                .sum::<f64>()
                / total_f,
            mr: metrics
                .iter()
                .map(|m| m.mr * m.num_triples as f64)
                .sum::<f64>()
                / total_f,
            hits_at_1: metrics
                .iter()
                .map(|m| m.hits_at_1 * m.num_triples as f64)
                .sum::<f64>()
                / total_f,
            hits_at_3: metrics
                .iter()
                .map(|m| m.hits_at_3 * m.num_triples as f64)
                .sum::<f64>()
                / total_f,
            hits_at_10: metrics
                .iter()
                .map(|m| m.hits_at_10 * m.num_triples as f64)
                .sum::<f64>()
                / total_f,
            num_triples: total_triples,
        }
    }

    /// Format as summary string.
    pub fn summary(&self) -> String {
        format!(
            "MRR: {:.4} | MR: {:.1} | H@1: {:.3} | H@3: {:.3} | H@10: {:.3} (n={})",
            self.mrr, self.mr, self.hits_at_1, self.hits_at_3, self.hits_at_10, self.num_triples
        )
    }
}

/// Triple identity for evaluation (hashable, without metadata).
///
/// Unlike [`Triple`], this is only the (head, relation, tail) tuple
/// without confidence/source metadata. Used for filtering known triples.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct EvalTriple {
    /// Head entity (subject).
    pub head: EntityId,
    /// Relation type (predicate).
    pub relation: RelationType,
    /// Tail entity (object).
    pub tail: EntityId,
}

impl EvalTriple {
    /// Create a new evaluation triple.
    pub fn new(
        head: impl Into<EntityId>,
        relation: impl Into<RelationType>,
        tail: impl Into<EntityId>,
    ) -> Self {
        Self {
            head: head.into(),
            relation: relation.into(),
            tail: tail.into(),
        }
    }
}

impl From<&Triple> for EvalTriple {
    /// Convert from [`Triple`], discarding metadata.
    fn from(t: &Triple) -> Self {
        Self {
            head: t.subject.clone(),
            relation: t.predicate.clone(),
            tail: t.object.clone(),
        }
    }
}

impl From<Triple> for EvalTriple {
    fn from(t: Triple) -> Self {
        Self {
            head: t.subject,
            relation: t.predicate,
            tail: t.object,
        }
    }
}

/// Link prediction evaluator.
///
/// Implements filtered ranking evaluation following the standard protocol.
///
/// # Example
///
/// ```rust,ignore
/// use grafene_kge::evaluation::{Evaluator, EvalTriple};
///
/// let mut eval = Evaluator::new();
///
/// // Add known triples (training + validation)
/// eval.add_known_triple(EvalTriple::new("Einstein", "won", "NobelPrize"));
///
/// // Evaluate test triple
/// let metrics = eval.evaluate_triples(&test_triples, &embeddings);
/// println!("{}", metrics.summary());
/// ```
pub struct Evaluator {
    /// All known triples (training + validation + test).
    /// Used for filtering false negatives.
    known_triples: HashSet<EvalTriple>,
}

impl Evaluator {
    /// Create a new evaluator with no known triples.
    pub fn new() -> Self {
        Self {
            known_triples: HashSet::new(),
        }
    }

    /// Add a known triple for filtering.
    pub fn add_known_triple(&mut self, triple: EvalTriple) {
        self.known_triples.insert(triple);
    }

    /// Add multiple known triples.
    pub fn add_known_triples(&mut self, triples: impl IntoIterator<Item = EvalTriple>) {
        self.known_triples.extend(triples);
    }

    /// Check if a triple is known (for filtering).
    pub fn is_known(&self, triple: &EvalTriple) -> bool {
        self.known_triples.contains(triple)
    }

    /// Compute filtered rank for tail prediction.
    ///
    /// Ranks entities by score for (head, relation, ?), returns rank of true tail.
    /// Entities with known true triples are filtered out.
    pub fn rank_tail_filtered(
        &self,
        head: &EntityId,
        relation: &RelationType,
        true_tail: &EntityId,
        entity_embeddings: &HashMap<String, Vec<f32>>,
        relation_embeddings: &HashMap<String, Vec<f32>>,
        scoring_fn: ScoringFunction,
    ) -> Option<usize> {
        let h_emb = entity_embeddings.get(head.as_str())?;
        let r_emb = relation_embeddings.get(relation.as_str())?;
        let true_tail_emb = entity_embeddings.get(true_tail.as_str())?;

        let true_score = scoring_fn.score(h_emb, r_emb, true_tail_emb);

        let mut rank = 1;
        for (entity, t_emb) in entity_embeddings {
            // Skip the true tail
            if entity == true_tail.as_str() {
                continue;
            }

            // Filter known triples
            let candidate = EvalTriple::new(head.clone(), relation.clone(), entity.as_str());
            if self.known_triples.contains(&candidate) {
                continue;
            }

            let score = scoring_fn.score(h_emb, r_emb, t_emb);
            if score > true_score {
                rank += 1;
            }
        }

        Some(rank)
    }

    /// Compute filtered rank for head prediction.
    ///
    /// Ranks entities by score for (?, relation, tail), returns rank of true head.
    pub fn rank_head_filtered(
        &self,
        true_head: &EntityId,
        relation: &RelationType,
        tail: &EntityId,
        entity_embeddings: &HashMap<String, Vec<f32>>,
        relation_embeddings: &HashMap<String, Vec<f32>>,
        scoring_fn: ScoringFunction,
    ) -> Option<usize> {
        let t_emb = entity_embeddings.get(tail.as_str())?;
        let r_emb = relation_embeddings.get(relation.as_str())?;
        let true_head_emb = entity_embeddings.get(true_head.as_str())?;

        let true_score = scoring_fn.score(true_head_emb, r_emb, t_emb);

        let mut rank = 1;
        for (entity, h_emb) in entity_embeddings {
            if entity == true_head.as_str() {
                continue;
            }

            let candidate = EvalTriple::new(entity.as_str(), relation.clone(), tail.clone());
            if self.known_triples.contains(&candidate) {
                continue;
            }

            let score = scoring_fn.score(h_emb, r_emb, t_emb);
            if score > true_score {
                rank += 1;
            }
        }

        Some(rank)
    }

    /// Evaluate a set of test triples.
    ///
    /// For each triple, computes both head and tail ranks and averages them.
    pub fn evaluate_triples(
        &self,
        test_triples: &[EvalTriple],
        entity_embeddings: &HashMap<String, Vec<f32>>,
        relation_embeddings: &HashMap<String, Vec<f32>>,
        scoring_fn: ScoringFunction,
    ) -> RankMetrics {
        let mut ranks = Vec::new();

        for triple in test_triples {
            // Tail prediction
            if let Some(tail_rank) = self.rank_tail_filtered(
                &triple.head,
                &triple.relation,
                &triple.tail,
                entity_embeddings,
                relation_embeddings,
                scoring_fn,
            ) {
                ranks.push(tail_rank);
            }

            // Head prediction
            if let Some(head_rank) = self.rank_head_filtered(
                &triple.head,
                &triple.relation,
                &triple.tail,
                entity_embeddings,
                relation_embeddings,
                scoring_fn,
            ) {
                ranks.push(head_rank);
            }
        }

        RankMetrics::from_ranks(&ranks)
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute rank of target value in a sorted (descending) score list.
///
/// If scores are tied, uses pessimistic ranking (highest rank among ties).
pub fn compute_rank(target_score: f32, all_scores: &[f32]) -> usize {
    let mut rank = 1;
    for &score in all_scores {
        if score > target_score {
            rank += 1;
        }
    }
    rank
}

/// Compute optimistic rank (lowest rank among ties).
pub fn compute_rank_optimistic(target_score: f32, all_scores: &[f32]) -> usize {
    let mut rank = 1;
    for &score in all_scores {
        if score >= target_score && (score - target_score).abs() > 1e-9 {
            rank += 1;
        }
    }
    rank
}

/// Compute average rank (average of optimistic and pessimistic).
/// This is the standard in many implementations.
pub fn compute_rank_average(target_score: f32, all_scores: &[f32]) -> f64 {
    let mut better = 0;
    let mut tied = 0;

    for &score in all_scores {
        if (score - target_score).abs() < 1e-9 {
            tied += 1;
        } else if score > target_score {
            better += 1;
        }
    }

    // Average rank = better + (tied + 1) / 2
    better as f64 + (tied as f64 + 1.0) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_metrics_from_ranks() {
        let ranks = vec![1, 2, 3, 10, 100];
        let metrics = RankMetrics::from_ranks(&ranks);

        // MRR = (1/1 + 1/2 + 1/3 + 1/10 + 1/100) / 5 = (1.0 + 0.5 + 0.333 + 0.1 + 0.01) / 5
        assert!((metrics.mrr - 0.3886).abs() < 0.001);

        // MR = (1 + 2 + 3 + 10 + 100) / 5 = 23.2
        assert!((metrics.mr - 23.2).abs() < 0.1);

        // Hits@1 = 1/5 = 0.2
        assert!((metrics.hits_at_1 - 0.2).abs() < 1e-6);

        // Hits@3 = 3/5 = 0.6
        assert!((metrics.hits_at_3 - 0.6).abs() < 1e-6);

        // Hits@10 = 4/5 = 0.8
        assert!((metrics.hits_at_10 - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_rank_metrics_empty() {
        let metrics = RankMetrics::from_ranks(&[]);
        assert_eq!(metrics.num_triples, 0);
    }

    #[test]
    fn test_compute_rank() {
        let scores = vec![0.9, 0.8, 0.7, 0.5, 0.3];

        // Highest score -> rank 1
        assert_eq!(compute_rank(0.9, &scores), 1);

        // Second highest -> rank 2
        assert_eq!(compute_rank(0.8, &scores), 2);

        // Lowest -> rank 5
        assert_eq!(compute_rank(0.3, &scores), 5);
    }

    #[test]
    fn test_evaluator_filtering() {
        let mut eval = Evaluator::new();

        // Add known triple
        eval.add_known_triple(EvalTriple::new("A", "r", "B"));

        assert!(eval.is_known(&EvalTriple::new("A", "r", "B")));
        assert!(!eval.is_known(&EvalTriple::new("A", "r", "C")));
    }

    #[test]
    fn test_simple_evaluation() {
        // Create a minimal test case where we know the expected ranks
        let mut entity_emb: HashMap<String, Vec<f32>> = HashMap::new();
        let mut relation_emb: HashMap<String, Vec<f32>> = HashMap::new();

        // TransE: h + r = t means score = -||h + r - t|| = 0 (best)
        // Entity A at [0, 0], Entity B at [1, 0], Relation r as [1, 0]
        // So A + r = [1, 0] = B (perfect match)
        entity_emb.insert("A".to_string(), vec![0.0, 0.0]);
        entity_emb.insert("B".to_string(), vec![1.0, 0.0]);
        entity_emb.insert("C".to_string(), vec![5.0, 5.0]); // Far away
        relation_emb.insert("r".to_string(), vec![1.0, 0.0]);

        let eval = Evaluator::new();

        // Test triple: (A, r, B)
        // For tail prediction: A + r = [1, 0], B = [1, 0], C = [5, 5]
        // Distances: B -> 0, C -> ||[1,0] - [5,5]|| = sqrt(41)
        // B should rank 1

        let head = EntityId::new("A");
        let rel = RelationType::new("r");
        let tail = EntityId::new("B");

        let rank = eval
            .rank_tail_filtered(
                &head,
                &rel,
                &tail,
                &entity_emb,
                &relation_emb,
                ScoringFunction::TransE,
            )
            .unwrap();

        assert_eq!(rank, 1, "Perfect match should rank 1");
    }

    #[test]
    fn test_metrics_merge() {
        let m1 = RankMetrics {
            mrr: 0.5,
            mr: 2.0,
            hits_at_1: 0.5,
            hits_at_3: 0.75,
            hits_at_10: 1.0,
            num_triples: 4,
        };

        let m2 = RankMetrics {
            mrr: 1.0,
            mr: 1.0,
            hits_at_1: 1.0,
            hits_at_3: 1.0,
            hits_at_10: 1.0,
            num_triples: 1,
        };

        let merged = RankMetrics::merge(&[m1.clone(), m2.clone()]);

        // Weighted average: (0.5 * 4 + 1.0 * 1) / 5 = 3.0 / 5 = 0.6
        assert!((merged.mrr - 0.6).abs() < 1e-6);
        assert_eq!(merged.num_triples, 5);
    }
}
