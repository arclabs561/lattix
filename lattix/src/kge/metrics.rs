//! Rank-based evaluation metrics for link prediction.
//!
//! All functions take 1-indexed ranks (rank 1 = best possible).

/// Mean Reciprocal Rank: average of `1/rank` over all queries.
///
/// Higher is better. Range: (0, 1]. Returns 0.0 for empty input.
pub fn mean_reciprocal_rank(ranks: &[usize]) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    ranks.iter().map(|&r| 1.0 / r as f64).sum::<f64>() / ranks.len() as f64
}

/// Hits@k: fraction of queries where the correct answer ranks at or above `k`.
///
/// Higher is better. Range: [0, 1]. Returns 0.0 for empty input.
pub fn hits_at_k(ranks: &[usize], k: usize) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    ranks.iter().filter(|&&r| r <= k).count() as f64 / ranks.len() as f64
}

/// Mean Rank: arithmetic mean of all ranks.
///
/// Lower is better. Returns 0.0 for empty input.
pub fn mean_rank(ranks: &[usize]) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    ranks.iter().sum::<usize>() as f64 / ranks.len() as f64
}

/// Compute the realistic rank of the true entity given all scores.
///
/// Realistic rank (PyKEEN convention) is the mean of optimistic and pessimistic:
/// - Optimistic: number of entities with strictly better score + 1
/// - Pessimistic: number of entities with score at least as good
///
/// `true_score` is the score of the correct entity. `all_scores` includes
/// all candidate scores (including the true entity, excluding filtered entities).
/// Lower scores are assumed better (distance convention).
pub fn realistic_rank(all_scores: &[f32], true_score: f32) -> f64 {
    let mut strictly_better = 0usize;
    let mut at_least_as_good = 0usize;
    for &s in all_scores {
        if s < true_score {
            strictly_better += 1;
        }
        if s <= true_score {
            at_least_as_good += 1;
        }
    }
    let optimistic = strictly_better + 1;
    let pessimistic = at_least_as_good;
    (optimistic as f64 + pessimistic as f64) / 2.0
}

/// Adjusted Mean Rank: `mean_rank / expected_random_mean_rank`.
///
/// The expected mean rank under a uniform random model is `(num_entities + 1) / 2`.
/// AMR < 1.0 means better than random; AMR = 1.0 means random performance.
///
/// Returns 0.0 for empty input or zero entities.
pub fn adjusted_mean_rank(ranks: &[usize], num_entities: usize) -> f64 {
    if ranks.is_empty() || num_entities == 0 {
        return 0.0;
    }
    let mr = mean_rank(ranks);
    let expected = (num_entities as f64 + 1.0) / 2.0;
    mr / expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrr_basic() {
        let ranks = vec![1, 2, 4];
        let mrr = mean_reciprocal_rank(&ranks);
        // (1/1 + 1/2 + 1/4) / 3 = 1.75 / 3
        assert!((mrr - 0.5833).abs() < 0.001);
    }

    #[test]
    fn hits_at_k_basic() {
        let ranks = vec![1, 2, 5, 10, 20];
        assert!((hits_at_k(&ranks, 10) - 0.8).abs() < 1e-9);
        assert!((hits_at_k(&ranks, 1) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn mean_rank_basic() {
        let ranks = vec![1, 3, 5];
        assert!((mean_rank(&ranks) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn adjusted_mean_rank_basic() {
        let ranks = vec![1, 1, 1]; // MR = 1.0, expected = 50.5
        let amr = adjusted_mean_rank(&ranks, 100);
        assert!((amr - 1.0 / 50.5).abs() < 1e-9);
    }

    #[test]
    fn realistic_rank_no_ties() {
        // Scores: [0.1, 0.5, 0.3, 0.9], true = 0.3
        // strictly_better = 1 (0.1), at_least_as_good = 2 (0.1, 0.3)
        // optimistic = 2, pessimistic = 2, realistic = 2.0
        let scores = vec![0.1, 0.5, 0.3, 0.9];
        assert!((realistic_rank(&scores, 0.3) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn realistic_rank_with_ties() {
        // Scores: [0.3, 0.3, 0.3, 0.9], true = 0.3
        // strictly_better = 0, at_least_as_good = 3
        // optimistic = 1, pessimistic = 3, realistic = 2.0
        let scores = vec![0.3, 0.3, 0.3, 0.9];
        assert!((realistic_rank(&scores, 0.3) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn realistic_rank_best() {
        // True entity is the best scorer
        let scores = vec![0.1, 0.5, 0.9];
        assert!((realistic_rank(&scores, 0.1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn realistic_rank_worst() {
        // True entity is the worst scorer
        let scores = vec![0.1, 0.5, 0.9];
        // strictly_better = 2, at_least_as_good = 3
        // optimistic = 3, pessimistic = 3, realistic = 3.0
        assert!((realistic_rank(&scores, 0.9) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn empty_ranks() {
        assert_eq!(mean_reciprocal_rank(&[]), 0.0);
        assert_eq!(hits_at_k(&[], 10), 0.0);
        assert_eq!(mean_rank(&[]), 0.0);
        assert_eq!(adjusted_mean_rank(&[], 100), 0.0);
    }
}
