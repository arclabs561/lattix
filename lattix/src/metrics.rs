//! Rank-based evaluation metrics for knowledge graph tasks.
//!
//! Standard metrics for evaluating knowledge graph embedding quality on
//! link prediction: MRR, Hits@k, Mean Rank, and Adjusted Mean Rank Index.
//!
//! Ranks are 1-indexed: rank 1 means the correct answer was the top candidate.
//! All functions return 0.0 for empty input.

/// Mean Reciprocal Rank: average of 1/rank for each test triple.
///
/// MRR gives exponentially more weight to top rankings:
/// rank 1 scores 1.0, rank 2 scores 0.5, rank 10 scores 0.1.
///
/// # Example
///
/// ```
/// use lattix::metrics::mean_reciprocal_rank;
///
/// let mrr = mean_reciprocal_rank(&[1, 2, 3]);
/// // (1/1 + 1/2 + 1/3) / 3
/// assert!((mrr - 0.6111).abs() < 1e-3);
/// ```
pub fn mean_reciprocal_rank(ranks: &[usize]) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    let sum: f64 = ranks.iter().map(|&r| 1.0 / r as f64).sum();
    sum / ranks.len() as f64
}

/// Hits@k: fraction of test triples ranked in the top k.
///
/// # Example
///
/// ```
/// use lattix::metrics::hits_at_k;
///
/// assert_eq!(hits_at_k(&[1, 2, 3], 1), 1.0 / 3.0);
/// assert_eq!(hits_at_k(&[1, 2, 3], 10), 1.0);
/// ```
pub fn hits_at_k(ranks: &[usize], k: usize) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    let hits = ranks.iter().filter(|&&r| r <= k).count();
    hits as f64 / ranks.len() as f64
}

/// Mean Rank: average rank across test triples. Lower is better.
///
/// # Example
///
/// ```
/// use lattix::metrics::mean_rank;
///
/// assert_eq!(mean_rank(&[1, 2, 3]), 2.0);
/// ```
pub fn mean_rank(ranks: &[usize]) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    let sum: f64 = ranks.iter().map(|&r| r as f64).sum();
    sum / ranks.len() as f64
}

/// Adjusted Mean Rank Index: chance-adjusted normalized mean rank.
///
/// AMRI = 1 - 2 * MR / (num_candidates + 1), where MR is the mean rank.
/// A random ranker scores 0.0, a perfect ranker scores 1.0, and a
/// maximally adversarial ranker scores -1.0.
///
/// See Berrendorf et al. (2020), "On the Ambiguity of Rank-Based Evaluation
/// of Entity Alignment or Link Prediction Methods."
///
/// Returns 0.0 if `ranks` is empty or `num_candidates` is 0.
///
/// # Example
///
/// ```
/// use lattix::metrics::adjusted_mean_rank;
///
/// // Perfect ranking (all rank 1) with 100 candidates.
/// let amri = adjusted_mean_rank(&[1, 1, 1], 100);
/// assert!(amri > 0.98);
/// ```
pub fn adjusted_mean_rank(ranks: &[usize], num_candidates: usize) -> f64 {
    if ranks.is_empty() || num_candidates == 0 {
        return 0.0;
    }
    let mr = mean_rank(ranks);
    1.0 - 2.0 * mr / (num_candidates as f64 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrr_all_rank_one() {
        assert_eq!(mean_reciprocal_rank(&[1, 1, 1]), 1.0);
    }

    #[test]
    fn mrr_mixed_ranks() {
        let mrr = mean_reciprocal_rank(&[1, 2, 3]);
        let expected = (1.0 + 0.5 + 1.0 / 3.0) / 3.0;
        assert!(
            (mrr - expected).abs() < 1e-10,
            "MRR = {mrr}, expected {expected}"
        );
    }

    #[test]
    fn hits_at_1() {
        let h = hits_at_k(&[1, 2, 3], 1);
        assert!((h - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn hits_at_10_all_hit() {
        assert_eq!(hits_at_k(&[1, 2, 3], 10), 1.0);
    }

    #[test]
    fn hits_at_k_none_hit() {
        assert_eq!(hits_at_k(&[4, 5, 6], 3), 0.0);
    }

    #[test]
    fn mean_rank_simple() {
        assert_eq!(mean_rank(&[1, 2, 3]), 2.0);
    }

    #[test]
    fn empty_ranks_return_zero() {
        assert_eq!(mean_reciprocal_rank(&[]), 0.0);
        assert_eq!(hits_at_k(&[], 10), 0.0);
        assert_eq!(mean_rank(&[]), 0.0);
        assert_eq!(adjusted_mean_rank(&[], 100), 0.0);
    }

    #[test]
    fn adjusted_mean_rank_perfect() {
        // All rank 1, 100 candidates: AMRI = 1 - 2/101 ~= 0.9802
        let amri = adjusted_mean_rank(&[1, 1, 1], 100);
        let expected = 1.0 - 2.0 / 101.0;
        assert!(
            (amri - expected).abs() < 1e-10,
            "AMRI = {amri}, expected {expected}"
        );
    }

    #[test]
    fn adjusted_mean_rank_random() {
        // Random ranker: mean rank = (N+1)/2 => AMRI = 0
        let n = 100;
        let mid = (n + 1) / 2; // 50 (integer), close to 50.5
        let amri = adjusted_mean_rank(&[mid, mid, mid], n);
        // With integer midpoint: AMRI = 1 - 2*50/101 = 1 - 100/101 ~= 0.0099
        assert!(amri.abs() < 0.02, "AMRI = {amri}, expected ~0.0");
    }

    #[test]
    fn adjusted_mean_rank_zero_candidates() {
        assert_eq!(adjusted_mean_rank(&[1, 2], 0), 0.0);
    }

    #[test]
    fn mrr_single_element() {
        assert_eq!(mean_reciprocal_rank(&[1]), 1.0);
        assert!((mean_reciprocal_rank(&[5]) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn mean_rank_single_element() {
        assert_eq!(mean_rank(&[7]), 7.0);
    }
}
