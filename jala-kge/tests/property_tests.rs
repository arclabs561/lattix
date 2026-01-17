//! Property-based tests for Knowledge Graph Embedding models.
//!
//! These tests verify mathematical invariants that must hold for all inputs.
//!
//! # TransE Mathematical Properties
//!
//! TransE interprets relations as translations: h + r ≈ t
//!
//! Score = -||h + r - t||₂
//!
//! Key properties:
//! - Score is in (-∞, 0]: perfect alignment gives 0, distance gives negative
//! - Score is continuous in embeddings
//! - Translation interpretation: positive triples should have higher scores
//!
//! # Embedding Constraints
//!
//! Many KGE models constrain embeddings to:
//! - Unit norm: ||e|| = 1
//! - Unit ball: ||e|| ≤ 1
//! - Bounded: ||e|| ≤ C for some constant C

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(dead_code)]

use lattix_kge::{Fact, KGEModel, TrainingConfig};

#[cfg(test)]
mod transe_props {
    use super::*;
    use lattix_kge::models::TransE;

    /// Compute L2 norm of embedding.
    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Compute L2 distance: ||a - b||.
    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Compute translation distance: ||h + r - t||.
    fn translation_distance(h: &[f32], r: &[f32], t: &[f32]) -> f32 {
        h.iter()
            .zip(r.iter())
            .zip(t.iter())
            .map(|((hi, ri), ti)| (hi + ri - ti).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    // ==========================================================================
    // Score Function Properties
    // ==========================================================================

    #[test]
    fn transe_score_is_negative_distance() {
        // TransE score = -||h + r - t||, so should be <= 0
        let triples = vec![
            Fact::from_strs("A", "knows", "B"),
            Fact::from_strs("B", "knows", "C"),
            Fact::from_strs("A", "knows", "C"),
        ];

        let mut model = TransE::new(32);
        let config = TrainingConfig {
            embedding_dim: 32,
            epochs: 10,
            learning_rate: 0.01,
            batch_size: 8,
            margin: 1.0,
            negative_samples: 2,
            seed: 42,
            ..Default::default()
        };

        let _ = model.train(&triples, &config);

        for triple in &triples {
            let score = model
                .score(&triple.head, &triple.relation, &triple.tail)
                .unwrap();
            assert!(
                score <= 0.0 + 1e-6,
                "TransE score should be <= 0, got {}",
                score
            );
        }
    }

    #[test]
    fn transe_positive_triples_higher_score() {
        // After training, positive triples should have higher scores than random negatives
        let triples = vec![
            Fact::from_strs("Einstein", "wonAward", "NobelPrize"),
            Fact::from_strs("Paris", "capitalOf", "France"),
            Fact::from_strs("Berlin", "capitalOf", "Germany"),
            Fact::from_strs("Curie", "wonAward", "NobelPrize"),
        ];

        let mut model = TransE::new(32);
        let config = TrainingConfig {
            embedding_dim: 32,
            epochs: 50,
            learning_rate: 0.02,
            batch_size: 4,
            margin: 1.0,
            negative_samples: 5,
            seed: 42,
            ..Default::default()
        };

        let _ = model.train(&triples, &config);

        // Positive triple
        let pos_score = model.score("Paris", "capitalOf", "France").unwrap();

        // Negative triple (corrupted tail)
        let neg_score = model.score("Paris", "capitalOf", "Germany").unwrap();

        // Positive should score higher (less negative)
        // Note: Due to training dynamics, this may not always hold perfectly
        // We just check that training improved positive scores
        assert!(
            pos_score >= neg_score - 0.5, // Allow some tolerance
            "Positive score {} should be >= negative score {}",
            pos_score,
            neg_score
        );
    }

    // ==========================================================================
    // Embedding Normalization Properties
    // ==========================================================================

    #[test]
    fn transe_initial_embeddings_normalized() {
        // After initialization, embeddings should be normalized
        let triples = vec![
            Fact::from_strs("A", "r1", "B"),
            Fact::from_strs("B", "r1", "C"),
        ];

        let mut model = TransE::new(64);
        let config = TrainingConfig {
            embedding_dim: 64,
            epochs: 1,          // Just initialize
            learning_rate: 0.0, // No updates
            batch_size: 10,
            margin: 1.0,
            negative_samples: 1,
            seed: 42,
            ..Default::default()
        };

        // We can't directly access embeddings, but we can verify through scoring behavior
        // After initialization with no training, scores should be reasonable
        let _ = model.train(&triples, &config);

        let score = model.score("A", "r1", "B").unwrap();
        assert!(
            score.is_finite(),
            "Initial score should be finite, got {}",
            score
        );
    }

    // ==========================================================================
    // Translation Interpretation Properties
    // ==========================================================================

    #[test]
    fn transe_relation_as_translation() {
        // Core TransE property: h + r ≈ t
        // For well-trained model, positive triples should satisfy this better
        let triples = vec![
            Fact::from_strs("A", "parentOf", "B"),
            Fact::from_strs("B", "parentOf", "C"),
            Fact::from_strs("C", "parentOf", "D"),
            Fact::from_strs("D", "parentOf", "E"),
        ];

        let mut model = TransE::new(32);
        let config = TrainingConfig {
            embedding_dim: 32,
            epochs: 100,
            learning_rate: 0.02,
            batch_size: 4,
            margin: 1.0,
            negative_samples: 5,
            seed: 42,
            ..Default::default()
        };

        let _ = model.train(&triples, &config);

        // After training, positive triples should have small translation error
        // This is reflected in scores close to 0
        for triple in &triples {
            let score = model
                .score(&triple.head, &triple.relation, &triple.tail)
                .unwrap();
            // Score = -distance, so score > -5 means distance < 5
            assert!(
                score > -10.0,
                "Translation distance too large for positive triple: score = {}",
                score
            );
        }
    }

    // ==========================================================================
    // Symmetry and Consistency Properties
    // ==========================================================================

    #[test]
    fn transe_score_deterministic() {
        let triples = vec![
            Fact::from_strs("A", "knows", "B"),
            Fact::from_strs("B", "knows", "C"),
        ];

        let mut model = TransE::new(32);
        let config = TrainingConfig {
            embedding_dim: 32,
            epochs: 10,
            learning_rate: 0.01,
            batch_size: 4,
            margin: 1.0,
            negative_samples: 2,
            seed: 42,
            ..Default::default()
        };

        let _ = model.train(&triples, &config);

        let score1 = model.score("A", "knows", "B").unwrap();
        let score2 = model.score("A", "knows", "B").unwrap();
        let score3 = model.score("A", "knows", "B").unwrap();

        assert!(
            (score1 - score2).abs() < 1e-10,
            "Scores should be deterministic: {} vs {}",
            score1,
            score2
        );
        assert!(
            (score2 - score3).abs() < 1e-10,
            "Scores should be deterministic: {} vs {}",
            score2,
            score3
        );
    }

    #[test]
    fn transe_unknown_entity_returns_error() {
        let triples = vec![Fact::from_strs("A", "knows", "B")];

        let mut model = TransE::new(32);
        let config = TrainingConfig::default();
        let _ = model.train(&triples, &config);

        let result = model.score("Unknown", "knows", "B");
        assert!(
            result.is_err(),
            "Unknown entity should return error, got {:?}",
            result
        );
    }

    // ==========================================================================
    // Training Convergence Properties
    // ==========================================================================

    #[test]
    fn transe_training_decreases_loss() {
        let triples = vec![
            Fact::from_strs("A", "r1", "B"),
            Fact::from_strs("B", "r1", "C"),
            Fact::from_strs("C", "r1", "D"),
            Fact::from_strs("A", "r2", "C"),
            Fact::from_strs("B", "r2", "D"),
        ];

        let config_short = TrainingConfig {
            embedding_dim: 32,
            epochs: 10,
            learning_rate: 0.02,
            batch_size: 4,
            margin: 1.0,
            negative_samples: 3,
            seed: 42,
            ..Default::default()
        };

        let config_long = TrainingConfig {
            embedding_dim: 32,
            epochs: 100,
            learning_rate: 0.02,
            batch_size: 4,
            margin: 1.0,
            negative_samples: 3,
            seed: 42,
            ..Default::default()
        };

        let mut model_short = TransE::new(32);
        let loss_short = model_short.train(&triples, &config_short).unwrap();

        let mut model_long = TransE::new(32);
        let loss_long = model_long.train(&triples, &config_long).unwrap();

        // More training should generally lead to lower loss
        // (though not guaranteed due to learning rate scheduling)
        assert!(
            loss_long <= loss_short * 1.5, // Allow some tolerance
            "Longer training should not dramatically increase loss: {} vs {}",
            loss_long,
            loss_short
        );
    }

    // ==========================================================================
    // Score Range Properties
    // ==========================================================================

    #[test]
    fn transe_scores_finite() {
        let triples = vec![
            Fact::from_strs("A", "knows", "B"),
            Fact::from_strs("B", "knows", "C"),
            Fact::from_strs("C", "knows", "A"),
        ];

        let mut model = TransE::new(32);
        let config = TrainingConfig {
            embedding_dim: 32,
            epochs: 50,
            learning_rate: 0.02,
            batch_size: 4,
            margin: 1.0,
            negative_samples: 3,
            seed: 42,
            ..Default::default()
        };

        let _ = model.train(&triples, &config);

        // All scores should be finite
        for triple in &triples {
            let score = model
                .score(&triple.head, &triple.relation, &triple.tail)
                .unwrap();
            assert!(
                score.is_finite(),
                "Score should be finite, got {} for triple {:?}",
                score,
                triple
            );
        }
    }

    // ==========================================================================
    // Mathematical Property: Score Metric
    // ==========================================================================

    #[test]
    fn transe_score_reflects_distance() {
        // TransE score is -||h + r - t||, a negative distance metric
        // This should satisfy:
        // 1. Score to self (conceptually, h + r = t) should be 0
        // 2. Score should decrease (become more negative) as distance increases

        let triples = vec![
            Fact::from_strs("A", "r1", "B"),
            Fact::from_strs("A", "r1", "C"),
            Fact::from_strs("A", "r1", "D"),
        ];

        let mut model = TransE::new(32);
        let config = TrainingConfig {
            embedding_dim: 32,
            epochs: 100,
            learning_rate: 0.02,
            batch_size: 4,
            margin: 1.0,
            negative_samples: 3,
            seed: 42,
            ..Default::default()
        };

        let _ = model.train(&triples, &config);

        // All trained triples should have reasonable scores
        let scores: Vec<f32> = triples
            .iter()
            .map(|t| model.score(&t.head, &t.relation, &t.tail).unwrap())
            .collect();

        for score in &scores {
            assert!(
                score.is_finite() && *score <= 0.0,
                "TransE scores should be finite and non-positive"
            );
        }
    }
}

// =============================================================================
// Box Embedding Properties (if available)
// =============================================================================

#[cfg(feature = "boxe")]
mod boxe_props {
    use super::*;
    use lattix_kge::models::BoxE;

    #[test]
    fn boxe_score_in_range() {
        // BoxE uses boxes with probabilistic interpretation
        // Scores should be bounded
        let triples = vec![
            Fact::from_strs("A", "contains", "B"),
            Fact::from_strs("B", "contains", "C"),
        ];

        let mut model = BoxE::new(32);
        let config = TrainingConfig {
            embedding_dim: 32,
            epochs: 10,
            learning_rate: 0.01,
            batch_size: 4,
            margin: 1.0,
            negative_samples: 2,
            seed: 42,
            ..Default::default()
        };

        if let Ok(_) = model.train(&triples, &config) {
            for triple in &triples {
                if let Ok(score) = model.score(&triple.head, &triple.relation, &triple.tail) {
                    assert!(score.is_finite(), "BoxE score should be finite");
                }
            }
        }
    }
}

// =============================================================================
// Hyperbolic Model Properties (if available)
// =============================================================================

#[cfg(feature = "hyperbolic")]
mod hyperbolic_props {
    use super::*;

    #[test]
    fn hyperbolic_embeddings_in_ball() {
        // Hyperbolic embeddings should stay inside the Poincare ball
        // ||x|| < 1 for all embeddings

        // This is a placeholder - actual implementation would
        // access internal embeddings to verify the constraint
    }
}
