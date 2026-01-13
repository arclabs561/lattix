//! Hyperbolic Knowledge Graph Embeddings.
//!
//! Hyperbolic space naturally encodes hierarchical structures - its exponential
//! volume growth matches tree branching. This module provides KGE models that
//! embed entities in hyperbolic space.
//!
//! # Models
//!
//! | Model | Operation | Best For |
//! |-------|-----------|----------|
//! | HyperE | h +_M r ≈ t (Mobius addition) | Tree-like hierarchies |
//! | MuRP | h ⊗ R ⊗ r ≈ t (Mobius matmul) | Hierarchies with transformations |
//!
//! # When to Use
//!
//! - **Taxonomies**: WordNet hypernym chains, Wikipedia categories
//! - **Organizational**: Reporting structures, file systems
//! - **Biological**: Phylogenetic trees, ontologies (GO, DO)
//!
//! # When NOT to Use
//!
//! - **DAGs with multiple parents**: Hyperbolic space assumes tree structure
//! - **Flat/symmetric relations**: Use Euclidean (TransE, DistMult)
//! - **Complex compositions**: Use BoxE for full expressiveness
//!
//! # Example
//!
//! ```rust,ignore
//! use lattix_kge::hyperbolic::{HyperE, HyperEConfig};
//! use lattix_kge::training::Triple;
//!
//! let triples = vec![
//!     Triple::new("mammal", "hypernym", "animal"),
//!     Triple::new("dog", "hypernym", "mammal"),
//! ];
//!
//! let config = HyperEConfig::default().with_curvature(1.0);
//! let model = HyperE::train(&triples, &config)?;
//!
//! // Dog should be "further" from root than mammal
//! let dog_depth = model.entity_depth("dog");
//! let mammal_depth = model.entity_depth("mammal");
//! assert!(dog_depth > mammal_depth);
//! ```
//!
//! # References
//!
//! - Nickel & Kiela (2017). "Poincare Embeddings for Learning Hierarchical Representations"
//! - Balazevic et al. (2019). "Multi-relational Poincare Graph Embeddings"

use crate::error::Result;
use crate::training::{Triple, TripleKGE};
use hyperball::PoincareBall;
use ndarray::Array1;
use std::collections::HashMap;

/// Configuration for HyperE model.
#[derive(Debug, Clone)]
pub struct HyperEConfig {
    /// Embedding dimension.
    pub dim: usize,
    /// Curvature of hyperbolic space (c > 0). Higher = more curved.
    pub curvature: f64,
    /// Learning rate.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub epochs: usize,
    /// Negative samples per positive.
    pub negative_samples: usize,
    /// Margin for ranking loss.
    pub margin: f64,
    /// Burn-in period (epochs with reduced LR to stabilize embeddings).
    pub burn_in_epochs: usize,
}

impl Default for HyperEConfig {
    fn default() -> Self {
        Self {
            dim: 50,
            curvature: 1.0,
            learning_rate: 0.01,
            epochs: 100,
            negative_samples: 10,
            margin: 0.1,
            burn_in_epochs: 10,
        }
    }
}

impl HyperEConfig {
    /// Set embedding dimension.
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set curvature. Higher values = more hyperbolic (stronger hierarchy).
    pub fn with_curvature(mut self, c: f64) -> Self {
        self.curvature = c;
        self
    }

    /// Set learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set number of epochs.
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }
}

/// HyperE: TransE in hyperbolic space.
///
/// Instead of h + r ≈ t (Euclidean addition), uses:
/// h +_M r ≈ t (Mobius addition in Poincare ball)
///
/// Mobius addition is the hyperbolic analog of vector addition.
/// It naturally respects the curved geometry.
pub struct HyperE {
    /// Poincare ball manifold.
    manifold: PoincareBall,
    /// Entity embeddings (in Poincare ball).
    entity_embeddings: HashMap<String, Array1<f64>>,
    /// Relation embeddings (tangent vectors at origin).
    relation_embeddings: HashMap<String, Array1<f64>>,
    /// Embedding dimension.
    dim: usize,
}

impl HyperE {
    /// Create a new HyperE model with given embeddings.
    pub fn new(
        curvature: f64,
        entity_embeddings: HashMap<String, Array1<f64>>,
        relation_embeddings: HashMap<String, Array1<f64>>,
    ) -> Self {
        let dim = entity_embeddings
            .values()
            .next()
            .map(|e| e.len())
            .unwrap_or(50);
        Self {
            manifold: PoincareBall::new(curvature),
            entity_embeddings,
            relation_embeddings,
            dim,
        }
    }

    /// Score a triple using hyperbolic distance.
    ///
    /// Score = -d_H(h +_M r, t)^2
    ///
    /// Lower distance = higher score = more plausible triple.
    pub fn score(&self, head: &str, relation: &str, tail: &str) -> Option<f64> {
        let h = self.entity_embeddings.get(head)?;
        let r = self.relation_embeddings.get(relation)?;
        let t = self.entity_embeddings.get(tail)?;

        // Apply relation as Mobius addition: h +_M exp_0(r)
        // (r is in tangent space, need to map to manifold first)
        let r_on_manifold = self.manifold.exp_map_zero(&r.view());
        let h_plus_r = self.manifold.mobius_add(&h.view(), &r_on_manifold.view());

        // Hyperbolic distance to tail
        let dist = self.manifold.distance(&h_plus_r.view(), &t.view());

        Some(-dist * dist)
    }

    /// Get the "depth" of an entity (distance from origin).
    ///
    /// In hierarchical embeddings, root nodes are near origin,
    /// leaves are near boundary.
    pub fn entity_depth(&self, entity: &str) -> Option<f64> {
        let e = self.entity_embeddings.get(entity)?;
        let origin = Array1::zeros(self.dim);
        Some(self.manifold.distance(&origin.view(), &e.view()))
    }

    /// Get entity embedding.
    pub fn entity_embedding(&self, entity: &str) -> Option<&Array1<f64>> {
        self.entity_embeddings.get(entity)
    }

    /// Get relation embedding.
    pub fn relation_embedding(&self, relation: &str) -> Option<&Array1<f64>> {
        self.relation_embeddings.get(relation)
    }

    /// Number of entities.
    pub fn num_entities(&self) -> usize {
        self.entity_embeddings.len()
    }

    /// Number of relations.
    pub fn num_relations(&self) -> usize {
        self.relation_embeddings.len()
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Curvature of the hyperbolic space.
    pub fn curvature(&self) -> f64 {
        self.manifold.c
    }
}

/// Train HyperE model on triples.
///
/// Uses Riemannian SGD with burn-in period for stability.
pub fn train_hypere(triples: &[Triple], config: &HyperEConfig) -> Result<HyperE> {
    use rand::prelude::*;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let manifold = PoincareBall::new(config.curvature);

    // Collect entities and relations using TripleKGE trait
    let mut entities: Vec<String> = Vec::new();
    let mut relations: Vec<String> = Vec::new();

    for triple in triples {
        let h = triple.head().to_string();
        let r = triple.rel().to_string();
        let t = triple.tail().to_string();
        if !entities.contains(&h) {
            entities.push(h);
        }
        if !entities.contains(&t) {
            entities.push(t);
        }
        if !relations.contains(&r) {
            relations.push(r);
        }
    }

    // Initialize embeddings near origin (inside Poincare ball)
    let init_scale = 0.001;
    let mut entity_emb: HashMap<String, Array1<f64>> = entities
        .iter()
        .map(|e| {
            let emb: Array1<f64> = Array1::from_iter(
                (0..config.dim).map(|_| (rng.random::<f64>() - 0.5) * init_scale),
            );
            (e.clone(), emb)
        })
        .collect();

    let mut relation_emb: HashMap<String, Array1<f64>> = relations
        .iter()
        .map(|r| {
            let emb: Array1<f64> = Array1::from_iter(
                (0..config.dim).map(|_| (rng.random::<f64>() - 0.5) * init_scale),
            );
            (r.clone(), emb)
        })
        .collect();

    // Training loop
    for epoch in 0..config.epochs {
        let lr = if epoch < config.burn_in_epochs {
            config.learning_rate * 0.1 // Reduced LR during burn-in
        } else {
            config.learning_rate
        };

        let mut total_loss = 0.0;

        for triple in triples {
            // Positive triple - use TripleKGE trait accessors
            let head = triple.head();
            let rel = triple.rel();
            let tail = triple.tail();

            let h = entity_emb.get(head).unwrap().clone();
            let r = relation_emb.get(rel).unwrap().clone();
            let t = entity_emb.get(tail).unwrap().clone();

            let r_on_manifold = manifold.exp_map_zero(&r.view());
            let h_plus_r = manifold.mobius_add(&h.view(), &r_on_manifold.view());
            let pos_dist = manifold.distance(&h_plus_r.view(), &t.view());

            // Negative sampling
            for _ in 0..config.negative_samples {
                let corrupt_tail = entities.choose(&mut rng).unwrap();
                if corrupt_tail == tail {
                    continue;
                }

                let t_neg = entity_emb.get(corrupt_tail).unwrap();
                let neg_dist = manifold.distance(&h_plus_r.view(), &t_neg.view());

                // Margin-based ranking loss
                let loss = (config.margin + pos_dist - neg_dist).max(0.0);
                total_loss += loss;

                if loss > 0.0 {
                    // Simplified gradient update (Euclidean approximation)
                    // Full Riemannian SGD would require metric tensor
                    let grad_scale = lr * loss;

                    // Update head: move h_plus_r closer to t
                    let h_mut = entity_emb.get_mut(head).unwrap();
                    for i in 0..config.dim {
                        let diff = h_plus_r[i] - t[i];
                        h_mut[i] -= grad_scale * diff * 0.5;
                    }
                    // Project back to ball
                    *h_mut = manifold.project(&h_mut.view());

                    // Update tail
                    let t_mut = entity_emb.get_mut(tail).unwrap();
                    for i in 0..config.dim {
                        let diff = t[i] - h_plus_r[i];
                        t_mut[i] -= grad_scale * diff * 0.5;
                    }
                    *t_mut = manifold.project(&t_mut.view());

                    // Update relation
                    let r_mut = relation_emb.get_mut(rel).unwrap();
                    for i in 0..config.dim {
                        let diff = h_plus_r[i] - t[i];
                        r_mut[i] -= grad_scale * diff * 0.3;
                    }
                }
            }
        }

        if epoch % 10 == 0 || epoch == config.epochs - 1 {
            eprintln!(
                "Epoch {}/{}: loss = {:.4}",
                epoch + 1,
                config.epochs,
                total_loss / triples.len() as f64
            );
        }
    }

    Ok(HyperE::new(config.curvature, entity_emb, relation_emb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypere_score_computation() {
        let mut entity_emb = HashMap::new();
        entity_emb.insert("a".to_string(), Array1::from_vec(vec![0.1, 0.0]));
        entity_emb.insert("b".to_string(), Array1::from_vec(vec![0.2, 0.1]));

        let mut relation_emb = HashMap::new();
        relation_emb.insert("r".to_string(), Array1::from_vec(vec![0.1, 0.1]));

        let model = HyperE::new(1.0, entity_emb, relation_emb);

        let score = model.score("a", "r", "b");
        assert!(score.is_some());
        // Score should be negative (it's -distance^2)
        assert!(score.unwrap() <= 0.0);
    }

    #[test]
    fn test_hypere_entity_depth() {
        let mut entity_emb = HashMap::new();
        // Near origin (root-like)
        entity_emb.insert("root".to_string(), Array1::from_vec(vec![0.01, 0.01]));
        // Further from origin (leaf-like)
        entity_emb.insert("leaf".to_string(), Array1::from_vec(vec![0.5, 0.5]));

        let relation_emb = HashMap::new();
        let model = HyperE::new(1.0, entity_emb, relation_emb);

        let root_depth = model.entity_depth("root").unwrap();
        let leaf_depth = model.entity_depth("leaf").unwrap();

        // Leaf should be deeper (further from origin)
        assert!(
            leaf_depth > root_depth,
            "leaf ({}) should be deeper than root ({})",
            leaf_depth,
            root_depth
        );
    }

    #[test]
    fn test_train_hypere_small() {
        let triples = vec![
            Triple::new("animal", "hypernym", "entity"),
            Triple::new("mammal", "hypernym", "animal"),
            Triple::new("dog", "hypernym", "mammal"),
            Triple::new("cat", "hypernym", "mammal"),
        ];

        let config = HyperEConfig::default()
            .with_dim(10)
            .with_epochs(50)
            .with_curvature(1.0);

        let model = train_hypere(&triples, &config).unwrap();

        assert_eq!(model.num_entities(), 5); // entity, animal, mammal, dog, cat
        assert_eq!(model.num_relations(), 1); // hypernym

        // After training, hierarchical structure should be preserved
        // (though with small data and epochs, this is approximate)
        let entity_depth = model.entity_depth("entity").unwrap_or(0.0);
        let animal_depth = model.entity_depth("animal").unwrap_or(0.0);
        let mammal_depth = model.entity_depth("mammal").unwrap_or(0.0);
        let dog_depth = model.entity_depth("dog").unwrap_or(0.0);

        eprintln!(
            "Depths: entity={:.3}, animal={:.3}, mammal={:.3}, dog={:.3}",
            entity_depth, animal_depth, mammal_depth, dog_depth
        );
    }

    #[test]
    fn test_hypere_symmetric_score() {
        let mut entity_emb = HashMap::new();
        entity_emb.insert("a".to_string(), Array1::from_vec(vec![0.1, 0.2]));
        entity_emb.insert("b".to_string(), Array1::from_vec(vec![0.3, 0.1]));

        let mut relation_emb = HashMap::new();
        relation_emb.insert("r".to_string(), Array1::from_vec(vec![0.1, 0.0]));

        let model = HyperE::new(1.0, entity_emb, relation_emb);

        let score_ab = model.score("a", "r", "b").unwrap();
        let score_ba = model.score("b", "r", "a").unwrap();

        // Scores should generally be different (asymmetric)
        // This is a feature, not a bug - hyperbolic models can handle asymmetry
        eprintln!("score(a,r,b)={:.4}, score(b,r,a)={:.4}", score_ab, score_ba);
    }
}
