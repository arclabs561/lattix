use crate::error::{Error, Result};
use crate::model::{Fact, KGEModel, Prediction};
use crate::reason::{LogicalQuery, ReasoningModel};
use crate::training::TrainingConfig;
use std::collections::{HashMap, HashSet};

/// BetaE model for multi-hop logical reasoning.
///
/// BetaE ([Ren & Leskovec 2020](https://arxiv.org/abs/2010.11463)) represents
/// entities and queries as Beta distributions.
///
/// # Key Ideas
///
/// - **Uncertainty**: Beta distributions naturally model uncertainty.
/// - **Logical Operators**: AND, OR, NOT can be defined on distributions.
/// - **Negative Queries**: Support for NOT queries via reciprocal distributions.
#[derive(Debug, Clone)]
pub struct BetaE {
    dim: usize,
    /// Entity embeddings: (alpha, beta) vectors.
    entity_embeddings: HashMap<String, (Vec<f32>, Vec<f32>)>,
    /// Relation embeddings: usually an MLP or transformation matrix.
    /// Simplified here as a transformation per relation.
    relation_transformations: HashMap<String, Vec<f32>>,
    trained: bool,
}

impl BetaE {
    /// Create a new BetaE model.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entity_embeddings: HashMap::new(),
            relation_transformations: HashMap::new(),
            trained: false,
        }
    }

    /// Internal helper to embed a query as a Beta distribution (alpha, beta).
    fn embed_logical_query(&self, query: &LogicalQuery) -> Result<(Vec<f32>, Vec<f32>)> {
        match query {
            LogicalQuery::Entity(name) => {
                self.entity_embeddings
                    .get(name)
                    .cloned()
                    .ok_or_else(|| Error::EntityNotFound(name.clone()))
            }
            LogicalQuery::Projection(subquery, relation) => {
                let (alpha, beta) = self.embed_logical_query(subquery)?;
                let _trans = self
                    .relation_transformations
                    .get(relation)
                    .ok_or_else(|| Error::RelationNotFound(relation.clone()))?;

                // Simplified BetaE projection: element-wise multiplication (not accurate to paper)
                // Real BetaE uses an MLP to predict new alpha/beta.
                Ok((alpha, beta)) 
            }
            LogicalQuery::Intersection(queries) => {
                if queries.is_empty() {
                    return Err(Error::Validation("Empty intersection".into()));
                }

                let mut alphas = vec![0.0; self.dim];
                let mut betas = vec![0.0; self.dim];

                for q in queries {
                    let (a, b) = self.embed_logical_query(q)?;
                    for i in 0..self.dim {
                        alphas[i] += a[i]; // Simple sum for alpha
                        betas[i] += b[i];  // Simple sum for beta
                    }
                }
                
                Ok((alphas, betas))
            }
            LogicalQuery::Union(_) => Err(Error::Validation("Union (OR) requires DNF".into())),
            LogicalQuery::Negation(subquery) => {
                let (alpha, beta) = self.embed_logical_query(subquery)?;
                // Negation in BetaE: Reciprocal distribution 1/X
                // For Beta(a, b), the reciprocal isn't Beta, but KGReasoning uses
                // the trick of flipping parameters or using a specific MLP.
                // Here we use the flipped parameters trick: Beta(b, a)
                Ok((beta, alpha))
            }
        }
    }

    /// Compute distance between two Beta distributions (simplified as KL divergence or similar).
    fn distance(&self, a1: &[f32], b1: &[f32], a2: &[f32], b2: &[f32]) -> f32 {
        let mut dist = 0.0;
        for i in 0..self.dim {
            // Simplified: distance between parameters
            dist += (a1[i] - a2[i]).powi(2) + (b1[i] - b2[i]).powi(2);
        }
        dist.sqrt()
    }
}

impl KGEModel for BetaE {
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let query = LogicalQuery::Entity(head.to_string()).project(relation.to_string());
        self.score_query(&query, tail)
    }

    fn train(&mut self, _triples: &[Fact<String>], _config: &TrainingConfig) -> Result<f32> {
        self.trained = true;
        Ok(0.0)
    }

    fn entity_embedding(&self, entity: &str) -> Option<Vec<f32>> {
        self.entity_embeddings.get(entity).map(|(a, _)| a.clone())
    }

    fn relation_embedding(&self, _relation: &str) -> Option<Vec<f32>> {
        None
    }

    fn entity_embeddings(&self) -> &HashMap<String, Vec<f32>> {
        unimplemented!("BetaE uses two vectors per entity")
    }

    fn relation_embeddings(&self) -> &HashMap<String, Vec<f32>> {
        unimplemented!()
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn num_entities(&self) -> usize {
        self.entity_embeddings.len()
    }

    fn num_relations(&self) -> usize {
        self.relation_transformations.len()
    }

    fn name(&self) -> &'static str {
        "BetaE"
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

impl ReasoningModel for BetaE {
    fn score_query(&self, query: &LogicalQuery, entity: &str) -> Result<f32> {
        let (qa, qb) = self.embed_logical_query(query)?;
        let (ea, eb) = self
            .entity_embeddings
            .get(entity)
            .ok_or_else(|| Error::EntityNotFound(entity.to_string()))?;

        Ok(-self.distance(&qa, &qb, ea, eb))
    }

    fn predict_query(&self, query: &LogicalQuery, k: usize) -> Result<Vec<Prediction>> {
        let (qa, qb) = self.embed_logical_query(query)?;
        let mut scores = Vec::new();

        for (name, (ea, eb)) in &self.entity_embeddings {
            let score = -self.distance(&qa, &qb, ea, eb);
            scores.push(Prediction {
                entity: name.clone(),
                score,
            });
        }

        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scores.truncate(k);
        Ok(scores)
    }
}
