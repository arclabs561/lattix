use crate::error::{Error, Result};
use crate::model::{KGEModel, Prediction};
use crate::reason::{LogicalQuery, ReasoningModel};
use crate::training::TrainingConfig;
use std::collections::{HashMap, HashSet};

/// GQE (Graph Query Embedding) model.
///
/// GQE ([Hamilton et al. 2018](https://arxiv.org/abs/1806.01445)) represents
/// logical queries as points in embedding space.
///
/// # Key Ideas
///
/// - **Entities and Queries** are points.
/// - **Intersection** is handled via a mean or MLP operator.
/// - **Score** is the negative distance between entity and query points.
#[derive(Debug, Clone)]
pub struct GQE {
    dim: usize,
    entity_embeddings: HashMap<String, Vec<f32>>,
    relation_embeddings: HashMap<String, Vec<f32>>,
    trained: bool,
}

impl GQE {
    /// Create a new GQE model.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            trained: false,
        }
    }

    /// Embed a logical query as a point.
    fn embed_logical_query(&self, query: &LogicalQuery) -> Result<Vec<f32>> {
        match query {
            LogicalQuery::Entity(name) => {
                self.entity_embeddings
                    .get(name)
                    .cloned()
                    .ok_or_else(|| Error::EntityNotFound(name.clone()))
            }
            LogicalQuery::Projection(subquery, relation) => {
                let q_emb = self.embed_logical_query(subquery)?;
                let r_emb = self
                    .relation_embeddings
                    .get(relation)
                    .ok_or_else(|| Error::RelationNotFound(relation.clone()))?;

                let mut new_emb = Vec::with_capacity(self.dim);
                for i in 0..self.dim {
                    new_emb.push(q_emb[i] + r_emb[i]);
                }
                Ok(new_emb)
            }
            LogicalQuery::Intersection(queries) => {
                if queries.is_empty() {
                    return Err(Error::Validation("Empty intersection".into()));
                }

                let mut new_emb = vec![0.0; self.dim];
                for q in queries {
                    let q_emb = self.embed_logical_query(q)?;
                    for i in 0..self.dim {
                        new_emb[i] += q_emb[i];
                    }
                }

                for i in 0..self.dim {
                    new_emb[i] /= queries.len() as f32;
                }
                Ok(new_emb)
            }
            LogicalQuery::Union(_) => Err(Error::Validation("Union not supported in GQE".into())),
            LogicalQuery::Negation(_) => Err(Error::Validation("Negation not supported in GQE".into())),
        }
    }
}

impl KGEModel for GQE {
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let query = LogicalQuery::Entity(head.to_string()).project(relation.to_string());
        self.score_query(&query, tail)
    }

    fn train(&mut self, _triples: &[crate::model::Fact<String>], _config: &TrainingConfig) -> Result<f32> {
        self.trained = true;
        Ok(0.0)
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
        "GQE"
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

impl ReasoningModel for GQE {
    fn score_query(&self, query: &LogicalQuery, entity: &str) -> Result<f32> {
        let q_emb = self.embed_logical_query(query)?;
        let e_emb = self
            .entity_embeddings
            .get(entity)
            .ok_or_else(|| Error::EntityNotFound(entity.to_string()))?;

        let mut dist_sq = 0.0;
        for i in 0..self.dim {
            dist_sq += (q_emb[i] - e_emb[i]).powi(2);
        }
        Ok(-dist_sq.sqrt())
    }

    fn predict_query(&self, query: &LogicalQuery, k: usize) -> Result<Vec<Prediction>> {
        let q_emb = self.embed_logical_query(query)?;
        let mut scores = Vec::new();

        for (name, emb) in &self.entity_embeddings {
            let mut dist_sq = 0.0;
            for i in 0..self.dim {
                dist_sq += (q_emb[i] - emb[i]).powi(2);
            }
            scores.push(Prediction {
                entity: name.clone(),
                score: -dist_sq.sqrt(),
            });
        }

        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scores.truncate(k);
        Ok(scores)
    }
}
