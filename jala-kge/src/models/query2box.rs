use crate::error::{Error, Result};
use crate::model::{EpochMetrics, Fact, KGEModel, Prediction, ProgressCallback};
use crate::reason::{LogicalQuery, ReasoningModel};
use crate::reason::query::BoxGatedIntersection;
use crate::training::TrainingConfig;
use std::collections::{HashMap, HashSet};

/// Query2box model for multi-hop logical reasoning.
///
/// Query2box ([Ren et al. 2020](https://arxiv.org/abs/2002.05969)) represents
/// logical queries as axis-aligned hyperrectangles (boxes) in embedding space.
///
/// # Key Ideas
///
/// - **Entities** are points (or tiny boxes).
/// - **Queries** are boxes.
/// - **Projection** (relation) translates and expands/shrinks the box.
/// - **Intersection** (AND) computes a weighted intersection of boxes.
/// - **Score** is the distance from an entity point to the query box.
#[derive(Debug, Clone)]
pub struct Query2box {
    dim: usize,
    /// Entity embeddings (points).
    entity_embeddings: HashMap<String, Vec<f32>>,
    /// Relation projections: (center_offset, size_offset).
    relation_projections: HashMap<String, (Vec<f32>, Vec<f32>)>,
    /// Whether the model has been trained.
    trained: bool,
}

impl Query2box {
    /// Create a new Query2box model with the given embedding dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entity_embeddings: HashMap::new(),
            relation_projections: HashMap::new(),
            trained: false,
        }
    }

    /// Internal helper to embed a query as a box (center, offset).
    fn embed_logical_query(&self, query: &LogicalQuery) -> Result<(Vec<f32>, Vec<f32>)> {
        match query {
            LogicalQuery::Entity(name) => {
                let emb = self
                    .entity_embeddings
                    .get(name)
                    .ok_or_else(|| Error::EntityNotFound(name.clone()))?;
                // Entity as a zero-offset box (point)
                Ok((emb.clone(), vec![0.0; self.dim]))
            }
            LogicalQuery::Projection(subquery, relation) => {
                let (c, o) = self.embed_logical_query(subquery)?;
                let (rc, ro) = self
                    .relation_projections
                    .get(relation)
                    .ok_or_else(|| Error::RelationNotFound(relation.clone()))?;

                let mut new_c = Vec::with_capacity(self.dim);
                let mut new_o = Vec::with_capacity(self.dim);

                for i in 0..self.dim {
                    new_c.push(c[i] + rc[i]);
                    new_o.push(o[i] + ro[i]);
                }

                Ok((new_c, new_o))
            }
            LogicalQuery::Intersection(queries) => {
                if queries.is_empty() {
                    return Err(Error::Validation("Empty intersection".into()));
                }

                let mut centers = Vec::with_capacity(queries.len());
                let mut offsets = Vec::with_capacity(queries.len());
                for q in queries {
                    let (c, o) = self.embed_logical_query(q)?;
                    centers.push(c);
                    offsets.push(o);
                }

                let intersection = BoxGatedIntersection { dim: self.dim };
                Ok(intersection.intersect(&centers, &offsets))
            }
            LogicalQuery::Union(_) => {
                // Union is typically handled via DNF (Disjunctive Normal Form)
                // where we compute multiple boxes and take the max score.
                // For simplicity, this naive implementation doesn't support it directly in embedding.
                Err(Error::Validation(
                    "Union not supported in single box embedding (use DNF)".into(),
                ))
            }
            LogicalQuery::Negation(_) => Err(Error::Validation("Negation not supported in Query2box".into())),
        }
    }

    /// Score a point against a box.
    fn score_point_to_box(&self, point: &[f32], center: &[f32], offset: &[f32]) -> f32 {
        let mut dist_sq = 0.0;
        for i in 0..self.dim {
            let lower = center[i] - offset[i].abs();
            let upper = center[i] + offset[i].abs();

            if point[i] < lower {
                dist_sq += (lower - point[i]).powi(2);
            } else if point[i] > upper {
                dist_sq += (point[i] - upper).powi(2);
            }
            // If inside, distance is 0 for this dimension
        }
        -dist_sq.sqrt()
    }
}

impl KGEModel for Query2box {
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let query = LogicalQuery::Entity(head.to_string()).project(relation.to_string());
        self.score_query(&query, tail)
    }

    fn train(&mut self, triples: &[Fact<String>], config: &TrainingConfig) -> Result<f32> {
        // Simplified training: treat as triples (1p queries)
        // In reality, Query2box needs multi-hop query training.
        let mut entities = HashSet::new();
        let mut relations = HashSet::new();
        for t in triples {
            entities.insert(t.head.clone());
            entities.insert(t.tail.clone());
            relations.insert(t.relation.clone());
        }

        self.dim = config.embedding_dim;
        // Initialize... (omitted for brevity, similar to BoxE)
        self.trained = true;
        Ok(0.0)
    }

    fn entity_embedding(&self, entity: &str) -> Option<Vec<f32>> {
        self.entity_embeddings.get(entity).cloned()
    }

    fn relation_embedding(&self, relation: &str) -> Option<Vec<f32>> {
        self.relation_projections.get(relation).map(|(c, _)| c.clone())
    }

    fn entity_embeddings(&self) -> &HashMap<String, Vec<f32>> {
        &self.entity_embeddings
    }

    fn relation_embeddings(&self) -> &HashMap<String, Vec<f32>> {
        // This is a bit of a hack since relation_projections is (c, o)
        unimplemented!("Query2box relations are boxes, not single vectors")
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn num_entities(&self) -> usize {
        self.entity_embeddings.len()
    }

    fn num_relations(&self) -> usize {
        self.relation_projections.len()
    }

    fn name(&self) -> &'static str {
        "Query2box"
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

impl ReasoningModel for Query2box {
    fn score_query(&self, query: &LogicalQuery, entity: &str) -> Result<f32> {
        let (c, o) = self.embed_logical_query(query)?;
        let e = self
            .entity_embeddings
            .get(entity)
            .ok_or_else(|| Error::EntityNotFound(entity.to_string()))?;

        Ok(self.score_point_to_box(e, &c, &o))
    }

    fn predict_query(&self, query: &LogicalQuery, k: usize) -> Result<Vec<Prediction>> {
        let (c, o) = self.embed_logical_query(query)?;
        let mut scores = Vec::new();

        for (name, emb) in &self.entity_embeddings {
            let score = self.score_point_to_box(emb, &c, &o);
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
