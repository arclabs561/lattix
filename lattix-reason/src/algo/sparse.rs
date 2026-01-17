use crate::{LogicalQuery, Reasoner, Result};
use lattix_core::KnowledgeGraph;
use lattix_kge::Prediction;
use std::collections::HashMap;

/// Directed Sparse Matrix in CSR-like format.
struct SparseMatrix {
    num_nodes: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
}

impl SparseMatrix {
    fn from_edges(num_nodes: usize, edges: &[(usize, usize)]) -> Self {
        let mut sorted_edges = edges.to_vec();
        sorted_edges.sort_by_key(|e| e.0);

        let mut row_ptr = vec![0; num_nodes + 1];
        let mut col_idx = Vec::with_capacity(edges.len());

        for (u, v) in sorted_edges {
            row_ptr[u + 1] += 1;
            col_idx.push(v);
        }

        for i in 0..num_nodes {
            row_ptr[i + 1] += row_ptr[i];
        }

        Self { num_nodes, row_ptr, col_idx }
    }

    /// Sparse Matrix-Vector Multiplication: y = v * A
    /// v is a bitset (indicator vector) represented as a Vec<bool>.
    fn spmv(&self, v: &[bool]) -> Vec<bool> {
        let mut y = vec![false; self.num_nodes];
        for (i, &active) in v.iter().enumerate() {
            if active {
                let start = self.row_ptr[i];
                let end = self.row_ptr[i + 1];
                for &neighbor in &self.col_idx[start..end] {
                    y[neighbor] = true;
                }
            }
        }
        y
    }
}

/// A reasoner that uses true Sparse Matrix-Vector multiplication for logical queries.
pub struct SparseReasoner {
    num_entities: usize,
    relation_matrices: HashMap<String, SparseMatrix>,
    entity_to_idx: HashMap<String, usize>,
    idx_to_entity: Vec<String>,
}

impl SparseReasoner {
    pub fn from_kg(kg: &KnowledgeGraph) -> Self {
        let mut entity_to_idx = HashMap::new();
        let mut idx_to_entity = Vec::new();
        
        for entity in kg.entities() {
            entity_to_idx.insert(entity.id.0.clone(), idx_to_entity.len());
            idx_to_entity.push(entity.id.0.clone());
        }

        let mut edges_by_rel: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
        for triple in kg.triples() {
            let h_idx = entity_to_idx[&triple.subject.0];
            let t_idx = entity_to_idx[&triple.object.0];
            edges_by_rel
                .entry(triple.predicate.0.clone())
                .or_default()
                .push((h_idx, t_idx));
        }

        let num_entities = kg.entity_count();
        let relation_matrices = edges_by_rel
            .into_iter()
            .map(|(rel, edges)| (rel, SparseMatrix::from_edges(num_entities, &edges)))
            .collect();

        Self {
            num_entities,
            relation_matrices,
            entity_to_idx,
            idx_to_entity,
        }
    }

    fn execute_recursive(&self, query: &LogicalQuery) -> Vec<bool> {
        match query {
            LogicalQuery::Entity(name) => {
                let mut v = vec![false; self.num_entities];
                if let Some(&idx) = self.entity_to_idx.get(name) {
                    v[idx] = true;
                }
                v
            }
            LogicalQuery::Projection(sub, rel) => {
                let v_sub = self.execute_recursive(sub);
                if let Some(matrix) = self.relation_matrices.get(rel) {
                    matrix.spmv(&v_sub)
                } else {
                    vec![false; self.num_entities]
                }
            }
            LogicalQuery::Intersection(queries) => {
                if queries.is_empty() {
                    return vec![true; self.num_entities];
                }
                let mut v_res = self.execute_recursive(&queries[0]);
                for q in &queries[1..] {
                    let v_q = self.execute_recursive(q);
                    for i in 0..self.num_entities {
                        v_res[i] &= v_q[i];
                    }
                }
                v_res
            }
            LogicalQuery::Union(queries) => {
                let mut v_res = vec![false; self.num_entities];
                for q in queries {
                    let v_q = self.execute_recursive(q);
                    for i in 0..self.num_entities {
                        v_res[i] |= v_q[i];
                    }
                }
                v_res
            }
            LogicalQuery::Negation(sub) => {
                let mut v = self.execute_recursive(sub);
                for val in v.iter_mut() {
                    *val = !*val;
                }
                v
            }
        }
    }
}

impl Reasoner for SparseReasoner {
    fn score(&self, query: &LogicalQuery, entity: &str) -> Result<f32> {
        let v = self.execute_recursive(query);
        let idx = self.entity_to_idx.get(entity)
            .ok_or_else(|| crate::ReasonError::Kg(format!("Entity not found: {}", entity)))?;
        Ok(if v[*idx] { 1.0 } else { 0.0 })
    }

    fn predict(&self, query: &LogicalQuery, k: usize) -> Result<Vec<Prediction>> {
        let v = self.execute_recursive(query);
        let mut predictions = Vec::new();
        for (idx, &active) in v.iter().enumerate() {
            if active {
                predictions.push(Prediction {
                    entity: self.idx_to_entity[idx].clone(),
                    score: 1.0,
                });
            }
        }
        predictions.truncate(k);
        Ok(predictions)
    }
}
