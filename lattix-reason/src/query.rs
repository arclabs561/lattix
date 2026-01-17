use serde::{Deserialize, Serialize};

/// A multi-hop logical query on a knowledge graph.
///
/// These queries combine entities and relations using logical operators
/// like projection, intersection (AND), union (OR), and negation (NOT).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalQuery {
    /// Base case: a specific entity anchor (e.g., "Einstein").
    Entity(String),
    /// Relation projection (q, r) -> {t | ∃h ∈ q: (h, r, t)}.
    Projection(Box<LogicalQuery>, String),
    /// Intersection of multiple queries (AND).
    Intersection(Vec<LogicalQuery>),
    /// Union of multiple queries (OR).
    Union(Vec<LogicalQuery>),
    /// Negation of a query (NOT).
    Negation(Box<LogicalQuery>),
}

/// Gated intersection module for boxes.
///
/// Stealing the sigmoid-gating trick from KGReasoning to handle box intersections stably.
#[derive(Debug, Clone)]
pub struct BoxGatedIntersection {
    pub dim: usize,
}

impl BoxGatedIntersection {
    pub fn intersect(&self, centers: &[Vec<f32>], offsets: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
        let n = centers.len();
        if n == 0 {
            return (vec![0.0; self.dim], vec![0.0; self.dim]);
        }

        let mut new_c = vec![0.0; self.dim];
        let mut new_o = vec![0.0; self.dim];

        // 1. Center intersection: mean center
        for c in centers {
            for i in 0..self.dim {
                new_c[i] += c[i] / n as f32;
            }
        }

        // 2. Offset intersection: sigmoid gating trick
        for i in 0..self.dim {
            let mut min_o = f32::MAX;
            let mut sum_o = 0.0;
            for o in offsets {
                min_o = min_o.min(o[i]);
                sum_o += o[i];
            }
            let gate = 1.0 / (1.0 + (-sum_o).exp());
            new_o[i] = min_o * gate;
        }

        (new_c, new_o)
    }
}

impl LogicalQuery {
    /// Convert the query to Disjunctive Normal Form (DNF).
    ///
    /// DNF represents a query as a union of intersections:
    /// (q1 AND q2) OR (q3 AND q4) OR ...
    pub fn to_dnf(&self) -> Vec<Vec<LogicalQuery>> {
        match self {
            LogicalQuery::Entity(name) => vec![vec![LogicalQuery::Entity(name.clone())]],
            LogicalQuery::Projection(sub, rel) => {
                let sub_dnf = sub.to_dnf();
                sub_dnf
                    .into_iter()
                    .map(|conjunction| {
                        conjunction
                            .into_iter()
                            .map(|q| q.project(rel.clone()))
                            .collect()
                    })
                    .collect()
            }
            LogicalQuery::Intersection(queries) => {
                let mut dnf_queries: Vec<Vec<Vec<LogicalQuery>>> =
                    queries.iter().map(|q| q.to_dnf()).collect();
                if dnf_queries.is_empty() {
                    return vec![vec![]];
                }
                let mut result = dnf_queries.remove(0);
                for next_dnf in dnf_queries {
                    let mut next_result = Vec::new();
                    for r in &result {
                        for n in &next_dnf {
                            let mut combined = r.clone();
                            combined.extend(n.clone());
                            next_result.push(combined);
                        }
                    }
                    result = next_result;
                }
                result
            }
            LogicalQuery::Union(queries) => {
                let mut result = Vec::new();
                for q in queries {
                    result.extend(q.to_dnf());
                }
                result
            }
            LogicalQuery::Negation(_) => vec![vec![self.clone()]],
        }
    }

    pub fn entity(name: impl Into<String>) -> Self {
        Self::Entity(name.into())
    }

    pub fn project(self, relation: impl Into<String>) -> Self {
        Self::Projection(Box::new(self), relation.into())
    }

    pub fn and(queries: Vec<LogicalQuery>) -> Self {
        Self::Intersection(queries)
    }

    pub fn or(queries: Vec<LogicalQuery>) -> Self {
        Self::Union(queries)
    }

    pub fn not(self) -> Self {
        Self::Negation(Box::new(self))
    }
}
