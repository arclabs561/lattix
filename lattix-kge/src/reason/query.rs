use serde::{Deserialize, Serialize};

/// A multi-hop logical query on a knowledge graph.
///
/// These queries combine entities and relations using logical operators
/// like projection, intersection (AND), and union (OR).
///
/// Based on:
/// - Ren et al. (2020): "Query2box: Reasoning over Knowledge Graphs with Geometric Projection and Intersection"
/// - Ren & Leskovec (2020): "Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs"
///
/// # Logic Pattern (DNF)
///
/// Following the Snap-Stanford KGReasoning standard, we represent queries
/// that can involve:
/// - 1p: (e, (r,))
/// - 2p: (e, (r, r))
/// - 2i: ((e, (r,)), (e, (r,)))
/// - 3i: ((e, (r,)), (e, (r,)), (e, (r,)))
/// - etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalQuery {
    /// Base case: a specific entity anchor (e.g., "Einstein").
    Entity(String),
    /// Relation projection (q, r) -> {t | ∃h ∈ q: (h, r, t)}.
    /// e.g., "Where was [Einstein] born?"
    Projection(Box<LogicalQuery>, String),
    /// Intersection of multiple queries (AND).
    /// e.g., "Scientists who were [born in Germany] AND [won Nobel Prize]."
    Intersection(Vec<LogicalQuery>),
    /// Union of multiple queries (OR).
    /// e.g., "People who are [British] OR [Canadian]."
    Union(Vec<LogicalQuery>),
    /// Negation of a query (NOT).
    /// e.g., "Scientists who [did NOT win Nobel Prize]."
    Negation(Box<LogicalQuery>),
}

/// Gated intersection operator for Box embeddings.
#[derive(Debug, Clone)]
pub struct BoxGatedIntersection {
    /// Dimension of the boxes.
    pub dim: usize,
}

impl BoxGatedIntersection {
    /// Compute the intersection of multiple boxes.
    pub fn intersect(&self, centers: &[Vec<f32>], offsets: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
        if centers.is_empty() {
            return (vec![0.0; self.dim], vec![0.0; self.dim]);
        }

        let n = centers.len() as f32;
        let mut new_center = vec![0.0; self.dim];
        let mut new_offset = vec![0.0; self.dim];

        for i in 0..self.dim {
            let mut c_sum = 0.0;
            let mut o_min = f32::MAX;

            for j in 0..centers.len() {
                c_sum += centers[j][i];
                o_min = o_min.min(offsets[j][i].abs());
            }

            new_center[i] = c_sum / n;
            new_offset[i] = o_min; // Simplified: min offset
        }

        (new_center, new_offset)
    }
}

impl LogicalQuery {
    /// Create a new entity anchor query.
    pub fn entity(name: impl Into<String>) -> Self {
        Self::Entity(name.into())
    }

    /// Create a projection query from this query through a relation.
    pub fn project(self, relation: impl Into<String>) -> Self {
        Self::Projection(Box::new(self), relation.into())
    }

    /// Create an intersection of multiple queries.
    pub fn and(queries: Vec<LogicalQuery>) -> Self {
        Self::Intersection(queries)
    }

    /// Create a union of multiple queries.
    pub fn or(queries: Vec<LogicalQuery>) -> Self {
        Self::Union(queries)
    }

    /// Create a negation of this query.
    pub fn not(self) -> Self {
        Self::Negation(Box::new(self))
    }
}
