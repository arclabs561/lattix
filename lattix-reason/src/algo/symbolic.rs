use crate::{LogicalQuery, Reasoner, Result};
use lattix_core::KnowledgeGraph;
use lattix_kge::Prediction;
use std::collections::HashSet;

/// A reasoner that uses symbolic path finding on the knowledge graph.
///
/// This is a "hard logic" reasoner that only returns entities
/// with valid structural paths matching the query.
pub struct SymbolicReasoner<'a> {
    kg: &'a KnowledgeGraph,
}

impl<'a> SymbolicReasoner<'a> {
    pub fn new(kg: &'a KnowledgeGraph) -> Self {
        Self { kg }
    }

    fn execute_symbolic(&self, query: &LogicalQuery) -> HashSet<String> {
        match query {
            LogicalQuery::Entity(name) => {
                let mut set = HashSet::new();
                set.insert(name.clone());
                set
            }
            LogicalQuery::Projection(subquery, relation) => {
                let heads = self.execute_symbolic(subquery);
                let mut tails = HashSet::new();
                for head in heads {
                    let triples = self.kg.relations_from(head);
                    for t in triples {
                        if t.predicate.as_str() == relation {
                            tails.insert(t.object.as_str().to_string());
                        }
                    }
                }
                tails
            }
            LogicalQuery::Intersection(queries) => {
                if queries.is_empty() {
                    return HashSet::new();
                }
                let mut results: Vec<HashSet<String>> = queries
                    .iter()
                    .map(|q| self.execute_symbolic(q))
                    .collect();
                
                let mut intersection = results.pop().unwrap_or_default();
                for res in results {
                    intersection.retain(|e| res.contains(e));
                }
                intersection
            }
            LogicalQuery::Union(queries) => {
                let mut union = HashSet::new();
                for q in queries {
                    union.extend(self.execute_symbolic(q));
                }
                union
            }
            LogicalQuery::Negation(_) => {
                // Hard to implement symbolic negation without a universe set
                HashSet::new()
            }
        }
    }
}

impl<'a> Reasoner for SymbolicReasoner<'a> {
    fn score(&self, query: &LogicalQuery, entity: &str) -> Result<f32> {
        let results = self.execute_symbolic(query);
        if results.contains(entity) {
            Ok(1.0)
        } else {
            Ok(0.0)
        }
    }

    fn predict(&self, query: &LogicalQuery, _k: usize) -> Result<Vec<Prediction>> {
        let results = self.execute_symbolic(query);
        Ok(results
            .into_iter()
            .map(|entity| Prediction { entity, score: 1.0 })
            .collect())
    }
}
