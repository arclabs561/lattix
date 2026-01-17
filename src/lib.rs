//! `lattix` is the small, dependency-light graph/KG substrate.
//!
//! Design goal: keep a minimal, stable surface for knowledge graph construction
//! and simple graph algorithms (PageRank, random walks) over RDF-style triples.

use std::collections::{HashMap, HashSet};

/// A single RDF-style triple: (subject, predicate, object).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Triple {
    pub fn new(subject: impl Into<String>, predicate: impl Into<String>, object: impl Into<String>) -> Self {
        Self { subject: subject.into(), predicate: predicate.into(), object: object.into() }
    }
}

/// Minimal stats about a knowledge graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnowledgeGraphStats {
    pub entity_count: usize,
    pub triple_count: usize,
    pub num_triples: usize, // alias for older call sites
}

/// In-memory knowledge graph built from triples.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeGraph {
    triples: Vec<Triple>,
    // adjacency is derived lazily (cheap for small graphs, keeps construction simple)
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self { triples: Vec::new() }
    }

    pub fn add_triple(&mut self, triple: Triple) {
        self.triples.push(triple);
    }

    pub fn triples(&self) -> &[Triple] {
        &self.triples
    }

    pub fn stats(&self) -> KnowledgeGraphStats {
        let mut entities: HashSet<&str> = HashSet::new();
        for t in &self.triples {
            entities.insert(t.subject.as_str());
            entities.insert(t.object.as_str());
        }
        KnowledgeGraphStats {
            entity_count: entities.len(),
            triple_count: self.triples.len(),
            num_triples: self.triples.len(),
        }
    }

    /// Parse an N-Triples file (very small, tolerant parser).
    ///
    /// This is intentionally conservative: we accept lines of the form:
    /// `<s> <p> <o> .` or `s p o .` and strip surrounding `<...>` / quotes.
    pub fn from_ntriples_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| Error::Io(e.to_string()))?;
        let mut kg = KnowledgeGraph::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // N-Triples terminator
            let line = line.strip_suffix('.').unwrap_or(line).trim();
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                continue;
            }
            let s = clean_term(parts[0]);
            let p = clean_term(parts[1]);
            let o = clean_term(parts[2]);
            kg.add_triple(Triple::new(s, p, o));
        }
        Ok(kg)
    }

    /// Outgoing relations from a subject.
    pub fn relations_from(&self, subject: &str) -> Vec<&Triple> {
        self.triples
            .iter()
            .filter(|t| t.subject == subject)
            .collect()
    }
}

fn clean_term(raw: &str) -> String {
    let raw = raw.trim();
    let raw = raw.strip_prefix('<').unwrap_or(raw);
    let raw = raw.strip_suffix('>').unwrap_or(raw);
    let raw = raw.strip_prefix('"').unwrap_or(raw);
    let raw = raw.strip_suffix('"').unwrap_or(raw);
    raw.to_string()
}

// -----------------------------------------------------------------------------
// Errors / Result
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Error {
    Io(String),
}

pub type Result<T> = std::result::Result<T, Error>;

// -----------------------------------------------------------------------------
// Compatibility placeholders (kept small, expanded when needed)
// -----------------------------------------------------------------------------

pub type EntityId = String;
pub type RelationType = String;
pub type NodeType = String;
pub type EdgeType = String;

#[derive(Debug, Clone)]
pub struct Entity {
    pub id: EntityId,
}

#[derive(Debug, Clone)]
pub struct Relation {
    pub ty: RelationType,
}

#[derive(Debug, Clone, Default)]
pub struct NodeStore;

#[derive(Debug, Clone, Default)]
pub struct EdgeStore;

#[derive(Debug, Clone, Default)]
pub struct HeteroGraph;

#[derive(Debug, Clone, Copy, Default)]
pub struct HeteroGraphStats;

// -----------------------------------------------------------------------------
// Re-exports
// -----------------------------------------------------------------------------

pub use petgraph;

/// RDF serialization formats (minimal for now).
pub mod formats {
    // Placeholder for future (RDF 1.2) format support.
}

/// Graph algorithms (PageRank, random walks).
pub mod algo {
    use super::{KnowledgeGraph, Triple};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashMap;

    pub mod pagerank {
        use super::*;

        #[derive(Debug, Clone)]
        pub struct PageRankConfig {
            pub damping_factor: f64,
            pub max_iterations: usize,
            pub tol: f64,
        }

        impl Default for PageRankConfig {
            fn default() -> Self {
                Self { damping_factor: 0.85, max_iterations: 50, tol: 1e-9 }
            }
        }

        /// PageRank over the directed graph induced by subject -> object edges.
        pub fn pagerank(kg: &KnowledgeGraph, cfg: PageRankConfig) -> HashMap<String, f64> {
            let (nodes, out_edges) = build_graph(kg.triples());
            let n = nodes.len().max(1);
            let mut rank: HashMap<String, f64> = nodes.iter().map(|k| (k.clone(), 1.0 / n as f64)).collect();

            for _ in 0..cfg.max_iterations {
                let mut next = HashMap::with_capacity(rank.len());
                for node in &nodes {
                    next.insert(node.clone(), (1.0 - cfg.damping_factor) / n as f64);
                }

                for (src, outs) in &out_edges {
                    let src_rank = *rank.get(src).unwrap_or(&0.0);
                    if outs.is_empty() {
                        continue;
                    }
                    let share = cfg.damping_factor * src_rank / outs.len() as f64;
                    for dst in outs {
                        *next.entry(dst.clone()).or_insert(0.0) += share;
                    }
                }

                let mut delta = 0.0;
                for k in &nodes {
                    delta += (next.get(k).unwrap() - rank.get(k).unwrap()).abs();
                }
                rank = next;
                if delta < cfg.tol {
                    break;
                }
            }

            rank
        }

        fn build_graph(triples: &[Triple]) -> (Vec<String>, HashMap<String, Vec<String>>) {
            let mut nodes_set = std::collections::HashSet::<String>::new();
            let mut out: HashMap<String, Vec<String>> = HashMap::new();
            for t in triples {
                nodes_set.insert(t.subject.clone());
                nodes_set.insert(t.object.clone());
                out.entry(t.subject.clone()).or_default().push(t.object.clone());
            }
            let mut nodes: Vec<String> = nodes_set.into_iter().collect();
            nodes.sort();
            (nodes, out)
        }
    }

    pub mod random_walk {
        use super::*;

        #[derive(Debug, Clone)]
        pub struct RandomWalkConfig {
            pub walk_length: usize,
            pub num_walks: usize,
            pub p: f32,
            pub q: f32,
            pub seed: u64,
        }

        impl Default for RandomWalkConfig {
            fn default() -> Self {
                Self { walk_length: 10, num_walks: 10, p: 1.0, q: 1.0, seed: 42 }
            }
        }

        /// Generate simple random walks over the subject->object adjacency.
        ///
        /// Note: `p` and `q` are accepted for Node2Vec compatibility but are not yet
        /// used to bias the walk (kept intentionally minimal until we need it).
        pub fn generate_walks(kg: &KnowledgeGraph, cfg: RandomWalkConfig) -> Vec<Vec<String>> {
            let adj = adjacency(kg.triples());
            let nodes: Vec<String> = adj.keys().cloned().collect();
            if nodes.is_empty() {
                return Vec::new();
            }

            let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
            let mut walks = Vec::new();

            for _ in 0..cfg.num_walks {
                let start = nodes[rng.gen_range(0..nodes.len())].clone();
                let mut walk = vec![start.clone()];
                let mut cur = start;

                for _ in 0..cfg.walk_length.saturating_sub(1) {
                    let Some(neigh) = adj.get(&cur) else { break };
                    if neigh.is_empty() {
                        break;
                    }
                    let next = neigh[rng.gen_range(0..neigh.len())].clone();
                    walk.push(next.clone());
                    cur = next;
                }

                walks.push(walk);
            }

            walks
        }

        fn adjacency(triples: &[Triple]) -> HashMap<String, Vec<String>> {
            let mut adj: HashMap<String, Vec<String>> = HashMap::new();
            for t in triples {
                adj.entry(t.subject.clone()).or_default().push(t.object.clone());
                // Ensure object appears as a node (even if no outgoing edges)
                adj.entry(t.object.clone()).or_default();
            }
            adj
        }
    }
}

