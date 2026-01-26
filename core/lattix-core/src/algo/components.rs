//! Connected components analysis.
//!
//! Provides both:
//! - **Strongly Connected Components (SCC)**: Every node reachable from every other within component
//! - **Weakly Connected Components (WCC)**: Connected if treating edges as undirected
//!
//! For most knowledge graph applications, WCC is typically more useful.

use crate::KnowledgeGraph;
use petgraph::algo::tarjan_scc;
use petgraph::visit::{EdgeRef, IntoNodeIdentifiers, NodeIndexable};
use std::cmp::Ordering;
use std::collections::HashMap;

// Union-Find helper functions (at module level per clippy)
fn uf_find(parent: &mut [usize], i: usize) -> usize {
    if parent[i] != i {
        parent[i] = uf_find(parent, parent[i]); // Path compression
    }
    parent[i]
}

fn uf_union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
    let px = uf_find(parent, x);
    let py = uf_find(parent, y);
    if px == py {
        return;
    }
    // Union by rank
    match rank[px].cmp(&rank[py]) {
        Ordering::Less => parent[px] = py,
        Ordering::Greater => parent[py] = px,
        Ordering::Equal => {
            parent[py] = px;
            rank[px] += 1;
        }
    }
}

/// Compute strongly connected components.
///
/// In an SCC, every node is reachable from every other node following edge directions.
/// For a chain A -> B -> C, each node is its own SCC.
#[must_use]
pub fn strongly_connected_components(kg: &KnowledgeGraph) -> Vec<Vec<String>> {
    let graph = kg.as_petgraph();
    let sccs = tarjan_scc(graph);

    sccs.into_iter()
        .map(|component| {
            component
                .into_iter()
                .map(|idx| graph[idx].id.0.clone())
                .collect()
        })
        .collect()
}

/// Compute weakly connected components.
///
/// In a WCC, nodes are connected if there's a path ignoring edge direction.
/// For a chain A -> B -> C, all nodes are in the same WCC.
///
/// Uses Union-Find for efficient O(V + E * alpha(V)) computation.
#[must_use]
pub fn weakly_connected_components(kg: &KnowledgeGraph) -> Vec<Vec<String>> {
    let graph = kg.as_petgraph();
    let n = graph.node_count();
    if n == 0 {
        return vec![];
    }

    // Union-Find
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];

    // Process all edges (treating as undirected)
    for edge in graph.edge_references() {
        let src = graph.to_index(edge.source());
        let dst = graph.to_index(edge.target());
        uf_union(&mut parent, &mut rank, src, dst);
    }

    // Group nodes by component root
    let mut components: HashMap<usize, Vec<String>> = HashMap::new();
    for node in graph.node_identifiers() {
        let idx = graph.to_index(node);
        let root = uf_find(&mut parent, idx);
        components
            .entry(root)
            .or_default()
            .push(graph[node].id.0.clone());
    }

    components.into_values().collect()
}

/// Compute connected components (weakly connected by default).
///
/// This is an alias for [`weakly_connected_components`] as WCC is more commonly
/// desired for knowledge graph analysis.
#[must_use]
pub fn connected_components(kg: &KnowledgeGraph) -> Vec<Vec<String>> {
    weakly_connected_components(kg)
}

/// Statistics about connected components.
#[derive(Debug, Clone)]
pub struct ComponentStats {
    /// Number of components.
    pub num_components: usize,
    /// Size of the largest component.
    pub max_component_size: usize,
    /// Size of the smallest component.
    pub min_component_size: usize,
    /// Average component size.
    pub avg_component_size: f64,
    /// Fraction of nodes in the largest component.
    pub largest_component_fraction: f64,
}

/// Compute statistics from components.
///
/// Returns default stats if components is empty.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn component_stats(components: &[Vec<String>]) -> ComponentStats {
    if components.is_empty() {
        return ComponentStats {
            num_components: 0,
            max_component_size: 0,
            min_component_size: 0,
            avg_component_size: 0.0,
            largest_component_fraction: 0.0,
        };
    }

    let sizes: Vec<usize> = components.iter().map(Vec::len).collect();
    let total: usize = sizes.iter().sum();
    let max_size = sizes.iter().copied().max().unwrap_or(0);
    let min_size = sizes.iter().copied().min().unwrap_or(0);

    ComponentStats {
        num_components: components.len(),
        max_component_size: max_size,
        min_component_size: min_size,
        avg_component_size: total as f64 / components.len() as f64,
        largest_component_fraction: if total > 0 {
            max_size as f64 / total as f64
        } else {
            0.0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn test_wcc_chain() {
        let mut kg = KnowledgeGraph::new();
        // A -> B -> C (chain)
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));

        let wcc = weakly_connected_components(&kg);
        assert_eq!(wcc.len(), 1, "Chain should be 1 WCC");
        assert_eq!(wcc[0].len(), 3);
    }

    #[test]
    fn test_scc_chain() {
        let mut kg = KnowledgeGraph::new();
        // A -> B -> C (chain, not strongly connected)
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));

        let scc = strongly_connected_components(&kg);
        assert_eq!(scc.len(), 3, "Chain should be 3 SCCs (one per node)");
    }

    #[test]
    fn test_scc_cycle() {
        let mut kg = KnowledgeGraph::new();
        // A -> B -> C -> A (cycle)
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));

        let scc = strongly_connected_components(&kg);
        assert_eq!(scc.len(), 1, "Cycle should be 1 SCC");
        assert_eq!(scc[0].len(), 3);
    }

    #[test]
    fn test_disconnected_wcc() {
        let mut kg = KnowledgeGraph::new();
        // Two disconnected components
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("X", "rel", "Y"));

        let wcc = weakly_connected_components(&kg);
        assert_eq!(wcc.len(), 2, "Should have 2 WCCs");
    }

    #[test]
    fn test_component_stats() {
        let components = vec![vec!["A".into(), "B".into(), "C".into()], vec!["X".into()]];
        let stats = component_stats(&components);

        assert_eq!(stats.num_components, 2);
        assert_eq!(stats.max_component_size, 3);
        assert_eq!(stats.min_component_size, 1);
        assert!((stats.avg_component_size - 2.0).abs() < 1e-6);
        assert!((stats.largest_component_fraction - 0.75).abs() < 1e-6);
    }
}
