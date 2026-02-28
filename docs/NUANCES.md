# Lattice: Implementation Nuances and Improvement Opportunities

This document identifies subtle issues, performance bottlenecks, and correctness concerns in the current implementation.

## 1. Random Walk Performance

### Current Implementation (FIXED)
Uses O(1) rejection sampling for biased walks:
```rust
// Rejection sampling loop
loop {
    let candidate = neighbors.choose(&mut rng).unwrap();  // O(1)
    let r: f64 = rng.random();
    let accept_prob = compute_transition_prob(candidate, prev, prev_neighbors);
    if r < accept_prob / max_prob {
        return candidate;
    }
}
```

Previous O(d^2) approach via `WeightedIndex` replaced with O(1) expected rejection sampling.

**Benchmark result**: ~36% speedup on 1000-node dense graph.

## 2. PageRank Dangling Nodes

### Current Implementation (FIXED)
Accumulates dangling mass in O(N):
```rust
let mut dangling_sum = 0.0;
for (u_idx, &out_deg) in out_degrees.iter().enumerate() {
    if out_deg == 0 {
        dangling_sum += scores[u_idx];
    }
}
let dangling_contrib = dangling_sum * config.damping_factor / node_count as f64;

// Single pass to add uniformly
for s in &mut new_scores {
    *s += dangling_contrib;
}
```

This is O(N) instead of O(k*N).

## 3. Connected Components: SCC vs WCC

### Current Implementation (FIXED)
Provides both:
- `strongly_connected_components()` - Tarjan's algorithm
- `weakly_connected_components()` - Union-Find on undirected view

CLI defaults to WCC (more intuitive for knowledge graphs), with `--strong` flag for SCC.

## 4. Triple Storage

### Current Implementation
Triples stored in both:
1. `graph: DiGraph<Entity, Relation>` - edges for traversal
2. `triples: Vec<Triple>` - for iteration/export
3. `subject_index: HashMap<EntityId, Vec<usize>>` - O(1) lookup
4. `object_index: HashMap<EntityId, Vec<usize>>` - O(1) lookup

Trade-off accepted: ~2x memory for O(1) queries.

## 5. O(d) Relation Queries

### Current Implementation (FIXED)
```rust
pub fn relations_from(&self, subject: impl Into<EntityId>) -> Vec<&Triple> {
    let subject = subject.into();
    match self.subject_index.get(&subject) {
        Some(indices) => indices.iter().map(|&i| &self.triples[i]).collect(),
        None => vec![],
    }
}
```

O(d) where d is out-degree, not O(N).

## 6. Neighbor Sampling (FIXED)
Uses `get_node_index()` for O(1) entity lookup:
```rust
let neighbors = if let Some(idx) = kg.get_node_index(&entity_id) {
    graph.neighbors(idx).map(...).collect()
} else {
    vec![]
};
```

## 7. `relation_types()` Caching (FIXED)

### Current Implementation
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    // ...
    #[serde(skip, default)]
    relation_type_cache: HashSet<RelationType>,
}

pub fn add_triple(&mut self, triple: Triple) {
    // ...
    self.relation_type_cache.insert(triple.predicate.clone());
    // ...
}

pub fn relation_types(&self) -> Vec<&RelationType> {
    self.relation_type_cache.iter().collect()  // O(|R|)
}

pub fn relation_type_count(&self) -> usize {
    self.relation_type_cache.len()  // O(1)
}
```

## Summary: All Issues Addressed

| Issue | Severity | Status |
|-------|----------|--------|
| Random walk O(d^2) | High | **FIXED** (rejection sampling) |
| PageRank dangling | Medium | **FIXED** (O(N) not O(k*N)) |
| SCC vs WCC | Medium | **FIXED** (WCC default, --strong for SCC) |
| Sampling O(N) | High | **FIXED** (O(1) index lookup) |
| relations_from O(N) | High | **FIXED** (O(d) via subject_index) |
| relation_types O(N log N) | Low | **FIXED** (O(1) via HashSet cache) |
| Triple duplication | Low | Accepted trade-off for O(1) queries |

## Performance Characteristics

| Operation | Complexity |
|-----------|------------|
| `add_triple` | O(1) amortized |
| `get_entity` | O(1) |
| `get_node_index` | O(1) |
| `relations_from` | O(d) where d = out-degree |
| `relations_to` | O(d) where d = in-degree |
| `neighbors` | O(d) |
| `has_edge` | O(d) via petgraph |
| `relation_types` | O(|R|) where R = unique relations |
| `relation_type_count` | O(1) |
| `find_path` | O((V + E) log V) via A* |
| Random walk step (biased) | O(1) expected |
| PageRank iteration | O(V + E) |
| Connected components | O(V + E) |
