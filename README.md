# lattix

[![crates.io](https://img.shields.io/crates/v/lattix.svg)](https://crates.io/crates/lattix)
[![Documentation](https://docs.rs/lattix/badge.svg)](https://docs.rs/lattix)
[![CI](https://github.com/arclabs561/lattix/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/lattix/actions/workflows/ci.yml)

Knowledge graph types and algorithms: triples, homogeneous
graphs over [petgraph](https://crates.io/crates/petgraph) for
traversal/centrality, and heterogeneous graphs with typed nodes/edges
for RGCN, HGT, and link prediction.

Dual-licensed under MIT or Apache-2.0.

```toml
[dependencies]
lattix = "0.5.4"
```

```rust
use lattix::{KnowledgeGraph, Triple};

let mut kg = KnowledgeGraph::new();
kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));
kg.add_triple(Triple::new("Apple", "headquartered_in", "Cupertino"));
kg.add_triple(Triple::new("Steve Jobs", "born_in", "San Francisco"));

// Query relations
let apple_rels = kg.relations_from("Apple");
assert_eq!(apple_rels.len(), 2);

// Find path: Apple -> founded_by -> Steve Jobs
if let Some(path) = kg.find_path("Apple", "Steve Jobs") {
    println!("Path: {} hops", path.len());
}
```

## What it does

`lattix` provides the types and algorithms for working with knowledge graphs:

- **Triples**: `(subject, predicate, object)` -- the atomic unit of knowledge
- **Homogeneous graphs**: `KnowledgeGraph` wraps petgraph for traversal, path-finding, and centrality
- **Heterogeneous graphs**: `HeteroGraph` with typed nodes and edges (for RGCN, HGT, link prediction)
- **Hypergraphs**: `HyperTriple` (qualifier-based, Wikidata-style) and `HyperEdge` (role-based n-ary relations)
- **Algorithms**: PageRank, HITS, degree/betweenness/closeness/eigenvector/Katz centrality, random walks, connected components, neighbor sampling
- **Formats**: N-Triples, N-Quads, Turtle, JSON-LD, CSV (all opt-in via `formats` feature)

## Features

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `formats` | yes | RDF parsing/serialization (N-Triples, Turtle, N-Quads, JSON-LD, CSV) |
| `algo` | yes | Centrality algorithms, PageRank, random walks, sampling (pulls in `rand`, `rayon`) |
| `binary` | no | Bincode serialization |
| `sophia` | no | Sophia RDF framework integration |

Minimal dependency footprint: `cargo add lattix --no-default-features` gives you just core types + petgraph + serde.

## Starting points

- **Core types**: [`KnowledgeGraph`](https://docs.rs/lattix/latest/lattix/struct.KnowledgeGraph.html), [`Triple`](https://docs.rs/lattix/latest/lattix/struct.Triple.html)
- **Typed graphs**: [`HeteroGraph`](https://docs.rs/lattix/latest/lattix/hetero/struct.HeteroGraph.html)
- **N-ary relations**: [`HyperTriple`](https://docs.rs/lattix/latest/lattix/hyper/struct.HyperTriple.html), [`HyperEdge`](https://docs.rs/lattix/latest/lattix/hyper/struct.HyperEdge.html)
- **Algorithms**: [`algo::pagerank`](https://docs.rs/lattix/latest/lattix/algo/pagerank/index.html), [`algo::centrality`](https://docs.rs/lattix/latest/lattix/algo/centrality/index.html)
- **Examples**: `cargo run --example pagerank_demo`, `cargo run --example triples`

## License

MIT OR Apache-2.0
