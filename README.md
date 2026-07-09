# lattix

[![crates.io](https://img.shields.io/crates/v/lattix.svg)](https://crates.io/crates/lattix)
[![Documentation](https://docs.rs/lattix/badge.svg)](https://docs.rs/lattix)

Knowledge graph data structures.

```toml
[dependencies]
lattix = "0.8.0"
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

- **Triples**: `(subject, predicate, object)`, the atomic unit of knowledge
- **Homogeneous graphs**: `KnowledgeGraph` wraps petgraph for traversal, path-finding, and centrality
- **Heterogeneous graphs**: `HeteroGraph` with typed nodes and edges (for RGCN, HGT, link prediction)
- **Hypergraphs**: `HyperTriple` (qualifier-based, Wikidata-style) and `HyperEdge` (role-based n-ary relations)
- **Algorithms**: PageRank, HITS, degree/betweenness/closeness/eigenvector/Katz centrality, random walks, connected components, neighbor sampling
- **Formats**: N-Triples, N-Quads, Turtle, JSON-LD, CSV (enabled by default via `formats`)

## Features

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `formats` | yes | RDF and CSV readers/writers (N-Triples, Turtle, N-Quads, JSON-LD, CSV) |
| `algo` | yes | Centrality algorithms, PageRank, random walks, sampling (pulls in `rand`, `rayon`) |
| `binary` | no | Postcard serialization |
| `sophia` | no | Sophia RDF framework integration |

Use `cargo add lattix --no-default-features` for core graph types without the
`formats` or `algo` modules.

RDF support maps RDF terms onto lattix's string triple model. IRIs and blank
nodes round trip directly; literals are stored in their N-Triples lexical form.
RDF 1.2 triple terms are not part of the public graph model.

## Starting points

- **Core types**: `KnowledgeGraph`, `Triple`
- **Typed graphs**: `HeteroGraph`
- **N-ary relations**: `HyperTriple`, `HyperEdge`
- **Algorithms**: `algo::pagerank`, `algo::centrality`
- **Examples**: `cargo run --example pagerank_demo`, `cargo run --example triples`

## License

MIT OR Apache-2.0
