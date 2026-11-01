# lattix

Knowledge graph data structures.

`lattix` provides three graph representations (`KnowledgeGraph`, `HeteroGraph`, `HyperGraph`),
centrality/community algorithms, neighbor sampling for GNN training, and RDF format support.

## Usage

```toml
[dependencies]
lattix = "0.8.2"
```

Default features include `formats` (N-Triples, Turtle, N-Quads, JSON-LD, CSV) and
`algo` (centrality, PageRank, random walks, sampling, label propagation).

```rust
use lattix::{Triple, KnowledgeGraph};

let mut kg = KnowledgeGraph::new();
kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));
kg.add_triple(Triple::new("Apple", "headquartered_in", "Cupertino"));
kg.add_triple(Triple::new("Steve Jobs", "born_in", "San Francisco"));

let apple_relations = kg.relations_from("Apple");
assert_eq!(apple_relations.len(), 2);
```

## Graph types

| Type | Purpose | Index structure |
|------|---------|-----------------|
| `KnowledgeGraph` | Homogeneous triple graph | petgraph + subject/object/predicate indexes |
| `HeteroGraph` | Typed nodes and edges (PyG-style) | COO + forward/reverse adjacency per edge type |
| `HyperGraph` | N-ary relations | Qualified triples (Wikidata-style) + hyperedges |

`HeteroGraph` and `HyperGraph` both convert to/from `KnowledgeGraph`.

## Features

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `formats` | yes | N-Triples, Turtle, N-Quads and JSON-LD readers/writers; CSV reader |
| `algo` | yes | Centrality, PageRank, PPR, random walks, components, sampling, label propagation |
| `kge` | no | KGE benchmark data: dataset loading, string interning, filtered eval metrics |
| `binary` | no | postcard serialization (`to_binary_file` / `from_binary_file`) |
| `sophia` | no | sophia_api 0.10 trait bridge (`Graph`, `MutableGraph`, `CollectibleGraph`) |

Disable defaults for core graph types without the `formats` or `algo` modules:

```toml
lattix = { version = "0.8.2", default-features = false }
```

For KGE benchmark pipelines (dataset loading, MRR/Hits@k evaluation):

```toml
lattix = { version = "0.8.2", default-features = false, features = ["kge"] }
```

## Algorithms

**Centrality** (7 algorithms): degree, betweenness, closeness, eigenvector, Katz, PageRank, HITS.

**Other**: personalized PageRank (PPR), Node2Vec-style random walks, connected components,
neighbor sampling (homogeneous and heterogeneous), label propagation community detection.

Sampling and random walks are deterministic under a given seed.

Examples:

```sh
cargo run --example triples
cargo run --example pagerank_demo
cargo run --example ppr_retrieval
```

## Formats

Reads and writes N-Triples, Turtle, N-Quads, and JSON-LD. CSV import is read-only.
N-Triples parsing uses [oxttl](https://crates.io/crates/oxttl), including
`Triple::from_ntriples`.
The W3C RDF 1.1 N-Triples syntax tests run in `tests/w3c_ntriples.rs`.

RDF support is a compatibility layer over lattix triples. IRIs and blank nodes
round trip directly; literals are stored in their N-Triples lexical form. RDF
1.2 triple terms are not part of the public graph model.

## Dependencies

`petgraph` is the graph backbone and is re-exported (`lattix::petgraph`) for advanced use.
Algorithm dependencies (`rand`, `rayon`, `graphops`) are optional behind the `algo` feature.
CSV support is optional behind `formats`; RDF parsing/writing uses `oxttl` and `oxrdf`.

## License

MIT OR Apache-2.0
