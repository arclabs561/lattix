# lattix

Knowledge graph substrate: core types, algorithms, and serialization formats.

`lattix` provides three graph representations (`KnowledgeGraph`, `HeteroGraph`, `HyperGraph`),
centrality/community algorithms, neighbor sampling for GNN training, and RDF format support.

## Usage

```toml
[dependencies]
lattix = "0.3.1"
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
| `formats` | yes | N-Triples, Turtle, N-Quads, JSON-LD, CSV (via oxttl/oxrdf) |
| `algo` | yes | Centrality, PageRank, PPR, random walks, components, sampling, label propagation |
| `binary` | no | bincode serialization (`to_binary_file` / `from_binary_file`) |
| `sophia` | no | sophia_api 0.8 trait bridge (`Graph`, `MutableGraph`, `CollectibleGraph`) |

Disable defaults for a minimal build (just types + serde):

```toml
lattix = { version = "0.3.1", default-features = false }
```

## Algorithms

**Centrality** (7 algorithms): degree, betweenness, closeness, eigenvector, Katz, PageRank, HITS.

**Other**: personalized PageRank (PPR), Node2Vec-style random walks, connected components,
neighbor sampling (homogeneous and heterogeneous), label propagation community detection.

All iterative algorithms have convergence controls (`max_iterations`, `tolerance`).
Sampling is deterministic under a given seed.

## Formats

Reads and writes N-Triples, Turtle, N-Quads, and JSON-LD. CSV import is read-only.
Parsing uses [oxttl](https://crates.io/crates/oxttl); the hand-rolled `Triple::from_ntriples`
parser is always available (no feature gate).

## Dependencies

`petgraph` is the graph backbone and is re-exported (`lattix::petgraph`) for advanced use.
Algorithm dependencies (`rand`, `rayon`, `graphops`) are optional behind the `algo` feature.
Format dependencies (`oxttl`, `oxrdf`, `csv`) are optional behind `formats`.

## License

MIT OR Apache-2.0
