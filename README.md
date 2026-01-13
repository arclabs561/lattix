# lattice

Knowledge graph construction, analysis, and embedding inference in Rust.

## Features

- **Knowledge Graph**: Efficient graph storage (petgraph-backed) and querying
- **RDF Formats**: N-Triples, N-Quads, Turtle, JSON-LD, and CSV (via rio)
- **Graph Algorithms**: 
  - Random walks (Node2Vec style, parallel via rayon)
  - PageRank centrality
  - Connected components (Tarjan SCC)
  - Neighbor sampling (for GNNs)
- **KGE Inference**: TransE, RotatE, ComplEx, DistMult scoring (ONNX runtime)
- **Binary Serialization**: Fast save/load via bincode
- **CLI**: Full-featured command-line tool
- **Python Bindings**: PyO3-based Python integration

## Crate Structure

```
lattice/
├── crates/
│   ├── nexus-core/    # Core types, formats, algorithms
│   ├── nexus-embed/   # KGE inference (ONNX)
│   ├── nexus-cli/     # Command-line interface
│   ├── nexus-py/      # Python bindings (maturin)
│   └── lattice/         # Facade crate (re-exports)
```

## Installation

```toml
[dependencies]
nexus = "0.1"

# With ONNX inference
nexus-embed = { version = "0.1", features = ["onnx"] }
```

## CLI

```bash
cargo install --path crates/nexus-cli

# Statistics
lattice stats graph.nt
lattice stats graph.csv

# Format conversion
lattice convert graph.nt -o graph.ttl --format turtle

# Random walks (Node2Vec corpus for word2vec/gensim)
lattice walks graph.csv -o walks.txt --length 80 --num-walks 10 --p 1.0 --q 1.0

# PageRank
lattice pagerank graph.csv --top 20

# Connected components
lattice components graph.csv

# Binary serialization (fast reload)
lattice save graph.csv graph.bin
lattice stats graph.bin  # loads in ~10ms vs ~100ms for parsing

# Query
lattice query graph.nt --from "Alice"
lattice path graph.nt --from "Alice" --to "Charlie"
lattice entities graph.nt --limit 50
lattice relations graph.nt
```

## Library

### Knowledge Graph

```rust
use nexus_core::{KnowledgeGraph, Triple};

let mut kg = KnowledgeGraph::new();

kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));
kg.add_triple(Triple::new("Apple", "headquartered_in", "Cupertino"));
kg.add_triple(Triple::new("Steve Jobs", "born_in", "San Francisco"));

// Query
let rels = kg.relations_from("Apple");
assert_eq!(rels.len(), 2);

// Path finding
if let Some(path) = kg.find_path("Apple", "San Francisco") {
    println!("Path: {} hops", path.len());
}

// Statistics
let stats = kg.stats();
println!("{} entities, {} triples", stats.entity_count, stats.triple_count);
```

### RDF Formats

```rust
use nexus_core::formats::{NTriples, Turtle, JsonLd, Csv};

// Load
let kg = KnowledgeGraph::from_ntriples_file("graph.nt")?;
let kg = Csv::read(std::fs::File::open("graph.csv")?)?;

// Serialize
let turtle = Turtle::to_string(&kg)?;
let jsonld = JsonLd::to_string(&kg)?;
```

### Random Walks

```rust
use nexus_core::algo::random_walk::{generate_walks, RandomWalkConfig};

let config = RandomWalkConfig {
    walk_length: 80,
    num_walks: 10,  // per node
    p: 1.0,         // return parameter
    q: 1.0,         // in-out parameter
    seed: 42,
};

let walks: Vec<Vec<String>> = generate_walks(&kg, config);
// Output to file for gensim Word2Vec training
```

### PageRank

```rust
use nexus_core::algo::pagerank::{pagerank, PageRankConfig};

let scores = pagerank(&kg, PageRankConfig::default());
let mut sorted: Vec<_> = scores.into_iter().collect();
sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

for (entity, score) in sorted.iter().take(10) {
    println!("{}: {:.4}", entity, score);
}
```

### Connected Components

```rust
use nexus_core::algo::components::{connected_components, component_stats};

let components = connected_components(&kg);
let stats = component_stats(&components);
println!("{} components, largest: {} nodes", stats.num_components, stats.max_component_size);
```

### Binary Serialization

```rust
// Save (fast)
kg.to_binary_file("graph.bin")?;

// Load (very fast)
let kg = KnowledgeGraph::from_binary_file("graph.bin")?;
```

## Python Bindings

Build with maturin:

```bash
cd crates/nexus-py
maturin develop --release
```

Usage:

```python
import lattice

# Load graph
g = lattice.Graph.from_csv("graph.csv")
# g = lattice.Graph.from_ntriples("graph.nt")

print(f"Entities: {g.num_entities()}, Triples: {g.num_triples()}")

# Random walks
walks = g.random_walks(num_walks=10, walk_length=80, p=1.0, q=1.0, seed=42)

# PageRank
scores = g.pagerank(damping=0.85, max_iters=100)

# Sample neighbors (for GNN mini-batching)
neighbors = g.sample_neighbors(nodes=["Alice", "Bob"], k=5, seed=42)
```

## Performance Notes

- Random walks are parallelized via rayon
- Binary serialization is ~10x faster than parsing text formats
- For 2nd-order walks (p != 1 or q != 1), current implementation is O(d) per step
  - Future: Alias sampling for O(1) per step (like PecanPy)

## License

MIT OR Apache-2.0
