# Lattice Python Bindings

High-performance graph algorithms for Python, powered by Rust.

## Usage

```python
import lattice

# Load Graph
kg = lattice.Graph.from_ntriples("graph.nt")
# or
kg = lattice.Graph.from_csv("edges.csv")

print(f"Entities: {kg.num_entities()}")

# Random Walks (Node2Vec)
walks = kg.random_walks(num_walks=10, walk_length=80, p=1.0, q=1.0, seed=42)

# PageRank
scores = kg.pagerank(damping=0.85, max_iters=100)

# Neighbor Sampling (for GNNs)
# Sample 10 neighbors for a list of nodes
samples = kg.sample_neighbors(["http://node1", "http://node2"], k=10, seed=42)
```

## Build

```bash
uv pip install maturin
maturin develop
```
