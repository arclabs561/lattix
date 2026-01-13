# grafene

Knowledge graph construction, analysis, and embedding inference.

Dual-licensed under MIT or Apache-2.0.

![KG Structure](hack/viz/kg_structure.png)

```rust
use grafene_core::{KnowledgeGraph, Triple};

let mut kg = KnowledgeGraph::new();
kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));

// Find path: Apple -> founded_by -> Steve Jobs
if let Some(path) = kg.find_path("Apple", "Steve Jobs") {
    println!("Path: {} hops", path.len());
}
```

## Features

- **Core**: Efficient graph storage, PageRank, Random Walks (Node2Vec)
- **Temporal**: Time-aware graph queries (`grafene-temporal`)
- **KGE**: Knowledge Graph Embeddings (`grafene-kge`)
  - **BoxE**: Containment embeddings (via `subsume`)
  - **Hyperbolic**: Hierarchy embeddings (via `hyperball`)
  - **ONNX**: TransE, RotatE, ComplEx (via ONNX)

See [`docs/README_DETAILED.md`](docs/README_DETAILED.md) for full features and architecture.
