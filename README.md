# lattix

Knowledge graph **substrate**: core types + basic algorithms + formats.

Dual-licensed under MIT or Apache-2.0.

```rust
use lattix::{KnowledgeGraph, Triple};

let mut kg = KnowledgeGraph::new();
kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));

// Find path: Apple -> founded_by -> Steve Jobs
if let Some(path) = kg.find_path("Apple", "Steve Jobs") {
    println!("Path: {} hops", path.len());
}
```

## Crates

| Crate | Purpose |
|-------|---------|
| `lattix` | Minimal facade crate (preferred import); **minimal by default** |
| `lattix-core` | Core implementation: storage, PageRank, random walks, formats |

## Embedding Backends

These backends and training/inference systems live in `webs/*` (L5). `lattix` stays substrate-only.

## Relationship to `webs`

`webs/*` is the home for higher-level KG systems (reasoning, training/inference, temporal systems, CLI).
It depends on this repo’s L3 substrate (`lattix` / `lattix-core`).
