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
| `lattix` | **Primary crate** (preferred import); minimal by default, opt-in features for algorithms/formats |
| `lattix-core` | Implementation crate used by `lattix`; **not** intended as a direct dependency |

## Why this exists

`lattix` is meant to be the small, stable layer you can build higher-level systems on:

- **Substrate-first**: triples + graph storage + a few basic algorithms (PageRank, random walks) with predictable behavior.
- **Format boundaries**: parsing/serialization as opt-in features so dependents can stay lean.
- **Interop**: designed to be a dependency of higher-level KG/graph learning systems (see `webs/*`).

## Best starting points

- **Core types**: `KnowledgeGraph`, `Triple`
- **Basic traversal**: `KnowledgeGraph::find_path`
- **When you want ML / training / temporal systems**: start in `webs/*` (not here)

## Embedding Backends

These backends and training/inference systems live in separate crates. `lattix` stays substrate-only.

## Relationship to `webs`

`webs/*` is the home for higher-level KG systems (reasoning, training/inference, temporal systems, CLI).
It depends on this repo’s substrate (`lattix`).
