# Research Notes: Rust RDF And Graph Crates

## Sophia (`sophia_rs`)
**URL:** https://github.com/pchampin/sophia_rs
**Description:** Rust toolkit for RDF and Linked Data.
**Notes:**
- **Trait-based architecture:** `Graph`, `Dataset`, `Term` traits allow for multiple backends (in-memory, disk-based, etc.) to share common algorithms.
- **Generalized RDF:** Supports RDF-star (nested triples) and potentially other variants.
- **Interoperability:** Acts as a standard interface for Rust RDF libraries.
- **Layout:** Flat workspace (`api`, `inmem`, `jsonld`, `turtle`, etc. at root).
- **Documentation:** Uses `mdBook` in `book/` directory.

**Relevance to `lattix`:**
- `lattix::KnowledgeGraph` is currently a specific implementation (petgraph-backed).
- Implementing `sophia::graph::Graph` for `lattix::KnowledgeGraph` would allow `lattix` to use Sophia serializers and reasoners.
- **Recommendation:** Study `sophia` traits and consider implementing them behind a `sophia` feature.

## Rio
**URL:** https://github.com/oxigraph/rio
**Description:** W3C-compliant RDF parsers.
**Notes:**
- **Performance:** Parser-focused implementation.
- **Streaming:** Event-based parsing (SAX-like) for N-Triples, Turtle, etc.
- **Safety:** Strict W3C compliance.
- **Layout:** Flat workspace (`api`, `turtle`, `xml` at root).

**Relevance to `lattix`:**
- `lattix-core` currently implements a manual N-Triples parser.
- Switching to `rio_turtle` and `rio_xml` would add RDF format coverage without the manual parser.
- **Recommendation:** Replace custom parsing logic in `lattix-core` with `rio` parsers. (Done)

## Node2Vec-rs
**URL:** https://github.com/GregorLueg/node2vec-rs
**Description:** Node2Vec implementation using the `burn` deep learning framework.
**Notes:**
- **Burn Integration:** Uses `burn` for the neural network part (SkipGram).
- **End-to-End in Rust:** Does both walking and training in Rust.
- **Layout:** Single crate (`src/`, `tests/` at root).

**Relevance to `lattix`:**
- Demonstrates how to do the training phase in Rust.
- Currently, `lattix` generates walks in Rust and can hand them to Python tooling for training.
- **Recommendation:** Keep as a reference for future Rust training work; compare against `gensim` before replacing that path.

## GraphRAG-rs
**URL:** https://github.com/automataIA/graphrag-rs
**Description:** Graph Retrieval-Augmented Generation in Rust.
**Relevance:**
- Similar domain: graph construction from text and graph-backed retrieval.
- Worth reviewing for their approach to graph construction from text and retrieval strategies.
- **Layout:** Prefixed flat workspace (`graphrag-cli`, `graphrag-core`, etc.).

## Repository Layout Analysis
- **Flat vs Nested:** `sophia` and `rio` use flat layouts (crates at root). `lattix` uses a nested crate directory.
- **Workspace Dependencies:** `lattix` centralizes version management in the root `Cargo.toml`.
- **Documentation:** `sophia`'s `book/` directory is a useful reference if `lattix` needs more than README-level documentation.

## Random Walk Performance
- **PecanPy**: Uses specialized sparse matrix formats (CSR) and Numba/C++ to optimize 2nd-order random walks. It precomputes transition probabilities (Alias method) to achieve O(1) sampling per step, but at O(V*deg) memory cost.
- **lattix**: Uses `petgraph` (adjacency list) and `rayon` (parallelism).
    - **Current Bottleneck**: `sample_biased` checks `contains_edge` which is O(degree). For dense graphs, this makes 2nd-order walks O(degree^2) per step.
    - **Optimization Path**:
        1.  **HashSet Cache**: Cache neighbors of the previous node to make `contains_edge` O(1) or O(log d).
        2.  **CSR**: Switch to `petgraph::csr::Csr` or a custom CSR implementation for static graphs (like PecanPy).
        3.  **Alias Sampling**: Implement Alias method for O(1) sampling if memory allows.

## Actionable Plan
1.  **Refactor Parsing:** Use `rio` in `lattix-core` for N-Triples/Turtle parsing. (Completed)
2.  **Standardize Interfaces:** Investigate implementing `sophia` traits. (Future)
3.  **Performance:** Verify `lattix` parallel random walks against `pecanpy` benchmarks if comparable inputs are available.
4.  **Structure:** Maintain `crates/` layout for cleanliness. Ensure all dependencies are managed via workspace.
