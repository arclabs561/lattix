# Research Findings: Rust RDF & Graph Ecosystem

## Sophia (`sophia_rs`)
**URL:** https://github.com/pchampin/sophia_rs
**Description:** A comprehensive Rust toolkit for RDF and Linked Data.
**Key Features:**
- **Trait-based architecture:** `Graph`, `Dataset`, `Term` traits allow for multiple backends (in-memory, disk-based, etc.) to share common algorithms.
- **Generalized RDF:** Supports RDF-star (nested triples) and potentially other variants.
- **Interoperability:** Acts as a standard interface for Rust RDF libraries.
- **Layout:** Flat workspace (`api`, `inmem`, `jsonld`, `turtle`, etc. at root).
- **Documentation:** Uses `mdBook` in `book/` directory.

**Relevance to `lattice`:**
- `grafene::KnowledgeGraph` is currently a specific implementation (petgraph-backed).
- Implementing `sophia::graph::Graph` for `grafene::KnowledgeGraph` would allow `lattice` to plug into the wider ecosystem (serializers, reasoners, etc.).
- **Recommendation:** Study `sophia` traits and consider implementing them in a `lattice-sophia` crate or feature.

## Rio
**URL:** https://github.com/oxigraph/rio
**Description:** Low-level, fast, W3C-compliant RDF parsers.
**Key Features:**
- **Speed:** Designed for performance.
- **Streaming:** Event-based parsing (SAX-like) for N-Triples, Turtle, etc.
- **Safety:** Strict W3C compliance.
- **Layout:** Flat workspace (`api`, `turtle`, `xml` at root).

**Relevance to `lattice`:**
- `grafene-core` currently implements a manual N-Triples parser.
- Switching to `rio_turtle` and `rio_xml` would provide robust, fast parsing for all RDF formats immediately.
- **Recommendation:** Replace custom parsing logic in `grafene-core` with `rio` parsers. (Done)

## Node2Vec-rs
**URL:** https://github.com/GregorLueg/node2vec-rs
**Description:** Node2Vec implementation using the `burn` deep learning framework.
**Key Features:**
- **Burn Integration:** Uses `burn` for the neural network part (SkipGram).
- **End-to-End in Rust:** Does both walking and training in Rust.
- **Layout:** Single crate (`src/`, `tests/` at root).

**Relevance to `lattice`:**
- Demonstrates how to do the training phase in Rust.
- Currently, `lattice` generates walks (Rust) -> Python (Gensim) for training.
- **Recommendation:** Keep as a reference for future "pure Rust" training feature, but `gensim` is likely more mature/optimized for Word2Vec specifically right now.

## GraphRAG-rs
**URL:** https://github.com/automataIA/graphrag-rs
**Description:** Graph Retrieval-Augmented Generation in Rust.
**Relevance:**
- Similar domain to `rank-rank` ecosystem.
- Worth reviewing for their approach to graph construction from text and retrieval strategies.
- **Layout:** Prefixed flat workspace (`graphrag-cli`, `graphrag-core`, etc.).

## Repository Layout Analysis
- **Flat vs Nested:** `sophia` and `rio` use flat layouts (crates at root). `lattice` uses `crates/` directory. `crates/` is cleaner for keeping non-code artifacts (scripts, docs, viz) separated from code.
- **Workspace Dependencies:** Standard practice to centralize version management in root `Cargo.toml`. `lattice` follows this.
- **Documentation:** `sophia`'s use of `book/` is excellent for complex ecosystems. `lattice` currently uses `README.md` but should consider `mdBook` if complexity grows.

## Deep Dive: Random Walk Performance
- **PecanPy**: Uses specialized sparse matrix formats (CSR) and Numba/C++ to optimize 2nd-order random walks. It precomputes transition probabilities (Alias method) to achieve O(1) sampling per step, but at O(V*deg) memory cost.
- **Lattice**: Uses `petgraph` (Adjacency List) and `rayon` (parallelism).
    - **Current Bottleneck**: `sample_biased` checks `contains_edge` which is O(degree). For dense graphs, this makes 2nd-order walks O(degree^2) per step.
    - **Optimization Path**:
        1.  **HashSet Cache**: Cache neighbors of the previous node to make `contains_edge` O(1) or O(log d).
        2.  **CSR**: Switch to `petgraph::csr::Csr` or a custom CSR implementation for static graphs (like PecanPy).
        3.  **Alias Sampling**: Implement Alias method for O(1) sampling if memory allows.

## Actionable Plan
1.  **Refactor Parsing:** Use `rio` in `grafene-core` for N-Triples/Turtle parsing. (Completed)
2.  **Standardize Interfaces:** Investigate implementing `sophia` traits. (Future)
3.  **Performance:** `lattice`'s parallel random walks (`rayon`) is a good start; verify performance against `pecanpy` benchmarks (if possible) or just ensure it saturates cores.
4.  **Structure:** Maintain `crates/` layout for cleanliness. Ensure all dependencies are managed via workspace.
