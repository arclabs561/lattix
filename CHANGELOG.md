# Changelog

## [Unreleased]

### Fixed
- Reading a literal whose datatype is a local (non-IRI) ID now decodes the
  `urn:lattix:local:` datatype, so `"5"^^<int>` survives an N-Triples round
  trip instead of growing a new encoding layer on every write and read.

## [0.8.2] - 2026-10-13

### Added
- Added checked PageRank, personalized PageRank, random-walk and rank-metric
  entry points. PageRank results expose iteration counts and convergence.
- Added W3C N-Triples positive/negative syntax fixtures and parser round-trip
  tests, retaining upstream provenance and bundled BSD license notices.

### Changed
- Reused deduplicated adjacency views across graph algorithms and precomputed
  adjacency for iterative centrality calculations.
- Routed PageRank and personalized PageRank iterations through `graphops`
  0.5.1 while retaining lattix's parameter validation and zero-iteration behavior.
- Rejected invalid PageRank damping/tolerance and node2vec bias parameters;
  existing infallible computations panic on invalid parameters, while checked
  entry points return errors.

### Fixed
- Corrected normalized undirected betweenness and classic disconnected-graph
  closeness scaling; documented outward distances for directed closeness.
- Stabilized node2vec sampling for extreme finite bias parameters and removed
  unbounded rejection sampling.
- Validated rank/score metric inputs and avoided integer overflow in mean rank.
- Quoted Cypher property keys containing spaces, backticks or literal Unicode
  escape text while preserving simple ASCII key spelling. Backslash escaping
  follows Neo4j's identifier preprocessing; unsupported property values remain
  omitted.

## [0.8.1] - 2026-07-10

### Fixed
- Replaced the removed nightly `doc_auto_cfg` feature with `doc_cfg` so docs.rs
  can build the crate documentation.

## [0.8.0] - 2026-07-09

### Added
- `triples`, `pagerank_demo`, and `ppr_retrieval` examples matching the README starting points.
- Regression tests for RDF escaping, parser consistency, parallel-edge centrality,
  hyper-triple reification, metapath start types, path reconstruction, and lazy queries.
- RDF conformance-focused tests for language tags, typed literals, local IRI decoding,
  Turtle base IRIs, and N-Quads graph names.
- `cargo-deny` supply-chain policy.

### Changed
- Replaced the hand-written N-Triples parser with the `oxttl` parser path used by the
  format readers.
- Switched binary serialization from `bincode` to `postcard`.
- Upgraded `petgraph` to 0.8.3 and `sophia_api` to 0.10.0.
- Changed centrality and PageRank score maps to use `EntityId` keys instead of `String`.
- Changed `TripleQuery::execute` to return an iterator instead of allocating a `Vec`.
- Treat parallel triples as one topological neighbor in centrality, PageRank, PPR, HITS,
  Katz, eigenvector, closeness, betweenness, and shortest-path traversal.
- Changed `KnowledgeGraph::find_path` from zero-heuristic A* to directed BFS.

### Fixed
- Escaped RDF literal lexical forms correctly when writing N-Triples, Turtle, and Sophia
  terms.
- Preserved non-IRI lattix IDs during strict RDF output with a reversible local IRI
  encoding.
- Emitted `rdf:subject`, `rdf:predicate`, and `rdf:object` anchors for hyper-triple
  qualifier statement nodes.
- Enforced the `start_type` argument in `HeteroGraph::metapath_neighbors`.
- Used Turtle prefix declarations and deterministic subject/predicate ordering in the
  Turtle writer.
- Decoded local IRI predicates and graph names consistently when reading N-Quads.
- Encoded local graph names when writing N-Quads.

## [0.7.1] - 2026-06-10

### Changed
- README and CONTRIBUTING polish; publish gated on cargo-semver-checks. No code changes.

Earlier releases predate this changelog; see git history.

[Unreleased]: https://github.com/arclabs561/lattix/compare/v0.8.2...HEAD
[0.8.2]: https://github.com/arclabs561/lattix/compare/v0.8.1...v0.8.2
