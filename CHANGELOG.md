# Changelog

## [Unreleased]

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
