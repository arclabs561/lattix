---
status: proposal
date: 2026-07-09
scope: critique follow-up after RDF, graph semantics, dependency, and API fixes
review-trigger: before publishing the next lattix crate release
---

# Critique Follow-Up Roadmap

This is the tracked version of the follow-up plan after the July 2026 critique
batch. It separates release-blocking cleanup from architectural forks that need
a recorded decision before more code lands.

## Evidence

- Rust SemVer needs automation as a release gate, but human classification is
  still required for caller-visible API changes.
  Source: <https://predr.ag/blog/semver-in-rust-tooling-breakage-and-edge-cases/>
- RDF 1.2 graph terms may be IRIs, blank nodes, literals, or triple terms. That
  makes lattix's string triple model an RDF compatibility boundary, not a full
  RDF model.
  Source: <https://www.w3.org/TR/rdf12-concepts/>
- W3C publishes N-Triples and Turtle conformance suites.
  Sources: <https://www.w3.org/TR/rdf12-n-triples/> and
  <https://w3c.github.io/rdf-turtle/spec/>
- Rio is unmaintained and directs users to `oxrdfio`, `oxttl`, and related
  Oxigraph crates.
  Sources: <https://github.com/oxigraph/rio> and
  <https://crates.io/crates/oxrdfio>
- `cargo-deny` covers dependency advisories, license policy, banned crates, and
  source policy.
  Source: <https://github.com/EmbarkStudios/cargo-deny>

## Completed In This Patch

- Contributor docs now name the real cargo gates and current feature flags.
- CI, publish validation, and publish workflows run the same local gate shape.
- `cargo-deny` policy is part of the repo.
- RDF conformance-focused regression tests cover language tags, typed literals,
  local IRI decoding, Turtle base IRIs, and N-Quads graph names.
- N-Quads local predicate and graph-name handling matches the N-Triples/Turtle
  local IRI convention.
- The RDF term-model and parallel-edge semantics decisions are recorded in
  separate docs.

## Remaining Before Publish

1. Install or select Rust 1.87 and run the MSRV gate locally.
2. Run the full release gate from `CONTRIBUTING.md`.
3. Classify the release as `0.8.0` because centrality map keys, query execution,
   and binary serialization changed.
4. Move `CHANGELOG.md` entries from `[Unreleased]` to the versioned release
   section immediately before publishing.
5. Run `cargo publish -p lattix --dry-run` on the versioned patch.
6. Publish only after dry-run, semver, policy, and docs.rs expectations are
   verified.

## Deferred Forks

- Do not implement typed RDF terms until the design in
  `docs/rdf-term-model.md` is accepted.
- Do not add weighted parallel-edge algorithms until a consumer needs them and
  small multigraph golden tests exist.
- Do not add CSR or alias sampling until benchmarks show the current random-walk
  implementation misses a concrete target.
