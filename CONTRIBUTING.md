# Contributing to lattix

Thanks for your interest. lattix is a knowledge-graph library: triples, homogeneous and heterogeneous graphs over petgraph, hypergraphs, and the standard algorithms (PageRank, HITS, centrality, walks).

## Before you start

For non-trivial work (new graph types, algorithm additions, format changes), open an issue first to align on scope. Drive-by bug fixes and doc patches don't need an issue.

## Setup

- Rust toolchain: stable, MSRV `1.87`. Use `rustup` to manage.
- Optional: `cargo-nextest` for faster test runs.
- Optional: `cargo-semver-checks` and `cargo-deny` for release gates.

## Style

- Direct, lowercase prose in commits. No marketing words ("powerful", "robust", "elegant"). No em-dashes in prose.
- Commit messages: `lattix: short lowercase description`. One commit per logical change.
- `cargo fmt` and `cargo clippy --workspace --all-targets --all-features -- -D warnings` must pass before `git add`.

## Testing

- `cargo test --workspace --all-features` runs the full feature suite.
- `cargo test --workspace` runs the default feature suite.
- `cargo check --workspace --no-default-features --lib` verifies the minimal build.
- New graph algorithms need at least one correctness test against a known small graph plus a property test (idempotency, monotonicity, normalization where applicable).
- Format parsers (N-Triples, Turtle, N-Quads, JSON-LD, CSV) need round-trip tests on the W3C test corpus or equivalent.

## Feature flags

`Cargo.toml` defines `formats` (RDF and CSV formats, default), `algo` (centrality + walks, default), `binary` (postcard serialization), `kge` (benchmark data and metrics), and `sophia` (sophia_api integration). The minimal footprint (`--no-default-features`) gives the core graph types without `formats` or `algo`.

## Release checks

Before publishing, run:

```sh
cargo fmt --check --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test --workspace
cargo check --workspace --no-default-features --lib
cargo package --manifest-path lattix/Cargo.toml
cargo semver-checks check-release --manifest-path lattix/Cargo.toml --all-features
cargo deny check
```

Also run the MSRV check with Rust 1.87:

```sh
cargo +1.87.0 check --workspace --all-features
```

## Pull requests

- Keep PRs scoped to one concern.
- Show before/after for behavior changes.
- Link the related issue.
- CI must be green before requesting review.

## License

Dual-licensed under MIT or Apache-2.0 at your option. By contributing you agree your contributions are licensed under both.
