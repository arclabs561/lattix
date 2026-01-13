# grafene

Graph + knowledge graph workspace for Rust (formats, algorithms, embeddings, CLI).

## Structure

This repo is a workspace. See `crates/` for the individual crates.

## CLI

The `grafene-cli` crate builds the `grafene` binary:

```bash
cargo install --path crates/grafene-cli

grafene --help
```

## Library

```toml
[dependencies]
grafene = "0.1"
```

## License

See `LICENSE`.
