//! RDF serialization formats.
//!
//! Provides read/write support for common RDF and tabular formats,
//! backed by [`oxttl`](https://docs.rs/oxttl) for N-Triples and Turtle parsing.
//!
//! | Format | Read | Write | Backend |
//! |--------|------|-------|---------|
//! | N-Triples | yes | yes | oxttl |
//! | N-Quads | yes | yes | hand-rolled |
//! | Turtle | yes | yes | oxttl |
//! | JSON-LD | yes | yes | serde_json |
//! | CSV | yes | -- | csv crate |
//!
//! RDF 1.2 introduces triple terms (quoted triples) for reification.
//! This is tracked but not yet fully implemented.

pub mod csv;
mod jsonld;
mod nquads;
mod ntriples;
mod oxrdf_helpers;
mod turtle;

pub use self::csv::Csv;
pub use jsonld::JsonLd;
pub use nquads::{NQuads, Quad};
pub use ntriples::NTriples;
pub use turtle::Turtle;
