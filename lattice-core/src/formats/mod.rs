//! RDF serialization formats.
//!
//! Supports modern RDF 1.2 specifications (2024):
//! - N-Triples (line-based, simple)
//! - N-Quads (N-Triples with named graphs)
//! - Turtle (human-readable)
//! - JSON-LD (linked data)
//!
//! Note: RDF 1.2 introduces triple terms (quoted triples) for reification.
//! This is tracked but not yet fully implemented.

mod jsonld;
mod nquads;
mod ntriples;
mod turtle;

pub use jsonld::JsonLd;
pub use nquads::{NQuads, Quad};
pub use ntriples::NTriples;
pub use turtle::Turtle;
