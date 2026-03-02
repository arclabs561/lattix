//! Turtle format (RDF 1.2).
//!
//! Human-readable RDF serialization with prefix support.
//!
//! Reference: <https://www.w3.org/TR/rdf12-turtle/>

use super::rio_helpers;
use crate::{KnowledgeGraph, Result, Triple};
use oxiri::Iri;
use rio_api::formatter::TriplesFormatter;
use rio_api::model::{NamedNode, Subject, Term};
use rio_api::parser::TriplesParser;
use rio_turtle::{TurtleFormatter, TurtleParser};
use std::collections::HashMap;
use std::io::{BufRead, Write};

/// Turtle format handler.
pub struct Turtle;

impl Turtle {
    /// Parse Turtle from a reader using Rio.
    pub fn read<R: BufRead>(reader: R, base_iri: Option<&str>) -> Result<KnowledgeGraph> {
        let base_iri_parsed = if let Some(base) = base_iri {
            Some(
                Iri::parse(base.to_string())
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?,
            )
        } else {
            None
        };

        let mut parser = TurtleParser::new(reader, base_iri_parsed);
        let mut kg = KnowledgeGraph::new();

        parser
            .parse_all(&mut |triple| {
                let s_str = match triple.subject {
                    Subject::NamedNode(n) => n.iri.to_string(),
                    Subject::BlankNode(n) => format!("_:{}", n.id),
                    Subject::Triple(t) => format!("{}", t),
                };

                let NamedNode { iri } = triple.predicate;
                let p_str = iri.to_string();

                let o_str = match triple.object {
                    Term::NamedNode(n) => n.iri.to_string(),
                    Term::BlankNode(n) => format!("_:{}", n.id),
                    Term::Literal(l) => format!("{}", l),
                    Term::Triple(t) => format!("{}", t),
                };

                kg.add_triple(Triple::new(s_str, p_str, o_str));
                Ok(()) as std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>
            })
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        Ok(kg)
    }

    /// Write knowledge graph to Turtle format using Rio.
    pub fn write<W: Write>(
        kg: &KnowledgeGraph,
        writer: W,
        _prefixes: &HashMap<String, String>,
    ) -> Result<()> {
        let mut formatter = TurtleFormatter::new(writer);

        for triple in kg.triples() {
            let s = rio_helpers::to_subject(triple.subject.as_str());
            let p = rio_helpers::to_named_node(triple.predicate.as_str());
            let o = rio_helpers::to_object(triple.object.as_str());

            if let (Some(s), Some(p), Some(o)) = (s, p, o) {
                formatter.format(&rio_api::model::Triple {
                    subject: s,
                    predicate: p,
                    object: o,
                })?;
            }
        }
        formatter.finish()?;
        Ok(())
    }

    /// Write with default prefixes.
    pub fn write_default<W: Write>(kg: &KnowledgeGraph, writer: W) -> Result<()> {
        let prefixes = default_prefixes();
        Self::write(kg, writer, &prefixes)
    }

    /// Convert to string.
    pub fn to_string(kg: &KnowledgeGraph) -> Result<String> {
        let mut buf = Vec::new();
        Self::write_default(kg, &mut buf)?;
        Ok(String::from_utf8_lossy(&buf).to_string())
    }
}

/// Default prefixes for common vocabularies.
pub fn default_prefixes() -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert(
        "rdf".into(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#".into(),
    );
    m.insert(
        "rdfs".into(),
        "http://www.w3.org/2000/01/rdf-schema#".into(),
    );
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn test_turtle_output() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new(
            "<http://example.org/Apple>",
            "<http://example.org/founded_by>",
            "<http://example.org/Steve_Jobs>",
        ));

        let output = Turtle::to_string(&kg).unwrap();
        assert!(output.contains("Apple"));
    }
}
