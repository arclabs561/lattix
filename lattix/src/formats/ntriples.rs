//! N-Triples format (RDF 1.2).
//!
//! Line-based, simple format for RDF triples.
//! Each line is: `<subject> <predicate> <object> .`
//!
//! Reference: <https://www.w3.org/TR/rdf12-n-triples/>

use super::rio_helpers;
use crate::{KnowledgeGraph, Result, Triple};
use rio_api::formatter::TriplesFormatter;
use rio_api::model::{NamedNode, Subject, Term};
use rio_api::parser::TriplesParser;
use rio_turtle::{NTriplesFormatter, NTriplesParser};
use std::io::{BufRead, Write};

/// N-Triples format handler.
pub struct NTriples;

impl NTriples {
    /// Parse N-Triples from a reader using Rio.
    pub fn read<R: BufRead>(reader: R) -> Result<KnowledgeGraph> {
        let mut parser = NTriplesParser::new(reader);
        let mut kg = KnowledgeGraph::new();

        parser
            .parse_all(&mut |triple| {
                let s_str = match triple.subject {
                    Subject::NamedNode(n) => n.iri.to_string(),
                    Subject::BlankNode(n) => format!("_:{}", n.id),
                    Subject::Triple(t) => format!("{}", t),
                };

                let p_str = match triple.predicate {
                    NamedNode { iri } => iri.to_string(),
                };

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

    /// Write knowledge graph to N-Triples format using Rio.
    ///
    /// Returns an error if any triple cannot be converted to valid RDF terms
    /// (e.g., a literal in subject position).
    pub fn write<W: Write>(kg: &KnowledgeGraph, writer: W) -> Result<()> {
        let mut formatter = NTriplesFormatter::new(writer);

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
            } else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Could not serialize triple to N-Triples: {:?}", triple),
                )
                .into());
            }
        }
        formatter.finish()?;
        Ok(())
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Result<KnowledgeGraph> {
        Self::read(std::io::Cursor::new(s))
    }

    /// Convert to string.
    pub fn to_string(kg: &KnowledgeGraph) -> Result<String> {
        let mut buf = Vec::new();
        Self::write(kg, &mut buf)?;
        Ok(String::from_utf8_lossy(&buf).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let input = r#"<http://example.org/Apple> <http://example.org/founded_by> <http://example.org/Steve_Jobs> .
<http://example.org/Apple> <http://example.org/type> <http://example.org/Company> .
"#;
        let kg = NTriples::from_str(input).unwrap();
        assert_eq!(kg.triple_count(), 2);

        let output = NTriples::to_string(&kg).unwrap();
        assert!(output.contains("Apple"));
        assert!(output.contains("founded_by"));
    }
}
