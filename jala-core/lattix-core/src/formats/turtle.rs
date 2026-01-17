//! Turtle format (RDF 1.2).
//!
//! Human-readable RDF serialization with prefix support.
//!
//! Reference: <https://www.w3.org/TR/rdf12-turtle/>

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
                    Subject::NamedNode(n) => format!("<{}>", n.iri),
                    Subject::BlankNode(n) => format!("_:{}", n.id),
                    Subject::Triple(t) => format!("{}", t),
                };

                let p_str = match triple.predicate {
                    NamedNode { iri } => format!("<{}>", iri),
                };

                let o_str = match triple.object {
                    Term::NamedNode(n) => format!("<{}>", n.iri),
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
        _prefixes: &HashMap<String, String>, // Rio handles prefixes if we register them, todo later
    ) -> Result<()> {
        let mut formatter = TurtleFormatter::new(writer);

        for triple in kg.triples() {
            let s_node = parse_term_str(triple.subject.as_str());
            let p_node = parse_named_node_str(triple.predicate.as_str());
            let o_node = parse_term_str_obj(triple.object.as_str());

            if let (Some(s), Some(p), Some(o)) = (s_node, p_node, o_node) {
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

fn parse_named_node_str(s: &str) -> Option<NamedNode<'_>> {
    if s.starts_with('<') && s.ends_with('>') {
        Some(NamedNode {
            iri: &s[1..s.len() - 1],
        })
    } else {
        None
    }
}

fn parse_term_str(s: &str) -> Option<Subject<'_>> {
    if s.starts_with('<') {
        Some(Subject::NamedNode(NamedNode {
            iri: &s[1..s.len() - 1],
        }))
    } else if s.starts_with("_:") {
        Some(Subject::BlankNode(rio_api::model::BlankNode {
            id: &s[2..],
        }))
    } else {
        None
    }
}

fn parse_term_str_obj(s: &str) -> Option<Term<'_>> {
    if s.starts_with('<') {
        Some(Term::NamedNode(NamedNode {
            iri: &s[1..s.len() - 1],
        }))
    } else if s.starts_with("_:") {
        Some(Term::BlankNode(rio_api::model::BlankNode { id: &s[2..] }))
    } else if s.starts_with('"') {
        // Simple literal fallback
        if let Some(end) = s.rfind('"') {
            if end > 0 {
                return Some(Term::Literal(rio_api::model::Literal::Simple {
                    value: &s[1..end],
                }));
            }
        }
        None
    } else {
        None
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
