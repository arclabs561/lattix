//! N-Triples format (RDF 1.2).
//!
//! Line-based, simple format for RDF triples.
//! Each line is: `<subject> <predicate> <object> .`
//!
//! Reference: <https://www.w3.org/TR/rdf12-n-triples/>

use crate::{KnowledgeGraph, Result, Triple};
use oxttl::NTriplesParser;
use std::io::{Read, Write};

/// Convert an oxrdf subject to a lattix string.
fn subject_to_string(s: &oxrdf::NamedOrBlankNode) -> String {
    match s {
        oxrdf::NamedOrBlankNode::NamedNode(n) => n.as_str().to_string(),
        oxrdf::NamedOrBlankNode::BlankNode(b) => format!("_:{}", b.as_str()),
    }
}

/// Convert an oxrdf term to a lattix string.
fn term_to_string(t: &oxrdf::Term) -> String {
    match t {
        oxrdf::Term::NamedNode(n) => n.as_str().to_string(),
        oxrdf::Term::BlankNode(b) => format!("_:{}", b.as_str()),
        oxrdf::Term::Literal(l) => {
            if let Some(lang) = l.language() {
                format!("\"{}\"@{}", l.value(), lang)
            } else {
                let dt = l.datatype().as_str();
                if dt == "http://www.w3.org/2001/XMLSchema#string" {
                    format!("\"{}\"", l.value())
                } else {
                    format!("\"{}\"^^<{}>", l.value(), dt)
                }
            }
        }
    }
}

/// Convert a lattix string to N-Triples term syntax for serialization.
fn to_nt_subject(s: &str) -> String {
    if let Some(id) = s.strip_prefix("_:") {
        format!("_:{}", id)
    } else {
        format!("<{}>", s)
    }
}

fn to_nt_object(s: &str) -> String {
    if s.starts_with("_:") || s.starts_with('"') {
        s.to_string()
    } else {
        format!("<{}>", s)
    }
}

/// N-Triples format handler.
pub struct NTriples;

impl NTriples {
    /// Parse N-Triples from a reader.
    pub fn read<R: Read>(reader: R) -> Result<KnowledgeGraph> {
        let mut kg = KnowledgeGraph::new();

        for result in NTriplesParser::new().for_reader(reader) {
            let triple =
                result.map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let s = subject_to_string(&triple.subject);
            let p = triple.predicate.as_str().to_string();
            let o = term_to_string(&triple.object);

            kg.add_triple(Triple::new(s, p, o));
        }

        Ok(kg)
    }

    /// Write knowledge graph to N-Triples format.
    ///
    /// Returns an error if any triple cannot be converted to valid RDF terms
    /// (e.g., a literal in subject position).
    pub fn write<W: Write>(kg: &KnowledgeGraph, mut writer: W) -> Result<()> {
        for triple in kg.triples() {
            let s = to_nt_subject(triple.subject.as_str());
            let p = format!("<{}>", triple.predicate.as_str());
            let o = to_nt_object(triple.object.as_str());
            writeln!(writer, "{} {} {} .", s, p, o)?;
        }
        Ok(())
    }

    /// Parse from string.
    pub fn parse(s: &str) -> Result<KnowledgeGraph> {
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
        let kg = NTriples::parse(input).unwrap();
        assert_eq!(kg.triple_count(), 2);

        let output = NTriples::to_string(&kg).unwrap();
        assert!(output.contains("Apple"));
        assert!(output.contains("founded_by"));
    }

    #[test]
    fn test_blank_nodes() {
        let input = "<http://example.org/s> <http://example.org/p> _:b0 .\n";
        let kg = NTriples::parse(input).unwrap();
        assert_eq!(kg.triple_count(), 1);
        let triple = kg.triples().next().unwrap();
        assert_eq!(triple.object.as_str(), "_:b0");
    }

    #[test]
    fn test_literals() {
        let input =
            "<http://example.org/s> <http://example.org/p> \"hello\"@en .\n";
        let kg = NTriples::parse(input).unwrap();
        let triple = kg.triples().next().unwrap();
        assert_eq!(triple.object.as_str(), "\"hello\"@en");
    }

    #[test]
    fn test_typed_literal() {
        let input = "<http://example.org/s> <http://example.org/p> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n";
        let kg = NTriples::parse(input).unwrap();
        let triple = kg.triples().next().unwrap();
        assert_eq!(
            triple.object.as_str(),
            "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"
        );
    }
}
