//! N-Triples format (RDF 1.2).
//!
//! Line-based, simple format for RDF triples.
//! Each line is: `<subject> <predicate> <object> .`
//!
//! Reference: <https://www.w3.org/TR/rdf12-n-triples/>

use super::oxrdf_helpers::{subject_to_string, term_to_string};
use crate::{KnowledgeGraph, Result, Triple};
use oxttl::NTriplesParser;
use std::io::{Read, Write};

/// Write a lattix string as an N-Triples subject/predicate term (IRI or blank node).
fn write_nt_subject(w: &mut impl Write, s: &str) -> std::io::Result<()> {
    if s.starts_with("_:") {
        w.write_all(s.as_bytes())
    } else {
        w.write_all(b"<")?;
        w.write_all(s.as_bytes())?;
        w.write_all(b">")
    }
}

/// Write a lattix string as an N-Triples object term (IRI, blank node, or literal).
fn write_nt_object(w: &mut impl Write, s: &str) -> std::io::Result<()> {
    if s.starts_with("_:") || s.starts_with('"') {
        w.write_all(s.as_bytes())
    } else {
        w.write_all(b"<")?;
        w.write_all(s.as_bytes())?;
        w.write_all(b">")
    }
}

/// N-Triples format handler.
///
/// Reads and writes the line-based N-Triples RDF serialization.
/// Each line encodes one triple as `<subject> <predicate> <object> .`
///
/// Parsing uses [`oxttl::NTriplesParser`] under the hood.
pub struct NTriples;

impl NTriples {
    /// Parse N-Triples from a reader.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> lattix::Result<()> {
    /// use lattix::formats::NTriples;
    ///
    /// let input = "<http://ex.org/A> <http://ex.org/knows> <http://ex.org/B> .\n";
    /// let kg = NTriples::read(std::io::Cursor::new(input))?;
    /// assert_eq!(kg.triple_count(), 1);
    /// # Ok(())
    /// # }
    /// ```
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
    /// Each triple is written as one line: `<s> <p> <o> .`
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> lattix::Result<()> {
    /// use lattix::{KnowledgeGraph, Triple, formats::NTriples};
    ///
    /// let mut kg = KnowledgeGraph::new();
    /// kg.add_triple(Triple::new("http://ex.org/A", "http://ex.org/knows", "http://ex.org/B"));
    ///
    /// let mut buf = Vec::new();
    /// NTriples::write(&kg, &mut buf)?;
    /// let output = String::from_utf8(buf).unwrap();
    /// assert!(output.contains("<http://ex.org/A>"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn write<W: Write>(kg: &KnowledgeGraph, writer: W) -> Result<()> {
        let mut writer = std::io::BufWriter::new(writer);
        for triple in kg.triples() {
            write_nt_subject(&mut writer, triple.subject.as_str())?;
            writer.write_all(b" <")?;
            writer.write_all(triple.predicate.as_str().as_bytes())?;
            writer.write_all(b"> ")?;
            write_nt_object(&mut writer, triple.object.as_str())?;
            writer.write_all(b" .\n")?;
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
        let input = "<http://example.org/s> <http://example.org/p> \"hello\"@en .\n";
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
