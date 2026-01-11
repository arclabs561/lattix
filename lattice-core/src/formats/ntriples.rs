//! N-Triples format (RDF 1.2).
//!
//! Line-based, simple format for RDF triples.
//! Each line is: `<subject> <predicate> <object> .`
//!
//! Reference: <https://www.w3.org/TR/rdf12-n-triples/>

use crate::{KnowledgeGraph, Result, Triple};
use std::io::{BufRead, BufReader, Read, Write};

/// N-Triples format handler.
pub struct NTriples;

impl NTriples {
    /// Parse N-Triples from a reader.
    pub fn read<R: Read>(reader: R) -> Result<KnowledgeGraph> {
        let buf = BufReader::new(reader);
        let mut kg = KnowledgeGraph::new();

        for line in buf.lines() {
            let line = line?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Ok(triple) = Triple::from_ntriples(line) {
                kg.add_triple(triple);
            }
        }

        Ok(kg)
    }

    /// Write knowledge graph to N-Triples format.
    pub fn write<W: Write>(kg: &KnowledgeGraph, mut writer: W) -> Result<()> {
        for triple in kg.triples() {
            writeln!(writer, "{}", triple.to_ntriples())?;
        }
        Ok(())
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Result<KnowledgeGraph> {
        Self::read(s.as_bytes())
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
        let input = r#"
<http://example.org/Apple> <http://example.org/founded_by> <http://example.org/Steve_Jobs> .
<http://example.org/Apple> <http://example.org/type> <http://example.org/Company> .
"#;
        let kg = NTriples::from_str(input).unwrap();
        assert_eq!(kg.triple_count(), 2);

        let output = NTriples::to_string(&kg).unwrap();
        assert!(output.contains("Apple"));
        assert!(output.contains("founded_by"));
    }
}
