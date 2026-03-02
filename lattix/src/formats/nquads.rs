//! N-Quads format (RDF 1.2).
//!
//! Extension of N-Triples with named graphs.
//! Each line is: `<subject> <predicate> <object> <graph> .`
//!
//! Reference: <https://www.w3.org/TR/rdf12-n-quads/>

use super::oxrdf_helpers::{graph_name_to_string, subject_to_string, term_to_string};
use crate::{KnowledgeGraph, Result, Triple};
use oxttl::NQuadsParser;
use std::collections::HashMap;
use std::io::{Read, Write};

/// A quad (triple + graph name).
#[derive(Debug, Clone)]
pub struct Quad {
    /// The triple.
    pub triple: Triple,
    /// Named graph URI (None = default graph).
    pub graph: Option<String>,
}

impl Quad {
    /// Create a new quad.
    pub fn new(triple: Triple, graph: Option<String>) -> Self {
        Self { triple, graph }
    }

    /// Parse from N-Quads line.
    pub fn from_nquads(line: &str) -> Result<Self> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return Err(crate::Error::ParseTriple("Empty or comment".into()));
        }

        // Use oxttl to parse the single line
        let mut quads: Vec<_> = NQuadsParser::new()
            .for_reader(std::io::Cursor::new(line))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let quad = quads
            .pop()
            .ok_or_else(|| crate::Error::ParseTriple("No quad found".into()))?;

        let s = subject_to_string(&quad.subject);
        let p = quad.predicate.as_str().to_string();
        let o = term_to_string(&quad.object);
        let g = graph_name_to_string(&quad.graph_name);

        Ok(Self {
            triple: Triple::new(s, p, o),
            graph: g,
        })
    }

    /// Convert to N-Quads format.
    pub fn to_nquads(&self) -> String {
        let base = self.triple.to_ntriples();
        // Remove trailing " ."
        let base = base.trim_end_matches(" .");

        match &self.graph {
            Some(g) => format!("{} <{}> .", base, g),
            None => format!("{} .", base),
        }
    }
}

/// N-Quads format handler.
pub struct NQuads;

impl NQuads {
    /// Parse N-Quads to multiple named graphs.
    pub fn read<R: Read>(reader: R) -> Result<HashMap<Option<String>, KnowledgeGraph>> {
        let mut graphs: HashMap<Option<String>, KnowledgeGraph> = HashMap::new();

        for result in NQuadsParser::new().for_reader(reader) {
            let quad =
                result.map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let s = subject_to_string(&quad.subject);
            let p = quad.predicate.as_str().to_string();
            let o = term_to_string(&quad.object);
            let g = graph_name_to_string(&quad.graph_name);

            graphs
                .entry(g)
                .or_default()
                .add_triple(Triple::new(s, p, o));
        }

        Ok(graphs)
    }

    /// Parse to single merged graph (ignores graph names).
    pub fn read_merged<R: Read>(reader: R) -> Result<KnowledgeGraph> {
        let mut kg = KnowledgeGraph::new();

        for result in NQuadsParser::new().for_reader(reader) {
            let quad =
                result.map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let s = subject_to_string(&quad.subject);
            let p = quad.predicate.as_str().to_string();
            let o = term_to_string(&quad.object);

            kg.add_triple(Triple::new(s, p, o));
        }

        Ok(kg)
    }

    /// Write graphs to N-Quads format.
    pub fn write<W: Write>(
        graphs: &HashMap<Option<String>, KnowledgeGraph>,
        mut writer: W,
    ) -> Result<()> {
        for (graph_name, kg) in graphs {
            for triple in kg.triples() {
                let quad = Quad::new(triple.clone(), graph_name.clone());
                writeln!(writer, "{}", quad.to_nquads())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_quad() {
        let line = "<http://example.org/s> <http://example.org/p> <http://example.org/o> <http://example.org/g> .";
        let quad = Quad::from_nquads(line).unwrap();
        assert_eq!(quad.triple.subject.as_str(), "http://example.org/s");
        assert_eq!(quad.graph, Some("http://example.org/g".to_string()));
    }

    #[test]
    fn test_parse_triple_as_quad() {
        let line = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .";
        let quad = Quad::from_nquads(line).unwrap();
        assert_eq!(quad.graph, None);
    }

    #[test]
    fn test_graph_name_roundtrip_named() {
        let input = "<http://example.org/s> <http://example.org/p> <http://example.org/o> <http://example.org/g1> .\n\
                     <http://example.org/a> <http://example.org/b> <http://example.org/c> <http://example.org/g2> .\n";
        let graphs = NQuads::read(std::io::Cursor::new(input)).unwrap();
        assert!(graphs.contains_key(&Some("http://example.org/g1".to_string())));
        assert!(graphs.contains_key(&Some("http://example.org/g2".to_string())));
        assert_eq!(graphs.len(), 2);
    }

    #[test]
    fn test_graph_name_roundtrip_default() {
        let input = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n";
        let graphs = NQuads::read(std::io::Cursor::new(input)).unwrap();
        assert!(graphs.contains_key(&None));
        assert_eq!(graphs.len(), 1);
    }

    #[test]
    fn test_graph_name_roundtrip_mixed() {
        let input = "<http://example.org/s> <http://example.org/p> <http://example.org/o> <http://example.org/g> .\n\
                     <http://example.org/a> <http://example.org/b> <http://example.org/c> .\n";
        let graphs = NQuads::read(std::io::Cursor::new(input)).unwrap();
        assert!(graphs.contains_key(&Some("http://example.org/g".to_string())));
        assert!(graphs.contains_key(&None));
        assert_eq!(graphs.len(), 2);
    }

    #[test]
    fn test_read_merged_ignores_graphs() {
        let input = "<http://example.org/s> <http://example.org/p> <http://example.org/o> <http://example.org/g1> .\n\
                     <http://example.org/a> <http://example.org/b> <http://example.org/c> <http://example.org/g2> .\n";
        let kg = NQuads::read_merged(std::io::Cursor::new(input)).unwrap();
        assert_eq!(kg.triple_count(), 2);
    }

    #[test]
    fn test_write_roundtrip() {
        let input = "<http://example.org/s> <http://example.org/p> <http://example.org/o> <http://example.org/g> .\n";
        let graphs = NQuads::read(std::io::Cursor::new(input)).unwrap();

        let mut buf = Vec::new();
        NQuads::write(&graphs, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(output.contains("http://example.org/s"));
        assert!(output.contains("http://example.org/g"));
    }
}
