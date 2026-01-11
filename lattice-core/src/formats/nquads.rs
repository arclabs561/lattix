//! N-Quads format (RDF 1.2).
//!
//! Extension of N-Triples with named graphs.
//! Each line is: `<subject> <predicate> <object> <graph> .`
//!
//! Reference: <https://www.w3.org/TR/rdf12-n-quads/>

use crate::{KnowledgeGraph, Result, Triple};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};

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

        // Parse: <s> <p> <o> [<g>] .
        let parts = parse_nquads_parts(line)?;

        let triple = Triple::new(
            parts.subject,
            parts.predicate,
            parts.object,
        );

        Ok(Self {
            triple,
            graph: parts.graph,
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

struct NQuadsParts {
    subject: String,
    predicate: String,
    object: String,
    graph: Option<String>,
}

fn parse_nquads_parts(line: &str) -> Result<NQuadsParts> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_uri = false;
    let mut in_literal = false;
    let mut escape_next = false;

    for c in line.chars() {
        if escape_next {
            current.push(c);
            escape_next = false;
            continue;
        }

        match c {
            '\\' => {
                escape_next = true;
                current.push(c);
            }
            '<' if !in_literal => {
                in_uri = true;
            }
            '>' if in_uri && !in_literal => {
                in_uri = false;
                parts.push(current.clone());
                current.clear();
            }
            '"' if !in_uri => {
                in_literal = !in_literal;
                current.push(c);
            }
            '.' if !in_uri && !in_literal && current.is_empty() => {
                break;
            }
            _ if in_uri || in_literal => {
                current.push(c);
            }
            _ => {}
        }
    }

    if parts.len() < 3 {
        return Err(crate::Error::InvalidNTriples(format!(
            "Expected at least 3 parts: {}",
            line
        )));
    }

    Ok(NQuadsParts {
        subject: parts[0].clone(),
        predicate: parts[1].clone(),
        object: parts[2].clone(),
        graph: parts.get(3).cloned(),
    })
}

/// N-Quads format handler.
pub struct NQuads;

impl NQuads {
    /// Parse N-Quads to multiple named graphs.
    pub fn read<R: Read>(reader: R) -> Result<HashMap<Option<String>, KnowledgeGraph>> {
        let buf = BufReader::new(reader);
        let mut graphs: HashMap<Option<String>, KnowledgeGraph> = HashMap::new();

        for line in buf.lines() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Ok(quad) = Quad::from_nquads(line) {
                graphs
                    .entry(quad.graph.clone())
                    .or_insert_with(KnowledgeGraph::new)
                    .add_triple(quad.triple);
            }
        }

        Ok(graphs)
    }

    /// Parse to single merged graph (ignores graph names).
    pub fn read_merged<R: Read>(reader: R) -> Result<KnowledgeGraph> {
        let graphs = Self::read(reader)?;
        let mut merged = KnowledgeGraph::new();

        for (_name, kg) in graphs {
            for triple in kg.triples() {
                merged.add_triple(triple.clone());
            }
        }

        Ok(merged)
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
}
