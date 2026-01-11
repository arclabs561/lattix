//! Turtle format (RDF 1.2).
//!
//! Human-readable RDF serialization with prefix support.
//!
//! Reference: <https://www.w3.org/TR/rdf12-turtle/>
//!
//! Example:
//! ```turtle
//! @prefix ex: <http://example.org/> .
//!
//! ex:Apple ex:founded_by ex:Steve_Jobs ;
//!          ex:headquartered_in ex:Cupertino .
//! ```

use crate::{KnowledgeGraph, Result};
use std::collections::HashMap;
use std::io::Write;

/// Turtle format handler.
pub struct Turtle;

impl Turtle {
    /// Write knowledge graph to Turtle format.
    ///
    /// Groups triples by subject for readability.
    pub fn write<W: Write>(
        kg: &KnowledgeGraph,
        mut writer: W,
        prefixes: &HashMap<String, String>,
    ) -> Result<()> {
        // Write prefixes
        for (prefix, uri) in prefixes {
            writeln!(writer, "@prefix {}: <{}> .", prefix, uri)?;
        }
        if !prefixes.is_empty() {
            writeln!(writer)?;
        }

        // Group triples by subject
        let mut by_subject: HashMap<String, Vec<_>> = HashMap::new();
        for triple in kg.triples() {
            by_subject
                .entry(triple.subject.as_str().to_string())
                .or_default()
                .push(triple);
        }

        // Write grouped triples
        for (subject, triples) in by_subject {
            let subject_str = format_uri(&subject, prefixes);
            write!(writer, "{}", subject_str)?;

            for (i, triple) in triples.iter().enumerate() {
                let pred = format_uri(triple.predicate.as_str(), prefixes);
                let obj = format_uri(triple.object.as_str(), prefixes);

                if i == 0 {
                    write!(writer, " {} {}", pred, obj)?;
                } else {
                    write!(writer, " ;\n    {} {}", pred, obj)?;
                }
            }
            writeln!(writer, " .")?;
        }

        Ok(())
    }

    /// Write with default prefixes.
    pub fn write_default<W: Write>(kg: &KnowledgeGraph, writer: W) -> Result<()> {
        let prefixes = default_prefixes();
        Self::write(kg, writer, &prefixes)
    }

    /// Convert to string with default prefixes.
    pub fn to_string(kg: &KnowledgeGraph) -> Result<String> {
        let mut buf = Vec::new();
        Self::write_default(kg, &mut buf)?;
        Ok(String::from_utf8_lossy(&buf).to_string())
    }
}

/// Format a URI, compacting with prefixes if possible.
fn format_uri(uri: &str, prefixes: &HashMap<String, String>) -> String {
    for (prefix, base) in prefixes {
        if uri.starts_with(base) {
            let local = &uri[base.len()..];
            // Check if local part is valid for Turtle prefixed name
            if is_valid_local_name(local) {
                return format!("{}:{}", prefix, local);
            }
        }
    }
    format!("<{}>", uri)
}

/// Check if a string is a valid Turtle local name.
fn is_valid_local_name(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    s.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

/// Default prefixes for common vocabularies.
pub fn default_prefixes() -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("rdf".into(), "http://www.w3.org/1999/02/22-rdf-syntax-ns#".into());
    m.insert("rdfs".into(), "http://www.w3.org/2000/01/rdf-schema#".into());
    m.insert("xsd".into(), "http://www.w3.org/2001/XMLSchema#".into());
    m.insert("owl".into(), "http://www.w3.org/2002/07/owl#".into());
    m.insert("prov".into(), "http://www.w3.org/ns/prov#".into());
    m.insert("schema".into(), "https://schema.org/".into());
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
            "http://example.org/Apple",
            "http://example.org/founded_by",
            "http://example.org/Steve_Jobs",
        ));

        let output = Turtle::to_string(&kg).unwrap();
        assert!(output.contains("@prefix"));
        assert!(output.contains("Apple"));
    }
}
