//! Turtle format.
//!
//! Human-readable RDF serialization with prefix support.
//!
//! Reference: <https://www.w3.org/TR/rdf12-turtle/>

use super::oxrdf_helpers::{subject_to_string, term_to_string};
use crate::{KnowledgeGraph, Result, Triple};
use oxttl::TurtleParser;
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};

/// Turtle format handler.
///
/// Reads Turtle syntax (with prefix declarations and base IRIs) via
/// [`oxttl::TurtleParser`]. Writing groups triples by subject and uses
/// configured prefixes when a term has a simple local name.
pub struct Turtle;

impl Turtle {
    /// Parse Turtle from a reader.
    ///
    /// `base_iri` is optional; when provided it resolves relative IRIs.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> lattix::Result<()> {
    /// use lattix::formats::Turtle;
    ///
    /// let ttl = r#"
    /// @prefix ex: <http://example.org/> .
    /// ex:Alice ex:knows ex:Bob .
    /// "#;
    /// let kg = Turtle::read(std::io::Cursor::new(ttl), None)?;
    /// assert_eq!(kg.triple_count(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn read<R: Read>(reader: R, base_iri: Option<&str>) -> Result<KnowledgeGraph> {
        let mut parser = TurtleParser::new();
        if let Some(base) = base_iri {
            parser = parser
                .with_base_iri(base)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        }

        let mut kg = KnowledgeGraph::new();

        for result in parser.for_reader(reader) {
            let triple =
                result.map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let s = subject_to_string(&triple.subject);
            let p = crate::rdf::iri_to_string(triple.predicate.as_str());
            let o = term_to_string(&triple.object);

            kg.add_triple(Triple::new(s, p, o));
        }

        Ok(kg)
    }

    /// Write knowledge graph to Turtle format.
    ///
    /// Outputs Turtle syntax grouped by subject. Multiple predicates for the
    /// same subject are joined with `;`, and configured prefixes are used when
    /// a term has a simple local name.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> lattix::Result<()> {
    /// use lattix::{KnowledgeGraph, Triple, formats::Turtle};
    ///
    /// let mut kg = KnowledgeGraph::new();
    /// kg.add_triple(Triple::new(
    ///     "http://example.org/Alice",
    ///     "http://example.org/knows",
    ///     "http://example.org/Bob",
    /// ));
    /// let output = Turtle::to_string(&kg)?;
    /// assert!(output.contains("Alice"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn write<W: Write>(
        kg: &KnowledgeGraph,
        mut writer: W,
        prefixes: &HashMap<String, String>,
    ) -> Result<()> {
        let prefixes: BTreeMap<_, _> = prefixes.iter().collect();
        for (prefix, iri) in &prefixes {
            writeln!(
                writer,
                "@prefix {}: <{}> .",
                prefix,
                crate::rdf::escape_iri(iri)
            )?;
        }
        if !prefixes.is_empty() {
            writeln!(writer)?;
        }

        let mut by_subject: BTreeMap<&str, Vec<_>> = BTreeMap::new();
        for triple in kg.triples() {
            by_subject
                .entry(triple.subject().as_str())
                .or_default()
                .push(triple);
        }

        for (subject, triples) in &mut by_subject {
            triples.sort_by(|a, b| {
                a.predicate()
                    .as_str()
                    .cmp(b.predicate().as_str())
                    .then_with(|| a.object().as_str().cmp(b.object().as_str()))
            });
            let s = render_term(subject, &prefixes);

            for (i, triple) in triples.iter().enumerate() {
                let p = render_term(triple.predicate().as_str(), &prefixes);
                let o = render_term(triple.object().as_str(), &prefixes);

                if i == 0 {
                    write!(writer, "{} {} {}", s, p, o)?;
                } else {
                    write!(writer, " ;\n    {} {}", p, o)?;
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

    /// Convert to string.
    pub fn to_string(kg: &KnowledgeGraph) -> Result<String> {
        let mut buf = Vec::new();
        Self::write_default(kg, &mut buf)?;
        Ok(String::from_utf8_lossy(&buf).to_string())
    }
}

fn render_term(s: &str, prefixes: &BTreeMap<&String, &String>) -> String {
    if s.starts_with("_:") {
        return s.to_string();
    }
    if s.starts_with('"') {
        return crate::rdf::render_object(s);
    }

    let iri = s
        .strip_prefix('<')
        .and_then(|s| s.strip_suffix('>'))
        .unwrap_or(s);

    if let Some((prefix, local)) = prefixes
        .iter()
        .filter_map(|(prefix, base)| {
            let local = iri.strip_prefix(base.as_str())?;
            is_simple_local_name(local).then_some((*prefix, *base, local))
        })
        .max_by_key(|(_, base, _)| base.len())
        .map(|(prefix, _, local)| (prefix, local))
    {
        return format!("{prefix}:{local}");
    }

    format!("<{}>", crate::rdf::iri_body_for(iri))
}

fn is_simple_local_name(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
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
    fn test_turtle_parse() {
        let input = r#"
@prefix ex: <http://example.org/> .
ex:Apple ex:founded_by ex:Steve_Jobs .
"#;
        let kg = Turtle::read(std::io::Cursor::new(input), None).unwrap();
        assert_eq!(kg.triple_count(), 1);
        let triple = kg.triples().next().unwrap();
        assert_eq!(triple.subject().as_str(), "http://example.org/Apple");
        assert_eq!(triple.predicate().as_str(), "http://example.org/founded_by");
        assert_eq!(triple.object().as_str(), "http://example.org/Steve_Jobs");
    }

    #[test]
    fn test_turtle_output() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new(
            "http://example.org/Apple",
            "http://example.org/founded_by",
            "http://example.org/Steve_Jobs",
        ));

        let output = Turtle::to_string(&kg).unwrap();
        assert!(output.contains("Apple"));
    }

    #[test]
    fn test_turtle_with_base_iri() {
        let input = r#"
@base <http://example.org/> .
<Apple> <founded_by> <Steve_Jobs> .
"#;
        let kg = Turtle::read(std::io::Cursor::new(input), Some("http://example.org/")).unwrap();
        assert_eq!(kg.triple_count(), 1);
    }
}
