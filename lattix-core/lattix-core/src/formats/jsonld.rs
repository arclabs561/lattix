//! JSON-LD format.
//!
//! JSON-based linked data serialization.
//!
//! Reference: <https://www.w3.org/TR/json-ld11/>

use crate::{KnowledgeGraph, Result};
use serde_json::{json, Value};
use std::io::Write;

/// JSON-LD format handler.
pub struct JsonLd;

impl JsonLd {
    /// Write knowledge graph to JSON-LD format.
    pub fn write<W: Write>(kg: &KnowledgeGraph, mut writer: W) -> Result<()> {
        let doc = Self::to_value(kg);
        let json = serde_json::to_string_pretty(&doc)?;
        writer.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Convert to JSON-LD Value.
    pub fn to_value(kg: &KnowledgeGraph) -> Value {
        let mut graph = Vec::new();

        // Group by subject for cleaner output
        let mut by_subject: std::collections::HashMap<String, Vec<_>> =
            std::collections::HashMap::new();
        for triple in kg.triples() {
            by_subject
                .entry(triple.subject.as_str().to_string())
                .or_default()
                .push(triple);
        }

        for (subject, triples) in by_subject {
            let mut node = json!({
                "@id": subject,
            });

            for triple in triples {
                let pred = triple.predicate.as_str();
                let obj = triple.object.as_str();

                // Use @id for object if it looks like a URI
                let obj_value = if obj.starts_with("http://") || obj.starts_with("https://") {
                    json!({ "@id": obj })
                } else {
                    json!(obj)
                };

                // Handle multiple values for same predicate
                if let Some(existing) = node.get_mut(pred) {
                    if let Some(arr) = existing.as_array_mut() {
                        arr.push(obj_value);
                    } else {
                        let old = existing.take();
                        *existing = json!([old, obj_value]);
                    }
                } else {
                    node[pred] = obj_value;
                }
            }

            graph.push(node);
        }

        json!({
            "@context": {
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "xsd": "http://www.w3.org/2001/XMLSchema#",
                "prov": "http://www.w3.org/ns/prov#"
            },
            "@graph": graph
        })
    }

    /// Convert to string.
    pub fn to_string(kg: &KnowledgeGraph) -> Result<String> {
        let mut buf = Vec::new();
        Self::write(kg, &mut buf)?;
        Ok(String::from_utf8_lossy(&buf).to_string())
    }

    /// Parse JSON-LD to knowledge graph (basic implementation).
    ///
    /// Note: Full JSON-LD parsing requires expansion/compaction algorithms.
    /// This is a simplified parser that handles the @graph format.
    pub fn from_value(doc: &Value) -> Result<KnowledgeGraph> {
        let mut kg = KnowledgeGraph::new();

        // Handle @graph array
        if let Some(graph) = doc.get("@graph").and_then(|v| v.as_array()) {
            for node in graph {
                Self::parse_node(node, &mut kg)?;
            }
        } else if doc.get("@id").is_some() {
            // Single node document
            Self::parse_node(doc, &mut kg)?;
        }

        Ok(kg)
    }

    fn parse_node(node: &Value, kg: &mut KnowledgeGraph) -> Result<()> {
        let subject = node
            .get("@id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| crate::Error::ParseTriple("Missing @id".into()))?;

        for (key, value) in node.as_object().into_iter().flatten() {
            // Skip JSON-LD keywords
            if key.starts_with('@') {
                continue;
            }

            // Handle single value or array
            let values: Vec<&Value> = if let Some(arr) = value.as_array() {
                arr.iter().collect()
            } else {
                vec![value]
            };

            for val in values {
                let object = if let Some(id) = val.get("@id").and_then(|v| v.as_str()) {
                    id.to_string()
                } else if let Some(s) = val.as_str() {
                    s.to_string()
                } else {
                    continue;
                };

                kg.add_triple(crate::Triple::new(subject, key.as_str(), object));
            }
        }

        Ok(())
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Result<KnowledgeGraph> {
        let doc: Value = serde_json::from_str(s)?;
        Self::from_value(&doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn test_roundtrip() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new(
            "http://example.org/Apple",
            "http://example.org/founded_by",
            "http://example.org/Steve_Jobs",
        ));

        let json = JsonLd::to_string(&kg).unwrap();
        let parsed = JsonLd::from_str(&json).unwrap();

        assert_eq!(parsed.triple_count(), 1);
    }
}
