//! Graph interchange / export types.
//!
//! This module provides a **serde-friendly** graph representation intended for interchange
//! (Neo4j, NetworkX, JSON-LD) rather than for high-performance algorithms.
//!
//! The structure is intentionally close to “property graph” conventions:
//! - nodes have `id`, `node_type`, `name`, and arbitrary `properties`
//! - edges have `source`, `target`, `relation`, and arbitrary `properties`
//!
//! Algorithmic code should generally use `KnowledgeGraph` / `HeteroGraph` / `HyperGraph`.
//! Extraction pipelines (NER/coref/RE) can emit a `GraphDocument` for downstream tooling.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A node in an interchange graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node identifier.
    pub id: String,
    /// Node type/label (e.g. "Person", "Organization").
    pub node_type: String,
    /// Display name (canonical mention text).
    pub name: String,
    /// Arbitrary properties.
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
}

impl GraphNode {
    /// Create a new graph node.
    #[must_use]
    pub fn new(id: impl Into<String>, node_type: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            node_type: node_type.into(),
            name: name.into(),
            properties: HashMap::new(),
        }
    }

    /// Add a property to the node.
    #[must_use]
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Add mention count property.
    #[must_use]
    pub fn with_mentions_count(self, count: usize) -> Self {
        self.with_property("mentions_count", count)
    }

    /// Add first occurrence offset.
    #[must_use]
    pub fn with_first_seen(self, offset: usize) -> Self {
        self.with_property("first_seen", offset)
    }
}

/// An edge in an interchange graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID.
    pub source: String,
    /// Target node ID.
    pub target: String,
    /// Relation type (edge label).
    pub relation: String,
    /// Confidence score \(0.0..=1.0\).
    #[serde(default)]
    pub confidence: f64,
    /// Arbitrary properties.
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
}

impl GraphEdge {
    /// Create a new graph edge.
    #[must_use]
    pub fn new(source: impl Into<String>, target: impl Into<String>, relation: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            relation: relation.into(),
            confidence: 1.0,
            properties: HashMap::new(),
        }
    }

    /// Set confidence score.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Add a property to the edge.
    #[must_use]
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Add trigger text property.
    #[must_use]
    pub fn with_trigger(self, trigger: impl Into<String>) -> Self {
        self.with_property("trigger", trigger.into())
    }
}

/// A complete graph document ready for export.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphDocument {
    /// Nodes (entities).
    pub nodes: Vec<GraphNode>,
    /// Edges (relations).
    pub edges: Vec<GraphEdge>,
    /// Document metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl GraphDocument {
    /// Create an empty graph document.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add metadata to the graph document.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get node count.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edge count.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Check if graph is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Export to Neo4j Cypher `CREATE` statements.
    #[must_use]
    pub fn to_cypher(&self) -> String {
        let mut cypher = String::new();

        // Create nodes
        for node in &self.nodes {
            let props = format_cypher_props(&node.properties, &node.name);
            cypher.push_str(&format!(
                "CREATE (n{}:{} {{id: '{}'{}}});\n",
                sanitize_cypher_name(&node.id),
                sanitize_cypher_name(&node.node_type),
                escape_cypher_string(&node.id),
                props
            ));
        }

        cypher.push('\n');

        // Create edges
        for edge in &self.edges {
            let props = if edge.confidence < 1.0 {
                format!(" {{confidence: {:.3}}}", edge.confidence)
            } else {
                String::new()
            };

            cypher.push_str(&format!(
                "MATCH (a {{id: '{}'}}), (b {{id: '{}'}}) CREATE (a)-[:{}{}]->(b);\n",
                escape_cypher_string(&edge.source),
                escape_cypher_string(&edge.target),
                sanitize_cypher_name(&edge.relation),
                props
            ));
        }

        cypher
    }

    /// Export to NetworkX-compatible JSON format.
    ///
    /// This format can be loaded directly with:
    /// ```python
    /// import networkx as nx
    /// import json
    /// with open('graph.json') as f:
    ///     data = json.load(f)
    /// G = nx.node_link_graph(data)
    /// ```
    #[must_use]
    pub fn to_networkx_json(&self) -> String {
        #[derive(Serialize)]
        struct NetworkXGraph<'a> {
            directed: bool,
            multigraph: bool,
            graph: HashMap<String, serde_json::Value>,
            nodes: Vec<NetworkXNode<'a>>,
            links: Vec<NetworkXLink<'a>>,
        }

        #[derive(Serialize)]
        struct NetworkXNode<'a> {
            id: &'a str,
            #[serde(rename = "type")]
            node_type: &'a str,
            name: &'a str,
            #[serde(flatten)]
            properties: &'a HashMap<String, serde_json::Value>,
        }

        #[derive(Serialize)]
        struct NetworkXLink<'a> {
            source: &'a str,
            target: &'a str,
            relation: &'a str,
            #[serde(skip_serializing_if = "is_default_confidence")]
            confidence: f64,
            #[serde(flatten)]
            properties: &'a HashMap<String, serde_json::Value>,
        }

        fn is_default_confidence(c: &f64) -> bool {
            (*c - 1.0).abs() < f64::EPSILON
        }

        let graph = NetworkXGraph {
            directed: true,
            multigraph: false,
            graph: self.metadata.clone(),
            nodes: self
                .nodes
                .iter()
                .map(|n| NetworkXNode {
                    id: &n.id,
                    node_type: &n.node_type,
                    name: &n.name,
                    properties: &n.properties,
                })
                .collect(),
            links: self
                .edges
                .iter()
                .map(|e| NetworkXLink {
                    source: &e.source,
                    target: &e.target,
                    relation: &e.relation,
                    confidence: e.confidence,
                    properties: &e.properties,
                })
                .collect(),
        };

        serde_json::to_string_pretty(&graph).unwrap_or_else(|_| "{}".to_string())
    }

    /// Export to JSON-LD format (for semantic web applications).
    #[must_use]
    pub fn to_json_ld(&self) -> String {
        #[derive(Serialize)]
        struct JsonLd<'a> {
            #[serde(rename = "@context")]
            context: JsonLdContext,
            #[serde(rename = "@graph")]
            graph: Vec<JsonLdNode<'a>>,
        }

        #[derive(Serialize)]
        struct JsonLdContext {
            #[serde(rename = "@vocab")]
            vocab: &'static str,
            name: &'static str,
            #[serde(rename = "type")]
            type_: &'static str,
        }

        #[derive(Serialize)]
        struct JsonLdNode<'a> {
            #[serde(rename = "@id")]
            id: &'a str,
            #[serde(rename = "@type")]
            node_type: &'a str,
            name: &'a str,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            relations: Vec<JsonLdRelation<'a>>,
        }

        #[derive(Serialize)]
        struct JsonLdRelation<'a> {
            #[serde(rename = "@type")]
            relation_type: &'a str,
            target: &'a str,
        }

        // Group edges by source
        let mut node_edges: HashMap<&str, Vec<&GraphEdge>> = HashMap::new();
        for edge in &self.edges {
            node_edges.entry(&edge.source).or_default().push(edge);
        }

        let doc = JsonLd {
            context: JsonLdContext {
                vocab: "http://schema.org/",
                name: "http://schema.org/name",
                type_: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            },
            graph: self
                .nodes
                .iter()
                .map(|n| JsonLdNode {
                    id: &n.id,
                    node_type: &n.node_type,
                    name: &n.name,
                    relations: node_edges
                        .get(n.id.as_str())
                        .map(|edges| {
                            edges
                                .iter()
                                .map(|e| JsonLdRelation {
                                    relation_type: &e.relation,
                                    target: &e.target,
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                })
                .collect(),
        };

        serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".to_string())
    }

    /// Export to the specified format.
    #[must_use]
    pub fn export(&self, format: GraphExportFormat) -> String {
        match format {
            GraphExportFormat::Cypher => self.to_cypher(),
            GraphExportFormat::NetworkXJson => self.to_networkx_json(),
            GraphExportFormat::JsonLd => self.to_json_ld(),
        }
    }
}

/// Supported graph export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphExportFormat {
    /// Neo4j Cypher CREATE statements.
    Cypher,
    /// NetworkX-compatible JSON (node_link_graph format).
    NetworkXJson,
    /// JSON-LD for semantic web.
    JsonLd,
}

/// Format properties for Cypher (excluding name which is handled separately).
fn format_cypher_props(props: &HashMap<String, serde_json::Value>, name: &str) -> String {
    let mut parts = vec![format!("name: '{}'", escape_cypher_string(name))];

    for (key, value) in props {
        let formatted = match value {
            serde_json::Value::String(s) => format!("{}: '{}'", key, escape_cypher_string(s)),
            serde_json::Value::Number(n) => format!("{}: {}", key, n),
            serde_json::Value::Bool(b) => format!("{}: {}", key, b),
            _ => continue,
        };
        parts.push(formatted);
    }

    if parts.len() > 1 {
        format!(", {}", parts[1..].join(", "))
    } else {
        String::new()
    }
}

/// Escape special characters in Cypher strings.
fn escape_cypher_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

/// Sanitize names for Cypher identifiers.
fn sanitize_cypher_name(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

