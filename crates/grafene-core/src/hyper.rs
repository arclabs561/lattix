//! Hypergraph extensions for n-ary relations.
//!
//! This module provides types for representing knowledge beyond binary relations:
//!
//! - [`HyperTriple`] - Triple with qualifiers (Wikidata-style)
//! - [`HyperEdge`] - N-ary relation with role-value bindings
//! - [`HyperGraph`] - Graph structure supporting hyperedges
//!
//! # When to Use What
//!
//! | Type | Use Case | Example |
//! |------|----------|---------|
//! | `Triple` | Simple binary facts | (Einstein, born_in, Ulm) |
//! | `HyperTriple` | Facts with context | (Einstein, won, Nobel) + {year: 1921} |
//! | `HyperEdge` | True n-ary relations | {buyer: Alice, item: Book, price: $20, date: 2024} |
//!
//! # Embedding Considerations
//!
//! Different structures need different embedding methods:
//!
//! | Structure | Embedding Methods | Key Idea |
//! |-----------|-------------------|----------|
//! | Triple | TransE, RotatE, ComplEx | h + r ≈ t |
//! | HyperTriple | StarE, HINGE | Qualifier-aware scoring |
//! | HyperEdge | HSimplE, HypE, HyCubE | Position/role-aware convolution |
//!
//! # References
//!
//! - Fatemi et al. (2019) "Knowledge Hypergraphs: Prediction Beyond Binary Relations"
//! - Galkin et al. (2020) "Message Passing for Hyper-Relational Knowledge Graphs"
//! - Wang et al. (2025) "Understanding Embedding Models on Hyper-relational KGs"

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{EntityId, RelationType, Triple};

/// A triple with optional qualifiers (key-value pairs).
///
/// This is the Wikidata-style representation where a core triple can have
/// additional context attached. For example:
///
/// ```text
/// Core triple: (Einstein, educated_at, ETH Zurich)
/// Qualifiers:  {degree: PhD, field: Physics, year: 1905}
/// ```
///
/// # Embedding
///
/// For embedding HyperTriples, consider:
/// - **StarE**: Transforms qualifiers into the relation space
/// - **HINGE**: Handles qualifiers via attention mechanism
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperTriple {
    /// The core triple (subject, predicate, object)
    pub core: Triple,

    /// Qualifier key-value pairs providing context
    /// Keys are relation types, values are entity IDs
    pub qualifiers: HashMap<RelationType, EntityId>,
}

impl HyperTriple {
    /// Create a new hyper-triple from a core triple.
    pub fn new(core: Triple) -> Self {
        Self {
            core,
            qualifiers: HashMap::new(),
        }
    }

    /// Create from components.
    pub fn from_parts(
        subject: impl Into<EntityId>,
        predicate: impl Into<RelationType>,
        object: impl Into<EntityId>,
    ) -> Self {
        Self::new(Triple::new(subject, predicate, object))
    }

    /// Add a qualifier to this hyper-triple.
    pub fn with_qualifier(
        mut self,
        key: impl Into<RelationType>,
        value: impl Into<EntityId>,
    ) -> Self {
        self.qualifiers.insert(key.into(), value.into());
        self
    }

    /// Add multiple qualifiers.
    pub fn with_qualifiers(
        mut self,
        qualifiers: impl IntoIterator<Item = (impl Into<RelationType>, impl Into<EntityId>)>,
    ) -> Self {
        for (k, v) in qualifiers {
            self.qualifiers.insert(k.into(), v.into());
        }
        self
    }

    /// Get the arity (number of entities involved).
    /// Core triple has 2 entities, each qualifier adds 1.
    pub fn arity(&self) -> usize {
        2 + self.qualifiers.len()
    }

    /// Iterate over all entities in this hyper-triple.
    pub fn entities(&self) -> impl Iterator<Item = &EntityId> {
        std::iter::once(&self.core.subject)
            .chain(std::iter::once(&self.core.object))
            .chain(self.qualifiers.values())
    }
}

/// A role-value binding in a hyperedge.
///
/// Unlike positional representations, role-value bindings explicitly
/// label the semantic function of each entity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RoleBinding {
    /// The semantic role (e.g., "buyer", "seller", "timestamp")
    pub role: String,

    /// The entity filling this role
    pub entity: EntityId,
}

impl RoleBinding {
    pub fn new(role: impl Into<String>, entity: impl Into<EntityId>) -> Self {
        Self {
            role: role.into(),
            entity: entity.into(),
        }
    }
}

/// An n-ary relation represented as a hyperedge.
///
/// A hyperedge connects multiple entities with explicit semantic roles.
/// This is the native representation for facts involving more than 2 entities.
///
/// ```text
/// HyperEdge {
///   relation: "purchase_event",
///   bindings: [
///     (buyer, Alice),
///     (seller, Amazon),
///     (item, "Rust Book"),
///     (price, "$50"),
///     (date, "2024-01-15")
///   ]
/// }
/// ```
///
/// # Embedding
///
/// For embedding HyperEdges, consider:
/// - **HSimplE**: Cyclic position encoding with SimplE-style scoring
/// - **HypE**: Position-specific convolutional filters
/// - **HyCubE**: 3D circular convolutions on entity-relation cubes
///
/// # Comparison with Reification
///
/// Reification creates artificial intermediate nodes:
/// ```text
/// Reified:  (Event_1, buyer, Alice), (Event_1, seller, Amazon), ...
/// ```
///
/// HyperEdge preserves atomic structure - all entities are first-class
/// participants in a single fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperEdge {
    /// The relation type for this hyperedge
    pub relation: RelationType,

    /// Role-value bindings (ordered for positional encoding)
    pub bindings: Vec<RoleBinding>,

    /// Optional confidence score in [0, 1]
    pub confidence: Option<f32>,
}

impl HyperEdge {
    /// Create a new hyperedge with the given relation.
    pub fn new(relation: impl Into<RelationType>) -> Self {
        Self {
            relation: relation.into(),
            bindings: Vec::new(),
            confidence: None,
        }
    }

    /// Add a role-entity binding.
    pub fn with_binding(mut self, role: impl Into<String>, entity: impl Into<EntityId>) -> Self {
        self.bindings.push(RoleBinding::new(role, entity));
        self
    }

    /// Add multiple bindings from an iterator.
    pub fn with_bindings(
        mut self,
        bindings: impl IntoIterator<Item = (impl Into<String>, impl Into<EntityId>)>,
    ) -> Self {
        for (role, entity) in bindings {
            self.bindings.push(RoleBinding::new(role, entity));
        }
        self
    }

    /// Set the confidence score.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Get the arity (number of entities).
    pub fn arity(&self) -> usize {
        self.bindings.len()
    }

    /// Get entity at a specific position (for positional encoding).
    pub fn entity_at(&self, position: usize) -> Option<&EntityId> {
        self.bindings.get(position).map(|b| &b.entity)
    }

    /// Get entity by role name.
    pub fn entity_by_role(&self, role: &str) -> Option<&EntityId> {
        self.bindings
            .iter()
            .find(|b| b.role == role)
            .map(|b| &b.entity)
    }

    /// Iterate over all entities.
    pub fn entities(&self) -> impl Iterator<Item = &EntityId> {
        self.bindings.iter().map(|b| &b.entity)
    }

    /// Iterate over all roles.
    pub fn roles(&self) -> impl Iterator<Item = &str> {
        self.bindings.iter().map(|b| b.role.as_str())
    }

    /// Convert to a set of reified triples (for compatibility with triple-based systems).
    ///
    /// Creates an intermediate node and connects all bindings to it.
    /// This loses the atomic nature of the hyperedge but enables use with
    /// triple-based embedding methods.
    pub fn reify(&self, intermediate_id: impl Into<EntityId>) -> Vec<Triple> {
        let intermediate = intermediate_id.into();
        let mut triples = Vec::with_capacity(self.bindings.len() + 1);

        // Add the relation type as a triple
        triples.push(Triple::new(
            intermediate.clone(),
            "rdf:type",
            self.relation.as_str(),
        ));

        // Add each binding as a triple
        for binding in &self.bindings {
            triples.push(Triple::new(
                intermediate.clone(),
                binding.role.clone(),
                binding.entity.clone(),
            ));
        }

        triples
    }
}

/// A knowledge hypergraph supporting both triples and hyperedges.
///
/// This structure allows mixing binary relations (triples) with
/// n-ary relations (hyperedges) in the same graph.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HyperGraph {
    /// Standard triples (binary relations)
    pub triples: Vec<Triple>,

    /// Hyper-triples (triples with qualifiers)
    pub hyper_triples: Vec<HyperTriple>,

    /// True hyperedges (n-ary relations)
    pub hyperedges: Vec<HyperEdge>,
}

impl HyperGraph {
    /// Create an empty hypergraph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a standard triple.
    pub fn add_triple(&mut self, triple: Triple) {
        self.triples.push(triple);
    }

    /// Add a hyper-triple (triple with qualifiers).
    pub fn add_hyper_triple(&mut self, hyper_triple: HyperTriple) {
        self.hyper_triples.push(hyper_triple);
    }

    /// Add a hyperedge (n-ary relation).
    pub fn add_hyperedge(&mut self, hyperedge: HyperEdge) {
        self.hyperedges.push(hyperedge);
    }

    /// Get total number of facts (triples + hyper-triples + hyperedges).
    pub fn fact_count(&self) -> usize {
        self.triples.len() + self.hyper_triples.len() + self.hyperedges.len()
    }

    /// Get all unique entities across all facts.
    pub fn entities(&self) -> std::collections::HashSet<&EntityId> {
        let mut entities = std::collections::HashSet::new();

        for t in &self.triples {
            entities.insert(&t.subject);
            entities.insert(&t.object);
        }

        for ht in &self.hyper_triples {
            for e in ht.entities() {
                entities.insert(e);
            }
        }

        for he in &self.hyperedges {
            for e in he.entities() {
                entities.insert(e);
            }
        }

        entities
    }

    /// Convert entire hypergraph to reified triples.
    ///
    /// This enables use with triple-based embedding methods but loses
    /// the semantic structure of n-ary relations.
    pub fn to_reified_triples(&self) -> Vec<Triple> {
        let mut result = self.triples.clone();

        // Convert hyper-triples: keep core, add qualifier triples
        for (i, ht) in self.hyper_triples.iter().enumerate() {
            result.push(ht.core.clone());
            let statement_id = format!("_:stmt_{}", i);
            for (key, value) in &ht.qualifiers {
                result.push(Triple::new(statement_id.clone(), key.clone(), value.clone()));
            }
        }

        // Convert hyperedges
        for (i, he) in self.hyperedges.iter().enumerate() {
            let node_id = format!("_:hyperedge_{}", i);
            result.extend(he.reify(node_id));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyper_triple_creation() {
        let ht = HyperTriple::from_parts("Einstein", "educated_at", "ETH Zurich")
            .with_qualifier("degree", "PhD")
            .with_qualifier("year", "1905");

        assert_eq!(ht.arity(), 4); // subject, object, + 2 qualifiers
        assert_eq!(ht.qualifiers.len(), 2);
    }

    #[test]
    fn test_hyper_edge_creation() {
        let he = HyperEdge::new("purchase")
            .with_binding("buyer", "Alice")
            .with_binding("seller", "Amazon")
            .with_binding("item", "Rust Book")
            .with_binding("price", "$50");

        assert_eq!(he.arity(), 4);
        assert_eq!(he.entity_by_role("buyer"), Some(&EntityId::from("Alice")));
        assert_eq!(he.entity_at(0), Some(&EntityId::from("Alice")));
    }

    #[test]
    fn test_hyper_edge_reification() {
        let he = HyperEdge::new("award")
            .with_binding("recipient", "Einstein")
            .with_binding("prize", "Nobel")
            .with_binding("year", "1921");

        let reified = he.reify("_:award_1");

        // Should have 4 triples: type + 3 bindings
        assert_eq!(reified.len(), 4);

        // First triple is the type
        assert_eq!(reified[0].predicate.as_str(), "rdf:type");
    }

    #[test]
    fn test_hyper_graph_mixed() {
        let mut hg = HyperGraph::new();

        // Add regular triple
        hg.add_triple(Triple::new("Einstein", "born_in", "Ulm"));

        // Add hyper-triple
        hg.add_hyper_triple(
            HyperTriple::from_parts("Einstein", "won", "Nobel Prize")
                .with_qualifier("year", "1921"),
        );

        // Add hyperedge
        hg.add_hyperedge(
            HyperEdge::new("collaboration")
                .with_binding("scientist_1", "Einstein")
                .with_binding("scientist_2", "Bohr")
                .with_binding("topic", "Quantum Mechanics"),
        );

        assert_eq!(hg.fact_count(), 3);

        let entities = hg.entities();
        assert!(entities.contains(&EntityId::from("Einstein")));
        assert!(entities.contains(&EntityId::from("Bohr")));
    }
}
