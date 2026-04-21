//! sophia_api trait implementations for [`KnowledgeGraph`].
//!
//! Implements [`Graph`], [`MutableGraph`], and [`CollectibleGraph`] from
//! sophia_api 0.8, gated behind the `sophia` feature.
//!
//! # Term mapping
//!
//! Lattix stores all triple components as plain strings. The sophia bridge
//! maps them to RDF terms using prefix conventions:
//!
//! - `_:xxx` is a blank node
//! - `"xxx"` (or `"xxx"^^dt` / `"xxx"@lang`) is a literal
//! - everything else is treated as an IRI
//!
//! [`Graph`]: sophia_api::graph::Graph
//! [`MutableGraph`]: sophia_api::graph::MutableGraph
//! [`CollectibleGraph`]: sophia_api::graph::CollectibleGraph

use sophia_api::graph::{CollectibleGraph, GTripleSource, Graph, MgResult, MutableGraph};
use sophia_api::source::{StreamResult, TripleSource};
use sophia_api::term::bnode_id::BnodeId;
use sophia_api::term::{IriRef, SimpleTerm, Term};

use crate::{KnowledgeGraph, Triple};

/// xsd:string IRI as a static str, used for plain literals.
const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";

/// Convert a string slice from lattix storage into a [`SimpleTerm`].
///
/// - `_:xxx` becomes a blank node
/// - strings starting with `"` become literals (passed through to SimpleTerm)
/// - everything else becomes an IRI (unchecked)
fn str_to_term(s: &str) -> SimpleTerm<'_> {
    if let Some(rest) = s.strip_prefix("_:") {
        SimpleTerm::BlankNode(BnodeId::new_unchecked(rest.into()))
    } else if s.starts_with('"') {
        // Literal: try to parse "lexical"^^<datatype> or "lexical"@lang
        // For simplicity, treat the whole string as a plain literal's lexical form
        // by stripping outer quotes.
        let inner = s.trim_start_matches('"');
        // Find the closing quote
        if let Some(close) = inner.find('"') {
            let lexical = &inner[..close];
            let rest = &inner[close + 1..];
            if let Some(lang) = rest.strip_prefix('@') {
                SimpleTerm::LiteralLanguage(
                    lexical.into(),
                    sophia_api::term::language_tag::LanguageTag::new_unchecked(lang.into()),
                )
            } else if let Some(dt) = rest.strip_prefix("^^<") {
                let dt = dt.trim_end_matches('>');
                SimpleTerm::LiteralDatatype(lexical.into(), IriRef::new_unchecked(dt.into()))
            } else {
                // Plain string literal
                SimpleTerm::LiteralDatatype(
                    lexical.into(),
                    IriRef::new_unchecked(XSD_STRING.into()),
                )
            }
        } else {
            // Malformed: treat whole thing as IRI fallback
            SimpleTerm::Iri(IriRef::new_unchecked(s.into()))
        }
    } else {
        SimpleTerm::Iri(IriRef::new_unchecked(s.into()))
    }
}

/// Extract a string representation from a sophia [`Term`] suitable for lattix storage.
///
/// - IRIs are stored as-is
/// - Blank nodes are stored as `_:id`
/// - Literals are stored as `"lexical"^^<datatype>` or `"lexical"@lang`
fn term_to_string(t: impl Term) -> String {
    match t.kind() {
        sophia_api::term::TermKind::Iri => {
            t.iri().map(|i| i.as_str().to_string()).unwrap_or_default()
        }
        sophia_api::term::TermKind::BlankNode => t
            .bnode_id()
            .map(|b| format!("_:{}", b.as_str()))
            .unwrap_or_default(),
        sophia_api::term::TermKind::Literal => {
            let lex = t.lexical_form().unwrap_or_else(|| "".into());
            if let Some(lang) = t.language_tag() {
                format!("\"{}\"@{}", lex, lang.as_str())
            } else if let Some(dt) = t.datatype() {
                format!("\"{}\"^^<{}>", lex, dt.as_str())
            } else {
                format!("\"{}\"", lex)
            }
        }
        sophia_api::term::TermKind::Variable => t
            .variable()
            .map(|v| format!("?{}", v.as_str()))
            .unwrap_or_default(),
        sophia_api::term::TermKind::Triple => {
            // Quoted triples: not directly supported in lattix storage.
            // Fall back to a debug representation.
            format!("{:?}", t)
        }
    }
}

impl Graph for KnowledgeGraph {
    type Triple<'x> = [SimpleTerm<'x>; 3];
    type Error = std::convert::Infallible;

    fn triples(&self) -> GTripleSource<'_, Self> {
        Box::new(self.triples().map(|t| {
            Ok([
                str_to_term(t.subject().as_str()),
                str_to_term(t.predicate().as_str()),
                str_to_term(t.object().as_str()),
            ])
        }))
    }
}

impl MutableGraph for KnowledgeGraph {
    type MutationError = std::convert::Infallible;

    fn insert<TS, TP, TO>(&mut self, s: TS, p: TP, o: TO) -> MgResult<Self, bool>
    where
        TS: Term,
        TP: Term,
        TO: Term,
    {
        let subject = term_to_string(s);
        let predicate = term_to_string(p);
        let object = term_to_string(o);
        self.add_triple(Triple::new(subject, predicate, object));
        Ok(true)
    }

    fn remove<TS, TP, TO>(&mut self, s: TS, p: TP, o: TO) -> MgResult<Self, bool>
    where
        TS: Term,
        TP: Term,
        TO: Term,
    {
        let subject = term_to_string(s);
        let predicate = term_to_string(p);
        let object = term_to_string(o);
        let removed = self.remove_triple(&subject.into(), &predicate.into(), &object.into());
        Ok(removed)
    }
}

impl CollectibleGraph for KnowledgeGraph {
    fn from_triple_source<TS: TripleSource>(
        triples: TS,
    ) -> StreamResult<Self, TS::Error, Self::Error> {
        let mut kg = KnowledgeGraph::new();
        let mut triples = triples;
        triples.try_for_each_triple(|t| -> Result<(), std::convert::Infallible> {
            use sophia_api::triple::Triple as SophiaTriple;
            let [s, p, o] = t.spo();
            let subject = term_to_string(s);
            let predicate = term_to_string(p);
            let object = term_to_string(o);
            kg.add_triple(Triple::new(subject, predicate, object));
            Ok(())
        })?;
        Ok(kg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple as LxTriple;
    use sophia_api::graph::Graph as SophiaGraph;
    use sophia_api::ns::Namespace;
    use sophia_api::source::IntoTripleSource;
    use sophia_api::term::matcher::Any;
    use sophia_api::term::SimpleTerm;

    fn sample_kg() -> KnowledgeGraph {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(LxTriple::new(
            "http://example.org/Apple",
            "http://example.org/founded_by",
            "http://example.org/SteveJobs",
        ));
        kg.add_triple(LxTriple::new(
            "http://example.org/Apple",
            "http://example.org/headquartered_in",
            "http://example.org/Cupertino",
        ));
        kg.add_triple(LxTriple::new(
            "http://example.org/SteveJobs",
            "http://example.org/born_in",
            "http://example.org/SanFrancisco",
        ));
        kg
    }

    #[test]
    fn graph_triples_count() {
        let kg = sample_kg();
        let count = SophiaGraph::triples(&kg).count();
        assert_eq!(count, 3);
    }

    #[test]
    fn graph_triples_all_ok() {
        let kg = sample_kg();
        for t in SophiaGraph::triples(&kg) {
            assert!(t.is_ok());
        }
    }

    #[test]
    fn graph_triples_matching() {
        let kg = sample_kg();
        let ex = Namespace::new("http://example.org/").unwrap();
        let apple = ex.get("Apple").unwrap();

        let matches: Vec<_> = kg
            .triples_matching([apple], Any, Any)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn graph_contains() {
        let kg = sample_kg();
        let ex = Namespace::new("http://example.org/").unwrap();
        let apple = ex.get("Apple").unwrap();
        let founded = ex.get("founded_by").unwrap();
        let jobs = ex.get("SteveJobs").unwrap();

        assert!(kg.contains(apple, founded, jobs).unwrap());
    }

    #[test]
    fn mutable_graph_insert() {
        let mut kg = KnowledgeGraph::new();
        let ex = Namespace::new("http://example.org/").unwrap();
        let a = ex.get("A").unwrap();
        let rel = ex.get("rel").unwrap();
        let b = ex.get("B").unwrap();

        let inserted = MutableGraph::insert(&mut kg, a, rel, b).unwrap();
        assert!(inserted);
        assert_eq!(kg.triple_count(), 1);

        // Verify roundtrip: the triple should be findable via Graph::contains
        assert!(kg.contains(a, rel, b).unwrap());
    }

    #[test]
    fn mutable_graph_remove() {
        let mut kg = sample_kg();
        assert_eq!(kg.triple_count(), 3);

        let ex = Namespace::new("http://example.org/").unwrap();
        let apple = ex.get("Apple").unwrap();
        let founded = ex.get("founded_by").unwrap();
        let jobs = ex.get("SteveJobs").unwrap();

        let removed = MutableGraph::remove(&mut kg, apple, founded, jobs).unwrap();
        assert!(removed);
        assert_eq!(kg.triple_count(), 2);
        assert!(!kg.contains(apple, founded, jobs).unwrap());
    }

    #[test]
    fn collectible_graph_from_triples() {
        let source_kg = sample_kg();
        let triples_vec: Vec<[SimpleTerm<'static>; 3]> = SophiaGraph::triples(&source_kg)
            .map(|r| {
                let t = r.unwrap();
                t.map(Term::into_term)
            })
            .collect();

        let rebuilt: KnowledgeGraph =
            CollectibleGraph::from_triple_source(triples_vec.into_iter().into_triple_source())
                .unwrap();
        assert_eq!(rebuilt.triple_count(), 3);
    }

    #[test]
    fn str_to_term_iri() {
        let t = str_to_term("http://example.org/Foo");
        assert_eq!(t.kind(), sophia_api::term::TermKind::Iri);
        assert_eq!(t.iri().unwrap().as_str(), "http://example.org/Foo");
    }

    #[test]
    fn str_to_term_blank_node() {
        let t = str_to_term("_:b0");
        assert_eq!(t.kind(), sophia_api::term::TermKind::BlankNode);
        assert_eq!(t.bnode_id().unwrap().as_str(), "b0");
    }

    #[test]
    fn str_to_term_literal() {
        let t = str_to_term("\"hello\"");
        assert_eq!(t.kind(), sophia_api::term::TermKind::Literal);
        assert_eq!(t.lexical_form().unwrap().as_ref(), "hello");
    }

    #[test]
    fn str_to_term_literal_lang() {
        let t = str_to_term("\"hello\"@en");
        assert_eq!(t.kind(), sophia_api::term::TermKind::Literal);
        assert_eq!(t.lexical_form().unwrap().as_ref(), "hello");
        assert_eq!(t.language_tag().unwrap().as_str(), "en");
    }

    #[test]
    fn str_to_term_literal_datatype() {
        let t = str_to_term("\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>");
        assert_eq!(t.kind(), sophia_api::term::TermKind::Literal);
        assert_eq!(t.lexical_form().unwrap().as_ref(), "42");
        assert_eq!(
            t.datatype().unwrap().as_str(),
            "http://www.w3.org/2001/XMLSchema#integer"
        );
    }

    #[test]
    fn term_to_string_roundtrip_iri() {
        let original = "http://example.org/Foo";
        let term = str_to_term(original);
        let back = term_to_string(term);
        assert_eq!(back, original);
    }

    #[test]
    fn term_to_string_roundtrip_bnode() {
        let original = "_:b42";
        let term = str_to_term(original);
        let back = term_to_string(term);
        assert_eq!(back, original);
    }

    #[test]
    fn graph_with_blank_nodes() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(LxTriple::new(
            "_:node1",
            "http://example.org/rel",
            "_:node2",
        ));

        let triples: Vec<_> = SophiaGraph::triples(&kg)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(triples.len(), 1);
        let [s, _p, o] = &triples[0];
        assert_eq!(s.kind(), sophia_api::term::TermKind::BlankNode);
        assert_eq!(o.kind(), sophia_api::term::TermKind::BlankNode);
    }
}
