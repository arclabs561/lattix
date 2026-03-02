//! Shared helpers for converting lattix strings to Rio RDF model types.
//!
//! Used by both the N-Triples and Turtle format modules.

use rio_api::model::{BlankNode, Literal, NamedNode, Subject, Term};

/// Parse a string as a `NamedNode` (IRI). Returns `None` for blank nodes and literals.
pub(crate) fn to_named_node(s: &str) -> Option<NamedNode<'_>> {
    if s.starts_with("_:") || s.starts_with('"') {
        return None;
    }
    if s.starts_with('<') && s.ends_with('>') {
        Some(NamedNode {
            iri: &s[1..s.len() - 1],
        })
    } else {
        Some(NamedNode { iri: s })
    }
}

/// Parse a string as a `Subject` (named node or blank node). Returns `None` for literals.
pub(crate) fn to_subject(s: &str) -> Option<Subject<'_>> {
    if s.starts_with("_:") {
        Some(Subject::BlankNode(BlankNode { id: &s[2..] }))
    } else if s.starts_with('"') {
        None
    } else if s.starts_with('<') && s.ends_with('>') {
        Some(Subject::NamedNode(NamedNode {
            iri: &s[1..s.len() - 1],
        }))
    } else {
        Some(Subject::NamedNode(NamedNode { iri: s }))
    }
}

/// Parse a string as a `Term` (named node, blank node, or literal) in the object position.
pub(crate) fn to_object(s: &str) -> Option<Term<'_>> {
    if s.starts_with("_:") {
        Some(Term::BlankNode(BlankNode { id: &s[2..] }))
    } else if s.starts_with('"') {
        // Simple literal: extract the value between the first and last quote.
        // This handles plain `"value"` and is a best-effort fallback for
        // `"value"^^<type>` or `"value"@lang` (the type/lang suffix is dropped).
        let end = s.rfind('"').filter(|&i| i > 0)?;
        Some(Term::Literal(Literal::Simple {
            value: &s[1..end],
        }))
    } else if s.starts_with('<') && s.ends_with('>') {
        Some(Term::NamedNode(NamedNode {
            iri: &s[1..s.len() - 1],
        }))
    } else {
        Some(Term::NamedNode(NamedNode { iri: s }))
    }
}
