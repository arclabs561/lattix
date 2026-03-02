//! Shared helpers for converting oxrdf types to lattix strings.

/// Convert an oxrdf subject to a lattix string.
pub(crate) fn subject_to_string(s: &oxrdf::NamedOrBlankNode) -> String {
    match s {
        oxrdf::NamedOrBlankNode::NamedNode(n) => n.as_str().to_string(),
        oxrdf::NamedOrBlankNode::BlankNode(b) => format!("_:{}", b.as_str()),
    }
}

/// Convert an oxrdf graph name to a lattix string.
///
/// `DefaultGraph` maps to `None`; named/blank nodes map to `Some(...)`.
pub(crate) fn graph_name_to_string(g: &oxrdf::GraphName) -> Option<String> {
    match g {
        oxrdf::GraphName::NamedNode(n) => Some(n.as_str().to_string()),
        oxrdf::GraphName::BlankNode(b) => Some(format!("_:{}", b.as_str())),
        oxrdf::GraphName::DefaultGraph => None,
    }
}

/// Convert an oxrdf term to a lattix string.
pub(crate) fn term_to_string(t: &oxrdf::Term) -> String {
    match t {
        oxrdf::Term::NamedNode(n) => n.as_str().to_string(),
        oxrdf::Term::BlankNode(b) => format!("_:{}", b.as_str()),
        oxrdf::Term::Literal(l) => {
            if let Some(lang) = l.language() {
                format!("\"{}\"@{}", l.value(), lang)
            } else {
                let dt = l.datatype().as_str();
                if dt == "http://www.w3.org/2001/XMLSchema#string" {
                    format!("\"{}\"", l.value())
                } else {
                    format!("\"{}\"^^<{}>", l.value(), dt)
                }
            }
        }
    }
}
