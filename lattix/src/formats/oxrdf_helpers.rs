//! Shared helpers for converting oxrdf types to lattix strings.

/// Convert an oxrdf subject to a lattix string.
pub(crate) fn subject_to_string(s: &oxrdf::NamedOrBlankNode) -> String {
    crate::rdf::subject_to_string(s)
}

/// Convert an oxrdf graph name to a lattix string.
///
/// `DefaultGraph` maps to `None`; named/blank nodes map to `Some(...)`.
pub(crate) fn graph_name_to_string(g: &oxrdf::GraphName) -> Option<String> {
    crate::rdf::graph_name_to_string(g)
}

/// Convert an oxrdf term to a lattix string.
pub(crate) fn term_to_string(t: &oxrdf::Term) -> String {
    crate::rdf::term_to_string(t)
}
