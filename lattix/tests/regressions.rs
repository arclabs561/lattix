//! Regression tests for fixed bugs. Each test states the correct behavior and
//! the reference it comes from.
#![allow(missing_docs)]

use lattix::{KnowledgeGraph, Triple};

/// Writes `Triple::new(s, p, o)` as N-Triples, reads it back, and requires the
/// same strings.
fn assert_ntriples_roundtrip(s: &str, p: &str, o: &str) {
    let line = Triple::new(s, p, o).to_ntriples();
    let back = Triple::from_ntriples(&line)
        .unwrap_or_else(|e| panic!("written line {line:?} does not parse: {e}"));
    assert_eq!(back.subject().as_str(), s, "subject changed via {line:?}");
    assert_eq!(
        back.predicate().as_str(),
        p,
        "predicate changed via {line:?}"
    );
    assert_eq!(back.object().as_str(), o, "object changed via {line:?}");
}

/// The N-Triples 1.1 IRIREF production excludes spaces, and a `\u0020`
/// escape decodes to a space, which RFC 3987 does not allow in an IRI. A
/// string that is not a valid IRI is a local ID and must round trip through
/// `urn:lattix:local:`.
#[test]
fn ntriples_roundtrip_keeps_iri_like_id_with_space() {
    assert_ntriples_roundtrip("http://example.org/a b", "p", "o");
    assert_ntriples_roundtrip("a", "http://example.org/p q", "o");
    assert_ntriples_roundtrip("a", "p", "http://example.org/a b");
}

/// `<x>` is not an IRIREF (an N-Triples IRI must be absolute), so writing it
/// verbatim produced a line no N-Triples reader accepts.
#[test]
fn ntriples_roundtrip_keeps_bracketed_relative_id() {
    assert_ntriples_roundtrip("<x>", "p", "o");
    assert_ntriples_roundtrip("a", "p", "<x>");
}

/// A blank node label cannot contain a space (N-Triples BLANK_NODE_LABEL), so
/// `_:b 1` is a local ID, not a blank node.
#[test]
fn ntriples_roundtrip_keeps_blank_like_id_with_space() {
    assert_ntriples_roundtrip("_:b 1", "p", "o");
    assert_ntriples_roundtrip("a", "p", "_:b 1");
}

/// Valid RDF syntax still passes through: an absolute IRI is written as
/// itself and a valid blank node label stays a blank node.
#[test]
fn ntriples_writes_valid_iris_and_blank_nodes_verbatim() {
    let line = Triple::new("_:b1", "http://example.org/p", "<http://example.org/o>").to_ntriples();
    assert_eq!(line, "_:b1 <http://example.org/p> <http://example.org/o> .");
}

/// Every triple written by `to_ntriples_file` is read back by the lenient
/// reader, which skips only lines it cannot parse.
#[cfg(feature = "formats")]
#[test]
fn ntriples_file_roundtrip_keeps_all_triples() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("http://example.org/a b", "p", "o"));
    kg.add_triple(Triple::new("<x>", "p", "_:b 1"));
    kg.add_triple(Triple::new("a", "p", "b"));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.nt");
    kg.to_ntriples_file(&path).unwrap();
    let (back, skipped) = KnowledgeGraph::from_ntriples_file_lenient(&path).unwrap();
    assert_eq!(skipped, 0, "lines written by to_ntriples_file were skipped");
    assert_eq!(back.triple_count(), 3);
}

/// `RandomWalkConfig::seed` accepts any `u64`; per-walk seeds derived from it
/// must not overflow.
#[cfg(feature = "algo")]
#[test]
fn random_walk_accepts_max_seed() {
    use lattix::algo::random_walk::{generate_walks, RandomWalkConfig};
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("a", "p", "b"));
    kg.add_triple(Triple::new("b", "p", "a"));
    let cfg = RandomWalkConfig {
        walk_length: 3,
        num_walks: 2,
        p: 1.0,
        q: 1.0,
        seed: u64::MAX,
    };
    assert_eq!(generate_walks(&kg, cfg).len(), 4);
}
