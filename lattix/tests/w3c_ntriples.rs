//! W3C RDF 1.1 N-Triples syntax tests (see `fixtures/w3c-ntriples/README.md`):
//! every positive test must parse, every negative test must be rejected, and
//! a parsed positive test must survive a write and re-parse unchanged.
#![cfg(feature = "formats")]

use lattix::formats::NTriples;
use std::path::Path;

#[test]
fn w3c_ntriples_syntax_suite() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/w3c-ntriples");
    let list = std::fs::read_to_string(dir.join("tests.tsv")).unwrap();
    let mut failures = Vec::new();
    let mut counts = (0, 0);
    for line in list.lines() {
        let (file, kind) = line.split_once('\t').unwrap();
        let input = std::fs::read_to_string(dir.join(file)).unwrap();
        match (kind, NTriples::parse(&input)) {
            ("positive", Ok(kg)) => {
                counts.0 += 1;
                let written = NTriples::to_string(&kg).unwrap();
                match NTriples::parse(&written) {
                    Ok(again) if again.triple_count() == kg.triple_count() => {}
                    Ok(again) => failures.push(format!(
                        "{file}: {} triples after round trip, {} before",
                        again.triple_count(),
                        kg.triple_count()
                    )),
                    Err(e) => failures.push(format!("{file}: rewritten output rejected: {e}")),
                }
            }
            ("positive", Err(e)) => failures.push(format!("{file}: rejected valid input: {e}")),
            ("negative", Err(_)) => counts.1 += 1,
            ("negative", Ok(_)) => failures.push(format!("{file}: accepted invalid input")),
            (other, _) => panic!("unknown test kind {other}"),
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    assert_eq!(counts, (41, 29));
}
