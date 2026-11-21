//! Basic triple graph queries.
//!
//! Run:
//! `cargo run --example triples`

use lattix::{EntityId, KnowledgeGraph, Triple};
use std::error::Error;
use std::io::{self, Write};
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [] => {
            demo();
            Ok(())
        }
        [file, from, to] => show_file_path(
            Path::new(file),
            from.to_str().ok_or("FROM must be valid UTF-8")?,
            to.to_str().ok_or("TO must be valid UTF-8")?,
            io::stdout().lock(),
        ),
        _ => Err("usage: triples [GRAPH.nt FROM TO]".into()),
    }
}

fn show_file_path(
    file: &Path,
    from: &str,
    to: &str,
    mut out: impl Write,
) -> Result<(), Box<dyn Error>> {
    let kg = KnowledgeGraph::from_ntriples_file_strict(file)
        .map_err(|e| format!("{}: {e}", file.display()))?;
    for id in [from, to] {
        if kg.get_entity(&EntityId::from(id)).is_none() {
            return Err(format!("unknown entity: {id}").into());
        }
    }
    match kg.find_path(from, to) {
        Some(path) => {
            writeln!(out, "path {from} -> {to} ({} hops)", path.len())?;
            for triple in path {
                writeln!(out, "{}", triple.to_ntriples())?;
            }
        }
        None => writeln!(out, "no directed path from {from} to {to}")?,
    }
    Ok(())
}

fn demo() {
    let mut kg = KnowledgeGraph::new();
    kg.add_triple(Triple::new("Apple", "founded_by", "Steve Jobs"));
    kg.add_triple(Triple::new("Apple", "headquartered_in", "Cupertino"));
    kg.add_triple(Triple::new("Steve Jobs", "born_in", "San Francisco"));
    kg.add_triple(Triple::new("Cupertino", "located_in", "California"));
    kg.add_triple(Triple::new("San Francisco", "located_in", "California"));

    println!("entities: {}", kg.entity_count());
    println!("triples:  {}", kg.triple_count());

    println!("\nrelations from Apple:");
    for triple in kg.relations_from("Apple") {
        println!(
            "  {} --{}--> {}",
            triple.subject(),
            triple.predicate(),
            triple.object()
        );
    }

    println!("\nentities connected to California:");
    for triple in kg.relations_to("California") {
        println!("  {} via {}", triple.subject(), triple.predicate());
    }

    let path = kg
        .find_path("Apple", "California")
        .expect("Apple should reach California");
    println!("\npath Apple -> California ({} hops):", path.len());
    for triple in &path {
        println!(
            "  {} --{}--> {}",
            triple.subject(),
            triple.predicate(),
            triple.object()
        );
    }

    assert_eq!(kg.relations_from("Apple").len(), 2);
    assert_eq!(kg.relations_to("California").len(), 2);
    assert_eq!(path.len(), 2);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_directed_paths_from_a_file_and_flags_unknown_ids() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let data = "<urn:a> <urn:uses> <urn:b> .\n<urn:b> <urn:uses> <urn:c> .\n";
        file.write_all(data.as_bytes()).unwrap();
        let inspect = |from, to| -> Result<String, Box<dyn Error>> {
            let mut out = Vec::new();
            show_file_path(file.path(), from, to, &mut out)?;
            Ok(String::from_utf8(out)?)
        };
        assert_eq!(
            inspect("urn:a", "urn:c").unwrap(),
            format!("path urn:a -> urn:c (2 hops)\n{data}")
        );
        assert_eq!(
            inspect("urn:c", "urn:a").unwrap(),
            "no directed path from urn:c to urn:a\n"
        );
        assert_eq!(
            inspect("urn:a", "urn:a").unwrap(),
            "path urn:a -> urn:a (0 hops)\n"
        );
        assert!(inspect("urn:a", "urn:missing")
            .unwrap_err()
            .to_string()
            .contains("unknown entity: urn:missing"));
    }

    #[test]
    fn a_bad_row_is_reported_before_any_path_is_printed() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "<urn:a> <urn:uses> <urn:b> .\nbad row").unwrap();
        let mut out = Vec::new();
        let error = show_file_path(file.path(), "urn:a", "urn:b", &mut out).unwrap_err();
        assert!(error.to_string().contains("line 2"));
        assert!(out.is_empty());
    }
}
