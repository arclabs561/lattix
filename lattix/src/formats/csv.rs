//! CSV format support.
//!
//! Reads triples from CSV:
//! - 3 columns: subject, predicate, object
//! - 2 columns: subject, object (predicate defaults to `"related_to"`)
//!
//! Rows with fewer than 2 columns are skipped.

use crate::{KnowledgeGraph, Result, Triple};
use std::io::Read;

/// CSV format handler.
///
/// Reads triples from headerless CSV. Three-column rows map to
/// (subject, predicate, object); two-column rows use `"related_to"`
/// as the default predicate. Rows with fewer than two columns are
/// skipped.
pub struct Csv;

impl Csv {
    /// Read triples from CSV.
    ///
    /// - 3+ columns: columns 0, 1, 2 are subject, predicate, object.
    /// - 2 columns: subject, object (predicate = `"related_to"`).
    /// - <2 columns: row is skipped.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> lattix::Result<()> {
    /// use lattix::formats::csv::Csv;
    ///
    /// let data = "Alice,knows,Bob\nBob,works_at,Acme\n";
    /// let kg = Csv::read(data.as_bytes())?;
    /// assert_eq!(kg.triple_count(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn read<R: Read>(reader: R) -> Result<KnowledgeGraph> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(reader);

        let mut kg = KnowledgeGraph::new();

        for result in reader.records() {
            let record =
                result.map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            if record.len() >= 3 {
                let s = &record[0];
                let p = &record[1];
                let o = &record[2];
                kg.add_triple(Triple::new(s, p, o));
            } else if record.len() == 2 {
                let s = &record[0];
                let o = &record[1];
                kg.add_triple(Triple::new(s, "related_to", o));
            }
        }

        Ok(kg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_three_column_csv() {
        let data = "Alice,knows,Bob\nBob,works_at,Acme\n";
        let kg = Csv::read(data.as_bytes()).unwrap();
        assert_eq!(kg.triple_count(), 2);
        assert_eq!(kg.entity_count(), 3);

        let from_alice = kg.relations_from("Alice");
        assert_eq!(from_alice.len(), 1);
        assert_eq!(from_alice[0].predicate().as_str(), "knows");
    }

    #[test]
    fn test_two_column_csv() {
        let data = "Alice,Bob\nBob,Charlie\n";
        let kg = Csv::read(data.as_bytes()).unwrap();
        assert_eq!(kg.triple_count(), 2);

        let from_alice = kg.relations_from("Alice");
        assert_eq!(from_alice[0].predicate().as_str(), "related_to");
    }

    #[test]
    fn test_empty_csv() {
        let data = "";
        let kg = Csv::read(data.as_bytes()).unwrap();
        assert_eq!(kg.triple_count(), 0);
    }

    #[test]
    fn test_single_column_skipped() {
        let data = "Alice\nBob\n";
        let kg = Csv::read(data.as_bytes()).unwrap();
        assert_eq!(kg.triple_count(), 0);
    }
}
