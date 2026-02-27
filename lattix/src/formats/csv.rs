//! CSV format support.
//!
//! Expects headerless or headered CSV: `subject,predicate,object` or `subject,object` (predicate default).

use crate::{KnowledgeGraph, Result, Triple};
use std::io::Read;

/// CSV format handler.
pub struct Csv;

impl Csv {
    /// Read triples from CSV.
    ///
    /// If 3 columns: s, p, o
    /// If 2 columns: s, o (uses default_relation)
    /// If 3rd column is float and 2 columns: s, o, weight? (Need specific handling)
    ///
    /// For now, let's support:
    /// - 3 cols: subject, predicate, object
    /// - 2 cols: subject, object (predicate = "related_to")
    pub fn read<R: Read>(reader: R) -> Result<KnowledgeGraph> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false) // Assume no headers for flexibility, or try to detect?
            .from_reader(reader);

        let mut kg = KnowledgeGraph::new();

        for result in reader.records() {
            let record =
                result.map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            if record.len() >= 3 {
                // Check if 3rd col looks like a number (weight) or a string (predicate)
                // This is ambiguous. Let's strict: 3 cols = s, p, o.
                // But DeckSage uses s, o, weight.
                // Maybe we need a specific loader for "Weighted Edgelist".

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
