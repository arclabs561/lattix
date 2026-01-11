//! Lattice CLI - Knowledge graph operations from the command line.
//!
//! # Usage
//!
//! ```bash
//! # Import N-Triples and show stats
//! lattice stats input.nt
//!
//! # Convert between formats
//! lattice convert input.nt -o output.ttl --format turtle
//!
//! # Query relations from an entity
//! lattice query input.nt --from "http://example.org/Apple"
//!
//! # Find path between entities
//! lattice path input.nt --from "http://example.org/A" --to "http://example.org/D"
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use lattice_core::formats::{JsonLd, NTriples, Turtle};
use lattice_core::KnowledgeGraph;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "lattice")]
#[command(about = "Knowledge graph CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show statistics about a knowledge graph
    Stats {
        /// Input file (N-Triples format)
        input: PathBuf,
    },

    /// Convert between RDF formats
    Convert {
        /// Input file (N-Triples format)
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Output format
        #[arg(short, long, default_value = "turtle")]
        format: OutputFormat,
    },

    /// Query relations from an entity
    Query {
        /// Input file (N-Triples format)
        input: PathBuf,

        /// Entity to query relations from
        #[arg(long)]
        from: Option<String>,

        /// Entity to query relations to
        #[arg(long)]
        to: Option<String>,

        /// Relation type to filter by
        #[arg(long, short)]
        relation: Option<String>,
    },

    /// Find path between two entities
    Path {
        /// Input file (N-Triples format)
        input: PathBuf,

        /// Starting entity
        #[arg(long)]
        from: String,

        /// Target entity
        #[arg(long)]
        to: String,
    },

    /// List all entities
    Entities {
        /// Input file (N-Triples format)
        input: PathBuf,

        /// Limit number of results
        #[arg(short, long, default_value = "100")]
        limit: usize,
    },

    /// List all relation types
    Relations {
        /// Input file (N-Triples format)
        input: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    /// N-Triples (line-based)
    Ntriples,
    /// Turtle (human-readable)
    Turtle,
    /// JSON-LD (linked data)
    Jsonld,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Stats { input } => cmd_stats(&input),
        Commands::Convert {
            input,
            output,
            format,
        } => cmd_convert(&input, &output, format),
        Commands::Query {
            input,
            from,
            to,
            relation,
        } => cmd_query(&input, from, to, relation),
        Commands::Path { input, from, to } => cmd_path(&input, &from, &to),
        Commands::Entities { input, limit } => cmd_entities(&input, limit),
        Commands::Relations { input } => cmd_relations(&input),
    }
}

fn load_kg(path: &PathBuf) -> Result<KnowledgeGraph> {
    KnowledgeGraph::from_ntriples_file(path)
        .with_context(|| format!("Failed to load {}", path.display()))
}

fn cmd_stats(input: &PathBuf) -> Result<()> {
    let kg = load_kg(input)?;
    let stats = kg.stats();

    println!("Knowledge Graph Statistics");
    println!("==========================");
    println!("Entities:       {}", stats.entity_count);
    println!("Triples:        {}", stats.triple_count);
    println!("Relation types: {}", stats.relation_type_count);
    println!("Avg out-degree: {:.2}", stats.avg_out_degree);

    Ok(())
}

fn cmd_convert(input: &PathBuf, output: &PathBuf, format: OutputFormat) -> Result<()> {
    let kg = load_kg(input)?;

    let content = match format {
        OutputFormat::Ntriples => NTriples::to_string(&kg)?,
        OutputFormat::Turtle => Turtle::to_string(&kg)?,
        OutputFormat::Jsonld => JsonLd::to_string(&kg)?,
    };

    fs::write(output, content).with_context(|| format!("Failed to write {}", output.display()))?;

    println!("Converted {} -> {}", input.display(), output.display());
    Ok(())
}

fn cmd_query(
    input: &PathBuf,
    from: Option<String>,
    to: Option<String>,
    relation: Option<String>,
) -> Result<()> {
    let kg = load_kg(input)?;

    let triples: Vec<_> = kg
        .triples()
        .filter(|t| {
            let from_match = from
                .as_ref()
                .is_none_or(|f| t.subject.as_str() == f);
            let to_match = to.as_ref().is_none_or(|t_| t.object.as_str() == t_);
            let rel_match = relation
                .as_ref()
                .is_none_or(|r| t.predicate.as_str().contains(r));
            from_match && to_match && rel_match
        })
        .collect();

    println!("Found {} triples:", triples.len());
    for triple in triples.iter().take(100) {
        println!("  {} --[{}]--> {}", triple.subject, triple.predicate, triple.object);
    }

    if triples.len() > 100 {
        println!("  ... and {} more", triples.len() - 100);
    }

    Ok(())
}

fn cmd_path(input: &PathBuf, from: &str, to: &str) -> Result<()> {
    let kg = load_kg(input)?;

    match kg.find_path(from, to) {
        Some(path) => {
            println!("Path found ({} hops):", path.len());
            for (i, triple) in path.iter().enumerate() {
                println!(
                    "  {}. {} --[{}]--> {}",
                    i + 1,
                    triple.subject,
                    triple.predicate,
                    triple.object
                );
            }
        }
        None => {
            println!("No path found between {} and {}", from, to);
        }
    }

    Ok(())
}

fn cmd_entities(input: &PathBuf, limit: usize) -> Result<()> {
    let kg = load_kg(input)?;

    println!("Entities ({} total):", kg.entity_count());
    for (i, entity) in kg.entities().take(limit).enumerate() {
        println!("  {}. {}", i + 1, entity.id);
    }

    if kg.entity_count() > limit {
        println!("  ... and {} more", kg.entity_count() - limit);
    }

    Ok(())
}

fn cmd_relations(input: &PathBuf) -> Result<()> {
    let kg = load_kg(input)?;
    let relations = kg.relation_types();

    println!("Relation types ({} total):", relations.len());
    for rel in &relations {
        let count = kg.triples_with_relation(rel.as_str()).len();
        println!("  {} ({})", rel, count);
    }

    Ok(())
}
