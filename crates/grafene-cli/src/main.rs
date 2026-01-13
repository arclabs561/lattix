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
//! # Generate random walks
//! lattice walks input.nt -o walks.txt --length 80 --num-walks 10
//!
//! # Compute PageRank
//! lattice pagerank input.nt --top 10
//!
//! # Compute Connected Components
//! lattice components input.nt
//!
//! # Binary serialization (fast loading)
//! lattice save input.nt output.bin
//! lattice stats output.bin
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::ProgressBar;
use grafene_core::algo::components::{component_stats, connected_components};
use grafene_core::algo::pagerank::{pagerank, PageRankConfig};
use grafene_core::algo::random_walk::{generate_walks, RandomWalkConfig};
use grafene_core::formats::{Csv, JsonLd, NTriples, Turtle};
use grafene_core::KnowledgeGraph;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "grafene")]
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

    /// Generate random walks (Node2Vec style)
    Walks {
        /// Input file (N-Triples or CSV)
        input: PathBuf,

        /// Output file (one walk per line)
        #[arg(short, long)]
        output: PathBuf,

        /// Walk length
        #[arg(long, default_value = "80")]
        length: usize,

        /// Number of walks per node
        #[arg(long, default_value = "10")]
        num_walks: usize,

        /// Return parameter p
        #[arg(long, default_value = "1.0")]
        p: f32,

        /// In-out parameter q
        #[arg(long, default_value = "1.0")]
        q: f32,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Compute PageRank centrality
    Pagerank {
        /// Input file
        input: PathBuf,

        /// Number of top entities to show
        #[arg(short, long, default_value = "20")]
        top: usize,

        /// Damping factor
        #[arg(long, default_value = "0.85")]
        damping: f64,
    },

    /// Analyze connected components
    Components {
        /// Input file
        input: PathBuf,

        /// Dump all components to stdout
        #[arg(long)]
        verbose: bool,

        /// Use strongly connected components (default: weakly connected)
        #[arg(long)]
        strong: bool,
    },

    /// Save graph to binary format (fast loading)
    Save {
        /// Input file (N-Triples, CSV, etc.)
        input: PathBuf,

        /// Output file (.bin)
        output: PathBuf,
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
        Commands::Walks {
            input,
            output,
            length,
            num_walks,
            p,
            q,
            seed,
        } => cmd_walks(&input, &output, length, num_walks, p, q, seed),
        Commands::Pagerank {
            input,
            top,
            damping,
        } => cmd_pagerank(&input, top, damping),
        Commands::Components {
            input,
            verbose,
            strong,
        } => cmd_components(&input, verbose, strong),
        Commands::Save { input, output } => cmd_save(&input, &output),
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
    let start = Instant::now();
    let pb = ProgressBar::new_spinner();
    pb.set_message(format!("Loading {}...", path.display()));

    let kg = if let Some(ext) = path.extension() {
        if ext == "bin" {
            KnowledgeGraph::from_binary_file(path)
                .with_context(|| format!("Failed to load binary file {}", path.display()))?
        } else if ext == "csv" {
            let file =
                File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
            Csv::read(file).with_context(|| format!("Failed to parse CSV {}", path.display()))?
        } else if ext == "json" {
            KnowledgeGraph::from_json_adjacency_file(path).with_context(|| {
                format!("Failed to parse JSON adjacency list {}", path.display())
            })?
        } else {
            KnowledgeGraph::from_ntriples_file(path)
                .with_context(|| format!("Failed to load {}", path.display()))?
        }
    } else {
        KnowledgeGraph::from_ntriples_file(path)
            .with_context(|| format!("Failed to load {}", path.display()))?
    };

    pb.finish_with_message(format!("Loaded in {:.2?}", start.elapsed()));
    Ok(kg)
}

fn cmd_save(input: &PathBuf, output: &PathBuf) -> Result<()> {
    let kg = load_kg(input)?;

    let start = Instant::now();
    let pb = ProgressBar::new_spinner();
    pb.set_message(format!("Saving to {}...", output.display()));

    kg.to_binary_file(output)?;

    pb.finish_with_message(format!("Saved in {:.2?}", start.elapsed()));
    Ok(())
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

fn cmd_walks(
    input: &PathBuf,
    output: &PathBuf,
    length: usize,
    num_walks: usize,
    p: f32,
    q: f32,
    seed: u64,
) -> Result<()> {
    let kg = load_kg(input)?;

    println!(
        "Generating random walks (l={}, n={}, p={}, q={})...",
        length, num_walks, p, q
    );
    let start = Instant::now();
    let config = RandomWalkConfig {
        walk_length: length,
        num_walks,
        p,
        q,
        seed,
    };

    let walks = generate_walks(&kg, config);
    println!("Generated {} walks in {:.2?}", walks.len(), start.elapsed());

    println!("Writing walks to {}...", output.display());
    let mut file = File::create(output)?;
    for walk in walks {
        writeln!(file, "{}", walk.join(" "))?;
    }

    println!("Done.");
    Ok(())
}

fn cmd_pagerank(input: &PathBuf, top: usize, damping: f64) -> Result<()> {
    let kg = load_kg(input)?;

    println!("Computing PageRank (damping={})...", damping);
    let start = Instant::now();
    let config = PageRankConfig {
        damping_factor: damping,
        ..Default::default()
    };

    let scores = pagerank(&kg, config);
    println!("Computed in {:.2?}", start.elapsed());

    let mut sorted_scores: Vec<_> = scores.into_iter().collect();
    sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top {} entities by PageRank:", top);
    for (i, (entity, score)) in sorted_scores.iter().take(top).enumerate() {
        println!("{}. {} ({:.6})", i + 1, entity, score);
    }

    Ok(())
}

fn cmd_components(input: &PathBuf, verbose: bool, strong: bool) -> Result<()> {
    use grafene_core::algo::components::strongly_connected_components;

    let kg = load_kg(input)?;

    let kind = if strong { "strongly" } else { "weakly" };
    println!("Analyzing {} connected components...", kind);
    let start = Instant::now();
    let components = if strong {
        strongly_connected_components(&kg)
    } else {
        connected_components(&kg)
    };
    println!("Analyzed in {:.2?}", start.elapsed());

    let stats = component_stats(&components);
    println!("Component Statistics");
    println!("====================");
    println!("Number of components: {}", stats.num_components);
    println!("Max component size:   {}", stats.max_component_size);
    println!("Min component size:   {}", stats.min_component_size);
    println!("Avg component size:   {:.2}", stats.avg_component_size);

    if verbose {
        println!("\nComponents:");
        for (i, c) in components.iter().enumerate() {
            println!("  Component {}: {} nodes", i + 1, c.len());
            if c.len() < 10 {
                println!("    {:?}", c);
            } else {
                println!("    (too large to list)");
            }
        }
    }

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
            let from_match = from.as_ref().is_none_or(|f| t.subject.as_str() == f);
            let to_match = to.as_ref().is_none_or(|t_| t.object.as_str() == t_);
            let rel_match = relation
                .as_ref()
                .is_none_or(|r| t.predicate.as_str().contains(r));
            from_match && to_match && rel_match
        })
        .collect();

    println!("Found {} triples:", triples.len());
    for triple in triples.iter().take(100) {
        println!(
            "  {} --[{}]--> {}",
            triple.subject, triple.predicate, triple.object
        );
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
