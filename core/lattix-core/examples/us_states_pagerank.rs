//! US States PageRank - Real Geography Network
//!
//! Computes PageRank on the US state adjacency graph.
//! States that border many other states get higher centrality.
//!
//! Fun fact: Missouri and Tennessee are the most "central" states
//! by this measure - they each border 8 other states!
//!
//! ```bash
//! cargo run --example us_states_pagerank --features algo
//! ```

use lattix_core::algo::pagerank::{pagerank, PageRankConfig};
use lattix_core::KnowledgeGraph;
use lattix_core::Triple;

fn main() {
    println!("US States PageRank (Real Geographic Network)");
    println!("=============================================\n");

    let (kg, state_count) = build_us_states_graph();
    println!(
        "Network: {} states, undirected adjacency graph",
        state_count
    );

    // Expected top states by degree (number of borders):
    // Missouri: 8, Tennessee: 8, Kentucky: 7, Colorado: 7
    println!("Expected: Missouri & Tennessee border most states (8 each)\n");

    // Compute PageRank
    let config = PageRankConfig {
        damping_factor: 0.85,
        max_iterations: 100,
        tolerance: 1e-8,
    };

    let scores = pagerank(&kg, config);

    // Sort by score
    let mut sorted: Vec<_> = scores.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("{:20} {:>10} {:>8}", "State", "PageRank", "Borders");
    println!("{}", "-".repeat(42));

    // Print top 15
    let borders = get_border_counts();
    for (state, score) in sorted.iter().take(15) {
        let border_count = borders.get(state.as_str()).unwrap_or(&0);
        println!("{:20} {:>10.4} {:>8}", state, score, border_count);
    }

    println!("\n--- Analysis ---");

    // Compute correlation between PageRank and degree
    let pr_vec: Vec<f64> = sorted.iter().map(|(_, s)| **s).collect();
    let deg_vec: Vec<f64> = sorted
        .iter()
        .map(|(s, _)| *borders.get(s.as_str()).unwrap_or(&0) as f64)
        .collect();
    let corr = pearson_correlation(&pr_vec, &deg_vec);
    println!("Correlation(PageRank, Degree): {:.3}", corr);

    // Geographic interpretation
    println!("\nInterpretation:");
    println!("- High PageRank = central geographic position (many neighbors)");
    println!("- Missouri/Tennessee dominate: 8 borders each, central US");
    println!("- Alaska/Hawaii would be isolated (degree 0) - excluded here");
    println!("- Coastal states (CA, FL, ME) have fewer neighbors");
}

/// Build undirected adjacency graph of US state borders
fn build_us_states_graph() -> (KnowledgeGraph, usize) {
    // US state adjacencies (contiguous 48 states)
    // Source: US Census Bureau
    let adjacencies = [
        (
            "Alabama",
            &["Florida", "Georgia", "Mississippi", "Tennessee"][..],
        ),
        (
            "Arizona",
            &["California", "Colorado", "Nevada", "New Mexico", "Utah"],
        ),
        (
            "Arkansas",
            &[
                "Louisiana",
                "Mississippi",
                "Missouri",
                "Oklahoma",
                "Tennessee",
                "Texas",
            ],
        ),
        ("California", &["Arizona", "Nevada", "Oregon"]),
        (
            "Colorado",
            &[
                "Arizona",
                "Kansas",
                "Nebraska",
                "New Mexico",
                "Oklahoma",
                "Utah",
                "Wyoming",
            ],
        ),
        (
            "Connecticut",
            &["Massachusetts", "New York", "Rhode Island"],
        ),
        ("Delaware", &["Maryland", "New Jersey", "Pennsylvania"]),
        ("Florida", &["Alabama", "Georgia"]),
        (
            "Georgia",
            &[
                "Alabama",
                "Florida",
                "North Carolina",
                "South Carolina",
                "Tennessee",
            ],
        ),
        (
            "Idaho",
            &[
                "Montana",
                "Nevada",
                "Oregon",
                "Utah",
                "Washington",
                "Wyoming",
            ],
        ),
        (
            "Illinois",
            &["Indiana", "Iowa", "Kentucky", "Missouri", "Wisconsin"],
        ),
        ("Indiana", &["Illinois", "Kentucky", "Michigan", "Ohio"]),
        (
            "Iowa",
            &[
                "Illinois",
                "Minnesota",
                "Missouri",
                "Nebraska",
                "South Dakota",
                "Wisconsin",
            ],
        ),
        ("Kansas", &["Colorado", "Missouri", "Nebraska", "Oklahoma"]),
        (
            "Kentucky",
            &[
                "Illinois",
                "Indiana",
                "Missouri",
                "Ohio",
                "Tennessee",
                "Virginia",
                "West Virginia",
            ],
        ),
        ("Louisiana", &["Arkansas", "Mississippi", "Texas"]),
        ("Maine", &["New Hampshire"]),
        (
            "Maryland",
            &["Delaware", "Pennsylvania", "Virginia", "West Virginia"],
        ),
        (
            "Massachusetts",
            &[
                "Connecticut",
                "New Hampshire",
                "New York",
                "Rhode Island",
                "Vermont",
            ],
        ),
        ("Michigan", &["Indiana", "Ohio", "Wisconsin"]),
        (
            "Minnesota",
            &["Iowa", "North Dakota", "South Dakota", "Wisconsin"],
        ),
        (
            "Mississippi",
            &["Alabama", "Arkansas", "Louisiana", "Tennessee"],
        ),
        (
            "Missouri",
            &[
                "Arkansas",
                "Illinois",
                "Iowa",
                "Kansas",
                "Kentucky",
                "Nebraska",
                "Oklahoma",
                "Tennessee",
            ],
        ),
        (
            "Montana",
            &["Idaho", "North Dakota", "South Dakota", "Wyoming"],
        ),
        (
            "Nebraska",
            &[
                "Colorado",
                "Iowa",
                "Kansas",
                "Missouri",
                "South Dakota",
                "Wyoming",
            ],
        ),
        (
            "Nevada",
            &["Arizona", "California", "Idaho", "Oregon", "Utah"],
        ),
        ("New Hampshire", &["Maine", "Massachusetts", "Vermont"]),
        ("New Jersey", &["Delaware", "New York", "Pennsylvania"]),
        (
            "New Mexico",
            &["Arizona", "Colorado", "Oklahoma", "Texas", "Utah"],
        ),
        (
            "New York",
            &[
                "Connecticut",
                "Massachusetts",
                "New Jersey",
                "Pennsylvania",
                "Vermont",
            ],
        ),
        (
            "North Carolina",
            &["Georgia", "South Carolina", "Tennessee", "Virginia"],
        ),
        ("North Dakota", &["Minnesota", "Montana", "South Dakota"]),
        (
            "Ohio",
            &[
                "Indiana",
                "Kentucky",
                "Michigan",
                "Pennsylvania",
                "West Virginia",
            ],
        ),
        (
            "Oklahoma",
            &[
                "Arkansas",
                "Colorado",
                "Kansas",
                "Missouri",
                "New Mexico",
                "Texas",
            ],
        ),
        ("Oregon", &["California", "Idaho", "Nevada", "Washington"]),
        (
            "Pennsylvania",
            &[
                "Delaware",
                "Maryland",
                "New Jersey",
                "New York",
                "Ohio",
                "West Virginia",
            ],
        ),
        ("Rhode Island", &["Connecticut", "Massachusetts"]),
        ("South Carolina", &["Georgia", "North Carolina"]),
        (
            "South Dakota",
            &[
                "Iowa",
                "Minnesota",
                "Montana",
                "Nebraska",
                "North Dakota",
                "Wyoming",
            ],
        ),
        (
            "Tennessee",
            &[
                "Alabama",
                "Arkansas",
                "Georgia",
                "Kentucky",
                "Mississippi",
                "Missouri",
                "North Carolina",
                "Virginia",
            ],
        ),
        (
            "Texas",
            &["Arkansas", "Louisiana", "New Mexico", "Oklahoma"],
        ),
        (
            "Utah",
            &[
                "Arizona",
                "Colorado",
                "Idaho",
                "Nevada",
                "New Mexico",
                "Wyoming",
            ],
        ),
        ("Vermont", &["Massachusetts", "New Hampshire", "New York"]),
        (
            "Virginia",
            &[
                "Kentucky",
                "Maryland",
                "North Carolina",
                "Tennessee",
                "West Virginia",
            ],
        ),
        ("Washington", &["Idaho", "Oregon"]),
        (
            "West Virginia",
            &["Kentucky", "Maryland", "Ohio", "Pennsylvania", "Virginia"],
        ),
        ("Wisconsin", &["Illinois", "Iowa", "Michigan", "Minnesota"]),
        (
            "Wyoming",
            &[
                "Colorado",
                "Idaho",
                "Montana",
                "Nebraska",
                "South Dakota",
                "Utah",
            ],
        ),
    ];

    let mut kg = KnowledgeGraph::new();
    let mut states = std::collections::HashSet::new();

    for (state, neighbors) in &adjacencies {
        states.insert(*state);
        for neighbor in *neighbors {
            states.insert(*neighbor);
            // Add bidirectional edge (undirected graph)
            kg.add_triple(Triple::new(*state, "borders", *neighbor));
        }
    }

    (kg, states.len())
}

fn get_border_counts() -> std::collections::HashMap<&'static str, usize> {
    // Pre-computed for verification
    [
        ("Missouri", 8),
        ("Tennessee", 8),
        ("Kentucky", 7),
        ("Colorado", 7),
        ("Nebraska", 6),
        ("Utah", 6),
        ("Idaho", 6),
        ("Arkansas", 6),
        ("Pennsylvania", 6),
        ("Oklahoma", 6),
        ("New York", 5),
        ("Wyoming", 6),
        ("South Dakota", 6),
        ("Iowa", 6),
        ("Illinois", 5),
        ("Ohio", 5),
        ("Virginia", 5),
        ("West Virginia", 5),
        ("Georgia", 5),
        ("Nevada", 5),
        ("New Mexico", 5),
        ("Massachusetts", 5),
        ("Arizona", 5),
        ("Montana", 4),
        ("Kansas", 4),
        ("Minnesota", 4),
        ("Wisconsin", 4),
        ("Indiana", 4),
        ("Texas", 4),
        ("North Carolina", 4),
        ("Michigan", 3),
        ("Oregon", 4),
        ("Alabama", 4),
        ("Mississippi", 4),
        ("Louisiana", 3),
        ("Maryland", 4),
        ("New Jersey", 3),
        ("Vermont", 3),
        ("Connecticut", 3),
        ("Delaware", 3),
        ("New Hampshire", 3),
        ("North Dakota", 3),
        ("California", 3),
        ("Washington", 2),
        ("South Carolina", 2),
        ("Florida", 2),
        ("Rhode Island", 2),
        ("Maine", 1),
    ]
    .into_iter()
    .collect()
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }

    num / (dx2.sqrt() * dy2.sqrt())
}
