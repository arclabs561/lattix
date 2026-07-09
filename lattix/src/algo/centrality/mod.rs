//! Centrality algorithms for measuring node importance.
//!
//! # Overview
//!
//! Centrality measures quantify the "importance" of nodes in a graph.
//! Different measures capture different notions of importance:
//!
//! | Algorithm | Question Answered | Complexity |
//! |-----------|-------------------|------------|
//! These complexities include the cost of deduplicating parallel triples into
//! unique neighbor nodes, where `d_max` is the maximum stored degree.
//!
//! | Algorithm | Question Answered | Complexity |
//! |-----------|-------------------|------------|
//! | Degree | How many connections? | O(V + E log d_max) |
//! | Betweenness | How often on shortest paths? | O(VE log d_max) |
//! | Closeness | How close to all others? | O(VE log d_max) |
//! | Eigenvector | Connected to important nodes? | O(E log d_max × iterations) |
//! | Katz | Reachable via damped paths? | O(E log d_max × iterations) |
//! | PageRank | Where do random walks end? | O(E log d_max + E × iterations) |
//! | HITS | Hub or authority? | O(E log d_max × iterations) |
//!
//! # Choosing the Right Measure
//!
//! ```text
//! Want to find...                 Use...
//! ─────────────────────────────────────────────
//! Well-connected nodes            Degree
//! Brokers / bridges               Betweenness
//! Fast information spreaders      Closeness
//! Influential via connections     Eigenvector
//! Important despite isolation     Katz
//! Web page importance             PageRank
//! Content hubs vs authorities     HITS
//! ```
//!
//! # Mathematical Relationships
//!
//! These measures are related. For undirected graphs:
//! - Eigenvector ≈ Katz (as α → 0)
//! - PageRank ≈ Eigenvector (as damping → 1)
//! - Degree is a special case (only 1-hop neighbors)
//!
//! # References
//!
//! - Freeman (1977). "A set of measures of centrality based on betweenness"
//! - Bonacich (1987). "Power and centrality"
//! - Kleinberg (1999). "Authoritative sources in a hyperlinked environment"
//! - Brandes (2001). "A faster algorithm for betweenness centrality"

mod betweenness;
mod closeness;
mod degree;
mod eigenvector;
mod hits;
mod katz;

pub use betweenness::{betweenness_centrality, BetweennessConfig};
pub use closeness::{closeness_centrality, ClosenessConfig};
pub use degree::{
    degree_centrality, in_degree_centrality, out_degree_centrality, DegreeCentrality,
};
pub use eigenvector::{eigenvector_centrality, EigenvectorConfig};
pub use hits::{hits, HitsConfig, HitsScores};
pub use katz::{katz_centrality, KatzConfig};
