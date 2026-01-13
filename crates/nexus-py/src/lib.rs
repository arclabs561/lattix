use nexus_core::algo::pagerank::{pagerank, PageRankConfig};
use nexus_core::algo::random_walk::{generate_walks, RandomWalkConfig};
use nexus_core::algo::sampling::sample_neighbors;
use nexus_core::formats::Csv;
use nexus_core::KnowledgeGraph;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
struct Graph {
    inner: KnowledgeGraph,
}

impl Graph {
    // Internal helper if needed
}

#[pymethods]
impl Graph {
    #[new]
    fn new() -> Self {
        Graph {
            inner: KnowledgeGraph::new(),
        }
    }

    #[staticmethod]
    fn from_ntriples(path: &str) -> PyResult<Self> {
        let kg = KnowledgeGraph::from_ntriples_file(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Graph { inner: kg })
    }

    #[staticmethod]
    fn from_csv(path: &str) -> PyResult<Self> {
        let file = std::fs::File::open(path)?;
        let kg = Csv::read(file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Graph { inner: kg })
    }

    fn num_entities(&self) -> usize {
        self.inner.entity_count()
    }

    fn num_triples(&self) -> usize {
        self.inner.triple_count()
    }

    fn pagerank(&self, damping: f64, max_iters: usize) -> HashMap<String, f64> {
        let config = PageRankConfig {
            damping_factor: damping,
            max_iterations: max_iters,
            tolerance: 1e-6,
        };
        pagerank(&self.inner, config)
    }

    fn random_walks(
        &self,
        num_walks: usize,
        walk_length: usize,
        p: f32,
        q: f32,
        seed: u64,
    ) -> Vec<Vec<String>> {
        let config = RandomWalkConfig {
            num_walks,
            walk_length,
            p,
            q,
            seed,
        };
        generate_walks(&self.inner, config)
    }

    fn sample_neighbors(
        &self,
        nodes: Vec<String>,
        k: usize,
        seed: u64,
    ) -> HashMap<String, Vec<String>> {
        sample_neighbors(&self.inner, &nodes, k, seed)
    }
}

#[pymodule]
fn nexus(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Graph>()?;
    Ok(())
}
