# Comparison With Other Graph Libraries

`lattix` provides knowledge graph data structures, RDF format support, and graph algorithms.

## vs. PecanPy
**URL:** https://github.com/krishnanlab/PecanPy
**Focus:** Fast, parallelized Node2Vec implementation in Python (using Numba/C++).

| Feature | lattix | PecanPy |
|---------|---------|---------|
| **Language** | Rust | Python + Numba/C++ |
| **Parallelism** | Yes (`rayon` work-stealing) | Yes (parallel execution) |
| **Input Formats** | N-Triples, Turtle, CSV, JSON-LD | Edgelist (CSV/TSV), CSR |
| **Memory** | `petgraph` graph storage | CSR/dense modes |
| **Focus** | Knowledge graphs, RDF, graph algorithms | Node2Vec random walks |

**Summary:** `lattix` covers RDF-backed knowledge graphs and graph algorithms. PecanPy is focused on Node2Vec random walks and uses CSR-oriented storage for that workload.

## vs. PyTorch Geometric (PyG)
**URL:** https://github.com/pyg-team/pytorch_geometric
**Focus:** Deep Learning on Graphs (GNNs) in PyTorch.

| Feature | lattix | PyG |
|---------|---------|---------|
| **Role** | Graph data preparation and traversal | Model training and GNN architectures |
| **Embeddings** | Generates random-walk corpora; does not train embeddings | GNN model training |
| **Interface** | Rust graph types | Python tensors and graph types |

**Summary:** `lattix` provides typed graphs and neighbor sampling for data preparation.
Model training and inference belong to the consuming application.

## vs. NetworkX
**URL:** https://networkx.org/
**Focus:** General graph analysis in pure Python.

| Feature | lattix | NetworkX |
|---------|---------|---------|
| **Implementation** | Rust | Python |
| **RDF Support** | Native (`oxttl`/`oxrdf`) | Via plugins (rdflib) |

**Summary:** `lattix` represents labeled triples and reads RDF formats. NetworkX
provides general graph analysis; capacity depends on the graph and workload.

## vs. node2vec-rs
**URL:** https://github.com/GregorLueg/node2vec-rs
**Focus:** Node2Vec training in Rust (using `burn`).

**Summary:** `node2vec-rs` implements embedding training. `lattix` currently focuses on generating random-walk corpora that can be trained by `gensim` or `node2vec-rs`.
