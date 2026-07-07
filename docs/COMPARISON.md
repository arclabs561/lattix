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
| **Role** | Pre-processing / Inference / Embedding Gen | Model Training / GNN Architectures |
| **Embeddings** | Shallow (TransE, Node2Vec) | Deep (GCN, GAT, GraphSAGE) |
| **Inference** | ONNX Runtime (CPU/GPU) | PyTorch (CPU/GPU) |
| **Integration** | `lattix-py` bindings | Native Python |

**Summary:** `lattix` can prepare RDF-derived graph data for PyG and can run exported KGE models through ONNX.

## vs. NetworkX
**URL:** https://networkx.org/
**Focus:** General graph analysis in pure Python.

| Feature | lattix | NetworkX |
|---------|---------|---------|
| **Implementation** | Rust | Python |
| **Scale** | Medium/Large (Millions of nodes) | Small/Medium (<100k nodes) |
| **RDF Support** | Native (Rio) | Via plugins (rdflib) |

**Summary:** `lattix` targets larger knowledge graph workloads and exposes Python bindings through `lattix-py`.

## vs. node2vec-rs
**URL:** https://github.com/GregorLueg/node2vec-rs
**Focus:** Node2Vec training in Rust (using `burn`).

**Summary:** `node2vec-rs` implements embedding training. `lattix` currently focuses on generating random-walk corpora that can be trained by `gensim` or `node2vec-rs`.
