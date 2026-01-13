# Comparison with Other Graph Libraries

`lattice` is designed as a specialized, high-performance toolkit for Knowledge Graphs in Rust, with a focus on interoperability (RDF, Python) and embedding generation.

## vs. PecanPy
**URL:** https://github.com/krishnanlab/PecanPy
**Focus:** Fast, parallelized Node2Vec implementation in Python (using Numba/C++).

| Feature | lattice | PecanPy |
|---------|---------|---------|
| **Language** | Rust (Native) | Python + Numba/C++ |
| **Parallelism** | Yes (`rayon` work-stealing) | Yes (parallel execution) |
| **Input Formats** | N-Triples, Turtle, CSV, JSON-LD | Edgelist (CSV/TSV), CSR |
| **Memory** | `petgraph` (pointer-based) | Optimized CSR/Dense modes |
| **Focus** | General KG, RDF, Embeddings | Node2Vec/Random Walks Optimization |

**Summary:** `lattice` matches `pecanpy`'s parallel walk generation capability while adding rich RDF support and a broader suite of graph algorithms (PageRank, Components). `pecanpy` likely edges out on raw memory efficiency for massive homogeneous graphs due to specialized CSR implementations, whereas `lattice` prioritizes flexibility and correctness for heterogeneous Knowledge Graphs.

## vs. PyTorch Geometric (PyG)
**URL:** https://github.com/pyg-team/pytorch_geometric
**Focus:** Deep Learning on Graphs (GNNs) in PyTorch.

| Feature | lattice | PyG |
|---------|---------|---------|
| **Role** | Pre-processing / Inference / Embedding Gen | Model Training / GNN Architectures |
| **Embeddings** | Shallow (TransE, Node2Vec) | Deep (GCN, GAT, GraphSAGE) |
| **Inference** | ONNX Runtime (CPU/GPU) | PyTorch (CPU/GPU) |
| **Integration** | `nexus-py` bindings | Native Python |

**Summary:** `lattice` complements PyG.
1.  **Data Loading:** `lattice` can efficiently parse massive RDF datasets, sample neighbors (`lattice.sample_neighbors`), and export edge lists or walks for PyG to consume.
2.  **Inference:** `nexus-embed` allows running trained KGE models (exported to ONNX) in production Rust environments without the heavy PyTorch dependency.

## vs. NetworkX
**URL:** https://networkx.org/
**Focus:** General graph analysis in pure Python.

| Feature | lattice | NetworkX |
|---------|---------|---------|
| **Performance** | High (Rust) | Low (Pure Python) |
| **Scale** | Medium/Large (Millions of nodes) | Small/Medium (<100k nodes) |
| **RDF Support** | Native (Rio) | Via plugins (rdflib) |

**Summary:** `lattice` is a performant alternative to NetworkX for heavy lifting (walks, PageRank) on larger KGs, while exposing a Python API (`nexus-py`) that feels similar to use.

## vs. node2vec-rs
**URL:** https://github.com/GregorLueg/node2vec-rs
**Focus:** Node2Vec training in Rust (using `burn`).

**Summary:** `node2vec-rs` implements the *training* of embeddings in Rust. `lattice` currently focuses on *generating the corpus* (walks) which can be trained by `gensim` or `node2vec-rs`. `lattice` focuses on the graph / data engineering side of the pipeline.
