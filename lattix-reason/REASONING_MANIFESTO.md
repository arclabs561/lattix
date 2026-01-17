# Multi-Hop Reasoning in Tekne

## Core Principles

1.  **Logic First, Backend Second**: Reasoning should be expressed in a declarative DSL (`LogicalQuery`) that is independent of whether the underlying graph is a symbolic triple-store, a vector database, or a GNN.
2.  **Geometric Primitives**: Use specialized crates for geometric operations (`subsume` for boxes, `hyp` for hyperbolic). `lattix` provides the representation mapping.
3.  **Inductive Capability**: Favor models that can reason on unseen graphs (ULTRA-style) to enable zero-shot RAG.
4.  **Neuro-Symbolic Hybridization**: Combine the precision of symbolic path-finding with the robustness of neural link prediction.

## Stack Organization

- **`lattix-core`**: The structural source of truth. Handles graph storage and neighborhood sampling.
- **`lattix-kge`**: Learns and stores representations (TransE, BoxE, RotatE).
- **`lattix-nn`**: Implements the neural layers (GCN, CompGCN) needed for reasoning.
- **`lattix-reason`**: The orchestration engine. Executes `LogicalQuery` via symbolic or neural backends.

## Research Connections

- **Query2box / BetaE**: Geometric models for multi-hop logic.
- **ULTRA**: Relation-interaction based inductive reasoning.
- **Scallop**: Soft logic and provenance-based reasoning.
- **GraphRAG**: Large-scale orchestration of graphs and LLMs.
