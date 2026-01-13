//! Graph Neural Network primitives and training.
//!
//! GNNs learn representations of nodes by aggregating information from
//! neighbors. The key insight: **a node is defined by its connections**.
//!
//! ## The Message Passing Framework
//!
//! All GNN layers follow the same pattern ([Gilmer et al. 2017](https://arxiv.org/abs/1704.01212)):
//!
//! ```text
//! h_i^{(l+1)} = UPDATE(h_i^{(l)}, AGGREGATE({MESSAGE(h_j^{(l)}) : j ∈ N(i)}))
//! ```
//!
//! 1. **MESSAGE**: Each neighbor j sends information to node i
//! 2. **AGGREGATE**: Combine messages (sum, mean, max, attention)
//! 3. **UPDATE**: Transform to produce new embedding
//!
//! Different GNN variants differ in how they define these three operations.
//!
//! ## GNN Variants
//!
//! ### GCN (Graph Convolutional Network)
//!
//! [Kipf & Welling 2017](https://arxiv.org/abs/1609.02907) - The simplest GNN.
//!
//! **Idea**: Average neighbor features, weighted by degree.
//!
//! ```text
//! h_i' = σ(W × mean(h_j : j ∈ N(i) ∪ {i}))
//! ```
//!
//! The degree normalization (D^{-1/2} A D^{-1/2}) prevents high-degree nodes
//! from dominating.
//!
//! ### GAT (Graph Attention Network)
//!
//! [Velickovic et al. 2018](https://arxiv.org/abs/1710.10903) - Learn which
//! neighbors matter.
//!
//! **Idea**: Instead of fixed averaging, learn attention weights.
//!
//! ```text
//! α_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))
//! h_i' = σ(Σ α_ij × W h_j)
//! ```
//!
//! Multi-head attention: Run K attention heads in parallel, concatenate.
//!
//! ### GraphSAGE
//!
//! [Hamilton et al. 2017](https://arxiv.org/abs/1706.02216) - Sample and
//! aggregate for scalability.
//!
//! **Idea**: Don't use all neighbors—sample a fixed number. Enables
//! mini-batch training on huge graphs.
//!
//! ```text
//! h_i' = σ(W × CONCAT(h_i, AGG({h_j : j ∈ Sample(N(i))})))
//! ```
//!
//! ## Node2Vec: Graph Embeddings Without Neural Networks
//!
//! [Grover & Leskovec 2016](https://arxiv.org/abs/1607.00653) - Learn
//! embeddings via random walks.
//!
//! **Idea**: Nodes appearing in similar random walk contexts should have
//! similar embeddings (like word2vec for graphs).
//!
//! 1. Generate random walks from each node
//! 2. Treat walks as "sentences", nodes as "words"
//! 3. Train skip-gram to predict context nodes
//!
//! The p and q parameters control walk behavior:
//! - **p** (return): Low p → stay local (BFS-like)
//! - **q** (in-out): Low q → explore outward (DFS-like)
//!
//! ## Contrastive Learning: A Critical Nuance
//!
//! Many GNN methods use contrastive loss (InfoNCE):
//!
//! ```text
//! L = -log[exp(sim(q, k+)/τ) / Σ exp(sim(q, k)/τ)]
//! ```
//!
//! **Temperature τ matters enormously**:
//! - Low τ (0.05-0.1): Sharp distributions, tight clusters
//! - High τ (0.5-1.0): Soft distributions, spread embeddings
//!
//! **Hard negatives are critical**: Random negatives are often too easy.
//! The model learns faster from negatives that are similar but not identical.
//! Mix query with negative: h' = β×q + (1-β)×n, β ∈ (0, 0.5).
//!
//! **In-batch negatives**: Use other samples in the batch as negatives.
//! A batch of N samples provides N-1 negatives per positive, efficiently.
//!
//! ## When to Use What
//!
//! | Method | Best For | Scalability |
//! |--------|----------|-------------|
//! | GCN | Homophilous graphs (similar nodes connect) | Medium |
//! | GAT | Heterogeneous importance of neighbors | Medium |
//! | GraphSAGE | Large graphs, inductive learning | High |
//! | Node2Vec | Unsupervised, no node features | High |
//!
//! ## Modules
//!
//! - [`conv`]: Message-passing layers (GCN, GAT, GraphSAGE)
//! - [`node2vec`]: Skip-gram training for node embeddings
//! - [`kge`]: Knowledge graph embedding scoring (TransE, RotatE)
//! - [`hyperbolic`]: Hyperbolic GNN layers
//!
//! ## Example: GCN Forward Pass
//!
//! ```rust,ignore
//! use nexus_nn::conv::GCNConv;
//! use candle_core::{Device, Tensor};
//!
//! let gcn = GCNConv::new(64, 32, &Device::Cpu)?;
//! let x = Tensor::randn(0., 1., (100, 64), &Device::Cpu)?;
//! let adj = /* normalized adjacency */;
//! let out = gcn.forward(&x, &adj)?;  // (100, 32)
//! ```
//!
//! ## References
//!
//! - Gilmer et al. (2017). "Neural Message Passing for Quantum Chemistry."
//! - Wu et al. (2020). "A Comprehensive Survey on Graph Neural Networks."

pub mod conv;
pub mod error;

#[cfg(feature = "node2vec")]
pub mod node2vec;

#[cfg(feature = "kge")]
pub mod kge;

#[cfg(feature = "hyperbolic")]
pub mod hyperbolic;

#[cfg(feature = "onnx")]
pub mod onnx;

pub mod nn;

pub use error::{Error, Result};
