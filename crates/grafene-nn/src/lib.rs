//! Graph Neural Network primitives and training.
//!
//! `grafene-gnn` provides GNN layers, node embedding training (Node2Vec),
//! and knowledge graph embedding scoring. It sits between the structure
//! layer (`grafene-core`) and application code.
//!
//! # Modules
//!
//! - [`conv`]: Message-passing layers (GCN, GAT, GraphSAGE)
//! - [`node2vec`]: Skip-gram training for node embeddings
//! - [`kge`]: Knowledge graph embedding scoring (TransE, RotatE)
//! - [`hyperbolic`]: Hyperbolic GNN layers (requires `hyperbolic` feature)
//!
//! # Example: GCN Forward Pass
//!
//! ```rust,ignore
//! use grafene_gnn::conv::GCNConv;
//! use candle_core::{Device, Tensor};
//!
//! let gcn = GCNConv::new(64, 32, &Device::Cpu)?;
//! let x = Tensor::randn(0., 1., (100, 64), &Device::Cpu)?;  // 100 nodes, 64 features
//! let adj = /* adjacency matrix */;
//! let out = gcn.forward(&x, &adj)?;  // (100, 32)
//! ```
//!
//! # Example: Node2Vec Training
//!
//! ```rust,ignore
//! use grafene_gnn::node2vec::{Node2VecConfig, Node2VecTrainer};
//!
//! let config = Node2VecConfig {
//!     embedding_dim: 128,
//!     walk_length: 80,
//!     num_walks: 10,
//!     window_size: 5,
//!     p: 1.0,
//!     q: 1.0,
//!     ..Default::default()
//! };
//!
//! let trainer = Node2VecTrainer::new(config, num_nodes);
//! let walks = generate_walks(&graph, &config);  // from grafene-core
//! trainer.train(&walks, epochs)?;
//! let embeddings = trainer.embeddings();
//! ```

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

pub use error::{Error, Result};
