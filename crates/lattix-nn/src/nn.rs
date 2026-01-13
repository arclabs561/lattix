//! Basic neural network building blocks.

// Reserved fields for future features
#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

/// Graph Convolutional Network Layer
pub struct GCNConv {
    lin: Linear,
}

impl GCNConv {
    pub fn forward(&self, x: &Tensor, adj: &Tensor) -> Result<Tensor> {
        // Basic GCN logic placeholder
        // A_hat * X * W
        let x = self.lin.forward(x)?;
        adj.matmul(&x)
    }
}

/// Graph Attention Network Layer
pub struct GATConv {
    lin: Linear,
    // att: Linear...
}
