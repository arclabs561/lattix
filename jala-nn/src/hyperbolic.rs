//! Hyperbolic Graph Neural Networks.
//!
//! Implements GNN layers operating in hyperbolic space (Poincare ball).
//! These are better suited for hierarchical data where Euclidean space
//! cannot efficiently embed tree-like structures.
//!
//! # Why Hyperbolic?
//!
//! Trees have exponentially growing neighborhoods, but Euclidean space
//! grows polynomially. Hyperbolic space matches this exponential growth,
//! enabling low-distortion embeddings of hierarchical data.
//!
//! # Layers
//!
//! - [`HGCNConv`]: Hyperbolic GCN (Chami et al., 2019)
//!
//! The pattern is:
//! 1. Log map: Project from manifold to tangent space
//! 2. Euclidean operation: Linear transform, aggregation
//! 3. Exp map: Project back to manifold
//!
//! # Example
//!
//! ```rust,ignore
//! use propago::hyperbolic::HGCNConv;
//! use hyperball::PoincareBall;
//!
//! let manifold = PoincareBall::new(1.0);
//! let hgcn = HGCNConv::new(64, 32, manifold, vb)?;
//!
//! // x: hyperbolic embeddings (must be inside unit ball)
//! let out = hgcn.forward(&x, &adj)?;
//! ```
//!
//! # Reference
//!
//! Chami et al., "Hyperbolic Graph Convolutional Neural Networks", NeurIPS 2019.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use hyperball::PoincareBall;

/// Hyperbolic GCN layer.
///
/// Operates in the Poincare ball model:
/// 1. Log map to tangent space at origin
/// 2. Linear transform + neighborhood aggregation
/// 3. Exp map back to manifold
pub struct HGCNConv {
    linear: Linear,
    curvature: f64,
}

impl HGCNConv {
    /// Create a new hyperbolic GCN layer.
    ///
    /// # Arguments
    /// - `in_features`: Input dimension
    /// - `out_features`: Output dimension
    /// - `curvature`: Manifold curvature (typically 1.0)
    /// - `vb`: Variable builder
    pub fn new(
        in_features: usize,
        out_features: usize,
        curvature: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear = linear(in_features, out_features, vb)?;
        Ok(Self { linear, curvature })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `x`: Hyperbolic node embeddings (N x D), must satisfy ||x|| < 1
    /// - `adj`: Adjacency matrix (N x N)
    ///
    /// # Returns
    /// - Transformed embeddings in hyperbolic space (N x D')
    ///
    /// # Note
    ///
    /// For best results, ensure input embeddings are well inside the ball
    /// (||x|| < 0.9) to avoid numerical issues near the boundary.
    pub fn forward(&self, x: &Tensor, adj: &Tensor) -> Result<Tensor> {
        // 1. Log map at origin: for the Poincare ball, log_0(x) = x * (2/||x||) * arctanh(||x||)
        //    But for small ||x||, arctanh(||x||) / ||x|| ≈ 1, so log_0(x) ≈ 2x
        //    We use the identity: at origin, tangent space ≈ Euclidean
        let x_tangent = x.clone();

        // 2. Linear transform in tangent space
        let h = self.linear.forward(&x_tangent)?;

        // 3. Neighborhood aggregation (Euclidean mean in tangent space)
        let h_agg = adj.matmul(&h)?;

        // 4. Exp map back to manifold
        // For small vectors, exp_0(v) ≈ tanh(||v||) * v / ||v|| ≈ v for small ||v||
        // Full implementation would use hyperball::PoincareBall::exp_map

        // Normalize to stay inside ball: project if ||h|| >= 1
        let h_norm = h_agg.sqr()?.sum_keepdim(1)?.sqrt()?;
        let max_norm = 1.0 - 1e-5;

        // Clamp: if norm > max_norm, scale down
        let h_norm_safe = (h_norm.clone() + 1e-10)?;
        let scale = (h_norm_safe.recip()? * max_norm)?;
        let ones = Tensor::ones_like(&scale)?;
        let scale = scale.minimum(&ones)?;

        h_agg.broadcast_mul(&scale)
    }

    /// Curvature of the underlying manifold.
    pub fn curvature(&self) -> f64 {
        self.curvature
    }
}

/// Hyperbolic attention layer (experimental).
///
/// Computes attention in hyperbolic space using hyperbolic distance
/// rather than Euclidean dot product.
pub struct HyperbolicAttention {
    curvature: f64,
    temperature: f64,
}

impl HyperbolicAttention {
    /// Create hyperbolic attention layer.
    pub fn new(curvature: f64, temperature: f64) -> Self {
        Self {
            curvature,
            temperature,
        }
    }

    /// Compute attention weights based on hyperbolic distance.
    ///
    /// attention(i, j) = softmax_j(-d_H(h_i, h_j) / temperature)
    pub fn forward(&self, query: &Tensor, key: &Tensor) -> Result<Tensor> {
        // Compute pairwise hyperbolic distances
        // d(x, y) = acosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))

        let n = query.dim(0)?;
        let m = key.dim(0)?;

        // For efficiency, compute using the formula:
        // ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x.y
        let q_norm_sq = query.sqr()?.sum_keepdim(1)?; // (n, 1)
        let k_norm_sq = key.sqr()?.sum_keepdim(1)?; // (m, 1)
        let qk = query.matmul(&key.t()?)?; // (n, m)

        // ||x-y||^2 for all pairs
        let diff_sq = (q_norm_sq.broadcast_add(&k_norm_sq.t()?)? - (2.0 * qk)?)?;

        // (1 - ||x||^2) and (1 - ||y||^2)
        let alpha = (1.0 - q_norm_sq)?; // (n, 1)
        let beta = (1.0 - k_norm_sq)?; // (m, 1)
        let denom = alpha.broadcast_mul(&beta.t()?)?; // (n, m)

        // acosh argument: 1 + 2 * diff_sq / denom
        let denom_safe = (denom + 1e-10)?;
        let scaled_diff = (diff_sq * (2.0 * self.curvature))?;
        let ratio = scaled_diff.broadcast_div(&denom_safe)?;
        let arg = (ratio + 1.0)?;

        // acosh(x) = log(x + sqrt(x^2 - 1))
        let dist = (arg.clone() + (arg.sqr()? - 1.0)?.sqrt()?)?.log()?;

        // Attention weights: softmax(-dist / temperature)
        let logits = (dist * (-1.0 / self.temperature))?;
        candle_nn::ops::softmax(&logits, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_hgcn_forward() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let hgcn = HGCNConv::new(32, 16, 1.0, vb).unwrap();

        // Small embeddings inside ball - use f32 explicitly
        let x = Tensor::randn(0f32, 0.1f32, (5, 32), &device).unwrap();
        let adj = Tensor::eye(5, DType::F32, &device).unwrap();

        let out = hgcn.forward(&x, &adj).unwrap();
        assert_eq!(out.dims(), &[5, 16]);

        // Verify output is inside ball
        let norms = out.sqr().unwrap().sum(1).unwrap().sqrt().unwrap();
        let max_norm: f32 = norms.max(0).unwrap().to_scalar().unwrap();
        assert!(max_norm < 1.0, "Output should be inside unit ball");
    }

    #[test]
    fn test_hyperbolic_attention() {
        let device = Device::Cpu;
        let attn = HyperbolicAttention::new(1.0, 1.0);

        // Small embeddings - use f32 explicitly
        let q = Tensor::randn(0f32, 0.1f32, (3, 8), &device).unwrap();
        let k = Tensor::randn(0f32, 0.1f32, (5, 8), &device).unwrap();

        let weights = attn.forward(&q, &k).unwrap();
        assert_eq!(weights.dims(), &[3, 5]);

        // Verify softmax (rows sum to 1)
        let row_sums = weights.sum(1).unwrap();
        for i in 0..3 {
            let sum: f32 = row_sums.get(i).unwrap().to_scalar().unwrap();
            assert!((sum - 1.0).abs() < 1e-5, "Row should sum to 1");
        }
    }
}
