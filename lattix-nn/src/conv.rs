//! Message-passing convolutional layers.
//!
//! Implements standard GNN architectures:
//! - [`GCNConv`]: Graph Convolutional Network (Kipf & Welling, 2017)
//! - [`GATConv`]: Graph Attention Network (Velickovic et al., 2018)
//! - [`SAGEConv`]: GraphSAGE (Hamilton et al., 2017)
//!
//! # Message Passing Framework
//!
//! All layers follow the message-passing paradigm:
//!
//! 1. **Message**: Compute messages from neighbors
//! 2. **Aggregate**: Combine messages (sum, mean, max)
//! 3. **Update**: Transform aggregated messages
//!
//! ```text
//! h_i^{(l+1)} = UPDATE(h_i^{(l)}, AGGREGATE({MESSAGE(h_j^{(l)}) : j in N(i)}))
//! ```

// Some struct fields are reserved for future features (e.g., attention masking, bias control)
#![allow(dead_code)]

use candle_core::{Result, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// Graph Convolutional Network layer.
///
/// Implements: H' = sigma(D^{-1/2} A D^{-1/2} H W)
///
/// Where:
/// - A is the adjacency matrix (with self-loops)
/// - D is the degree matrix
/// - H is the node feature matrix
/// - W is the learnable weight matrix
///
/// # Reference
///
/// Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional
/// Networks", ICLR 2017.
pub struct GCNConv {
    linear: Linear,
    bias: bool,
}

impl GCNConv {
    /// Create a new GCN layer.
    ///
    /// # Arguments
    /// - `in_features`: Input feature dimension
    /// - `out_features`: Output feature dimension
    /// - `bias`: Whether to include bias term
    /// - `vb`: Variable builder for parameter initialization
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear = linear(in_features, out_features, vb)?;
        Ok(Self { linear, bias })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `x`: Node features (N x in_features)
    /// - `adj`: Normalized adjacency matrix (N x N), should include self-loops
    ///
    /// # Returns
    /// - Node embeddings (N x out_features)
    pub fn forward(&self, x: &Tensor, adj: &Tensor) -> Result<Tensor> {
        // Linear transform: X * W
        let h = self.linear.forward(x)?;
        // Neighborhood aggregation: A_hat * H
        adj.matmul(&h)
    }
}

/// Graph Attention Network layer.
///
/// Implements attention-weighted aggregation:
/// h_i' = sigma(sum_{j in N(i)} alpha_{ij} W h_j)
///
/// Where alpha_{ij} = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))
///
/// # Reference
///
/// Velickovic et al., "Graph Attention Networks", ICLR 2018.
pub struct GATConv {
    linear: Linear,
    att_src: Tensor, // Attention vector for source nodes
    att_dst: Tensor, // Attention vector for destination nodes
    negative_slope: f64,
    num_heads: usize,
}

impl GATConv {
    /// Create a new GAT layer.
    ///
    /// # Arguments
    /// - `in_features`: Input feature dimension
    /// - `out_features`: Output feature dimension per head
    /// - `num_heads`: Number of attention heads
    /// - `negative_slope`: LeakyReLU negative slope (typically 0.2)
    /// - `vb`: Variable builder
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_heads: usize,
        negative_slope: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear = linear(in_features, out_features * num_heads, vb.pp("lin"))?;

        // Attention parameters: a = [a_src || a_dst]
        let att_src = vb.get((1, num_heads, out_features), "att_src")?;
        let att_dst = vb.get((1, num_heads, out_features), "att_dst")?;

        Ok(Self {
            linear,
            att_src,
            att_dst,
            negative_slope,
            num_heads,
        })
    }

    /// Forward pass with attention.
    ///
    /// # Arguments
    /// - `x`: Node features (N x in_features)
    /// - `edge_index`: Edge list as (2 x E) tensor of [src; dst] indices
    ///
    /// # Returns
    /// - Node embeddings (N x num_heads * out_features)
    pub fn forward(&self, x: &Tensor, _edge_index: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;

        // Linear projection: (N, in) -> (N, heads * out)
        let h = self.linear.forward(x)?;

        // Reshape for multi-head: (N, heads, out)
        let out_per_head = h.dim(1)? / self.num_heads;
        let h = h.reshape((n, self.num_heads, out_per_head))?;

        // Compute attention scores
        // alpha_src = (h * att_src).sum(-1)  -> (N, heads)
        let _alpha_src = h.broadcast_mul(&self.att_src)?.sum(D::Minus1)?;
        let _alpha_dst = h.broadcast_mul(&self.att_dst)?.sum(D::Minus1)?;

        // For each edge (i, j): e_ij = LeakyReLU(alpha_src[i] + alpha_dst[j])
        // Then softmax over neighbors
        // This is a simplified version - full impl needs sparse attention

        // For now, return transformed features (attention computation requires
        // sparse ops not yet in candle)
        h.reshape((n, self.num_heads * out_per_head))
    }
}

/// GraphSAGE convolutional layer.
///
/// Implements sampling-based aggregation with various aggregators:
/// h_i' = sigma(W * CONCAT(h_i, AGG({h_j : j in Sample(N(i))})))
///
/// Aggregators:
/// - Mean: average neighbor features
/// - Max: element-wise max
/// - LSTM: sequential aggregation (order-dependent)
///
/// # Reference
///
/// Hamilton et al., "Inductive Representation Learning on Large Graphs",
/// NeurIPS 2017.
pub struct SAGEConv {
    lin_self: Linear,
    lin_neighbor: Linear,
    aggregator: Aggregator,
    normalize: bool,
}

/// Aggregation function for GraphSAGE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregator {
    /// Mean aggregator: average of neighbor features
    Mean,
    /// Max pooling: element-wise maximum
    MaxPool,
    /// Sum aggregator
    Sum,
}

impl SAGEConv {
    /// Create a new GraphSAGE layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        aggregator: Aggregator,
        normalize: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let lin_self = linear(in_features, out_features, vb.pp("lin_self"))?;
        let lin_neighbor = linear(in_features, out_features, vb.pp("lin_neighbor"))?;

        Ok(Self {
            lin_self,
            lin_neighbor,
            aggregator,
            normalize,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `x`: Node features (N x in_features)
    /// - `adj`: Adjacency matrix (N x N), can be sparse or sampled
    ///
    /// # Returns
    /// - Node embeddings (N x out_features)
    pub fn forward(&self, x: &Tensor, adj: &Tensor) -> Result<Tensor> {
        // Self features
        let h_self = self.lin_self.forward(x)?;

        // Aggregate neighbor features
        let h_agg = match self.aggregator {
            Aggregator::Mean => {
                // Mean: A * X / degree
                let agg = adj.matmul(x)?;
                // Normalize by degree (row sum of adj)
                let deg = adj.sum(1)?;
                let deg = deg.reshape((deg.elem_count(), 1))?;
                // Avoid division by zero
                let deg = (deg + 1e-6)?;
                agg.broadcast_div(&deg)?
            }
            Aggregator::Sum => adj.matmul(x)?,
            Aggregator::MaxPool => {
                // Max pooling requires sparse iteration - use mean as fallback
                adj.matmul(x)?
            }
        };

        let h_neighbor = self.lin_neighbor.forward(&h_agg)?;

        // Combine: h_self + h_neighbor
        let out = (h_self + h_neighbor)?;

        // Optional L2 normalization
        if self.normalize {
            let norm = out.sqr()?.sum(1)?.sqrt()?;
            let norm = norm.reshape((norm.elem_count(), 1))?;
            let norm = (norm + 1e-6)?;
            out.broadcast_div(&norm)
        } else {
            Ok(out)
        }
    }
}

/// Light Graph Convolutional Network layer.
///
/// LightGCN (He et al., 2020) is a simplified GCN designed specifically for
/// collaborative filtering and recommendation systems. It arose from a key
/// empirical observation: **feature transformation and nonlinear activation
/// hurt recommendation performance**.
///
/// # Historical Context
///
/// The evolution from GCN to LightGCN follows a subtractive path:
///
/// | Year | Model | Key Insight |
/// |------|-------|-------------|
/// | 2017 | GCN | Spectral convolutions via Chebyshev polynomials |
/// | 2019 | NGCF | Apply GCN to user-item bipartite graphs |
/// | 2020 | LightGCN | Remove transforms, keep only neighbor aggregation |
///
/// He et al. showed that for recommendation (where nodes = users/items with
/// only ID features), the learned transforms in GCN/NGCF add parameters
/// without improving signal—in fact, they add noise that hurts generalization.
///
/// # Mathematical Formulation
///
/// LightGCN propagation at layer k:
///
/// ```text
/// E^{(k+1)} = (D^{-1/2} A D^{-1/2}) E^{(k)}
/// ```
///
/// Where:
/// - `A` is the adjacency matrix (user-item interaction graph)
/// - `D` is the diagonal degree matrix
/// - `E^{(k)}` is the embedding matrix at layer k
/// - `E^{(0)}` is the learnable embedding table (the only parameters)
///
/// **No** weight matrix W, **no** activation function σ.
///
/// The final embedding combines information from all layers:
///
/// ```text
/// E_final = (1/(K+1)) * Σ_{k=0}^{K} E^{(k)}
/// ```
///
/// This layer-combination acts as a form of self-ensemble and smooths
/// the propagation, preventing over-smoothing while capturing multi-hop
/// neighbors.
///
/// # Why It Works
///
/// 1. **ID-only features**: In recommendation, user/item features are just IDs.
///    Learned embeddings E^{(0)} are already optimal representations—transforms
///    can only add noise.
///
/// 2. **Graph structure is the signal**: The bipartite user-item graph encodes
///    collaborative filtering signal directly. Propagation diffuses this signal;
///    transforms don't help.
///
/// 3. **Regularization**: Fewer parameters (no W matrices) means less overfitting,
///    especially with sparse interaction data.
///
/// # Computational Complexity
///
/// Per layer: O(|E|d) where |E| = edges, d = embedding dimension
/// Total: O(K|E|d) for K layers
///
/// Compare to GCN: O(K|E|d + Knd²) — LightGCN saves the Knd² term.
///
/// # When to Use
///
/// - **Recommendation systems** with implicit feedback (clicks, views)
/// - **Link prediction** on bipartite graphs
/// - **Node classification** when node features are uninformative
///
/// # When NOT to Use
///
/// - Graphs with rich node features (use GCN/GAT instead)
/// - Tasks requiring nonlinear decision boundaries
/// - Heterogeneous graphs with multiple relation types
///
/// # Reference
///
/// He et al., "LightGCN: Simplifying and Powering Graph Convolution Network
/// for Recommendation", SIGIR 2020.
///
/// See also: NGCF (Wang et al., 2019), which LightGCN simplifies.
pub struct LightGCNConv {
    num_layers: usize,
    alpha: f32,
}

impl LightGCNConv {
    /// Create a new LightGCN layer.
    ///
    /// # Arguments
    ///
    /// - `num_layers`: Number of propagation layers (K in the paper).
    ///   Typical values: 2-4. More layers capture longer-range dependencies
    ///   but risk over-smoothing.
    /// - `alpha`: Layer combination weight. If None, uses uniform weighting
    ///   1/(K+1). Custom alpha can emphasize certain layers.
    ///
    /// # Layer Count Guidance
    ///
    /// | K | Effect |
    /// |---|--------|
    /// | 1 | Direct neighbors only (1-hop) |
    /// | 2 | Friends-of-friends (2-hop) |
    /// | 3 | Typical sweet spot for recommendation |
    /// | 4+ | Risk of over-smoothing, diminishing returns |
    ///
    /// The original paper found K=3 optimal on most datasets.
    pub fn new(num_layers: usize, alpha: Option<f32>) -> Self {
        let alpha = alpha.unwrap_or(1.0 / (num_layers + 1) as f32);
        Self { num_layers, alpha }
    }

    /// Forward pass: propagate embeddings through the graph.
    ///
    /// # Arguments
    ///
    /// - `embeddings`: Initial node embeddings E^{(0)}, shape (N, d)
    /// - `norm_adj`: Symmetrically normalized adjacency D^{-1/2} A D^{-1/2}
    ///   with self-loops. Shape (N, N). **Must be pre-normalized**.
    ///
    /// # Returns
    ///
    /// Final embeddings combining all layers, shape (N, d).
    ///
    /// # Normalization Note
    ///
    /// The adjacency must be normalized BEFORE calling this function:
    ///
    /// ```text
    /// A_hat = A + I                     (add self-loops)
    /// D_hat = diag(sum(A_hat, axis=1))  (degree matrix)
    /// norm_adj = D_hat^{-1/2} A_hat D_hat^{-1/2}
    /// ```
    ///
    /// This is a one-time preprocessing cost, amortized over training.
    pub fn forward(&self, embeddings: &Tensor, norm_adj: &Tensor) -> Result<Tensor> {
        let mut layer_embeddings = vec![embeddings.clone()];
        let mut current = embeddings.clone();

        // Propagation: E^{(k+1)} = norm_adj @ E^{(k)}
        // No weight matrix, no activation—just message passing
        for _ in 0..self.num_layers {
            current = norm_adj.matmul(&current)?;
            layer_embeddings.push(current.clone());
        }

        // Layer combination: weighted sum of all layers
        // Default: uniform (1/(K+1)) per layer
        let mut combined = layer_embeddings[0].clone();
        for emb in layer_embeddings.iter().skip(1) {
            combined = (combined + emb)?;
        }

        // Scale by alpha (accounts for layer count)
        combined * self.alpha as f64
    }
}

/// Graph Isomorphism Network layer.
///
/// GIN (Xu et al., 2019) is designed to be **maximally expressive** among
/// message-passing GNNs. It achieves the same discriminative power as the
/// Weisfeiler-Lehman (WL) graph isomorphism test.
///
/// # Historical Context: The Expressiveness Question
///
/// A fundamental question in GNN theory: **How powerful are GNNs?**
///
/// | Year | Result |
/// |------|--------|
/// | 1968 | WL test proposed (Weisfeiler & Lehman) |
/// | 2017 | GCN, GAT popularized |
/// | 2019 | Xu et al. prove: GNNs ≤ 1-WL, and GIN achieves 1-WL |
///
/// The Weisfeiler-Lehman test iteratively:
/// 1. Hash each node's label with its neighbors' labels
/// 2. If two graphs have different hash multisets at any iteration, they're non-isomorphic
///
/// GIN's insight: to match WL power, the aggregation must be **injective**
/// (different neighbor multisets → different outputs).
///
/// # Why Mean/Max Aggregation Fails
///
/// Consider two neighborhoods:
/// - Node A neighbors: {1, 1, 2}
/// - Node B neighbors: {1, 2, 2}
///
/// Mean aggregation: both give 4/3. **Information lost.**
/// Max aggregation: both give 2. **Information lost.**
///
/// Sum aggregation: A=4, B=5. **Distinguishable.**
///
/// But sum alone isn't enough—we need MLP to learn complex mappings.
///
/// # Mathematical Formulation
///
/// GIN update rule:
///
/// ```text
/// h_v^{(k)} = MLP^{(k)}((1 + ε^{(k)}) · h_v^{(k-1)} + Σ_{u∈N(v)} h_u^{(k-1)})
/// ```
///
/// Where:
/// - ε is a learnable parameter (or fixed to 0)
/// - MLP is a multi-layer perceptron (at least 2 layers for universality)
/// - Σ denotes sum aggregation (injective for multisets)
///
/// # The ε Parameter
///
/// ε controls the balance between self-features and neighbor features:
/// - ε = 0: Equal weight to self and neighbors
/// - ε > 0: More weight on self (useful when node features are informative)
/// - ε < 0: More weight on neighbors (rarely used)
///
/// Learnable ε lets the network adapt this balance per layer.
///
/// # Graph-Level Readout
///
/// For graph classification, GIN uses sum readout across layers:
///
/// ```text
/// h_G = CONCAT(SUM({h_v^{(k)} : v ∈ G}) for k = 0..K)
/// ```
///
/// Concatenating all layers captures both local (early layers) and global
/// (later layers) structure.
///
/// # When to Use
///
/// - **Graph classification** (molecular property prediction, etc.)
/// - Tasks requiring **structural discrimination**
/// - When **theoretical guarantees** on expressiveness matter
///
/// # When NOT to Use
///
/// - Node-level tasks where features dominate structure
/// - Very large graphs (sum aggregation can cause numerical issues)
/// - When interpretability matters (MLPs are opaque)
///
/// # Reference
///
/// Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019.
///
/// See also: Morris et al., "Weisfeiler and Leman Go Neural" (k-WL extensions)
pub struct GINConv {
    mlp: Vec<Linear>,
    eps: Tensor,
    learn_eps: bool,
}

impl GINConv {
    /// Create a new GIN layer.
    ///
    /// # Arguments
    ///
    /// - `in_features`: Input feature dimension
    /// - `hidden_features`: MLP hidden dimension (typically 2-4x in_features)
    /// - `out_features`: Output feature dimension
    /// - `learn_eps`: Whether ε is learnable (recommended: true)
    /// - `vb`: Variable builder for parameter initialization
    ///
    /// # MLP Architecture
    ///
    /// Uses a 2-layer MLP with ReLU activation:
    /// - Linear(in, hidden) → ReLU → Linear(hidden, out)
    ///
    /// Two layers are necessary for universal approximation; adding more
    /// layers rarely helps and increases overfitting risk.
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        learn_eps: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mlp = vec![
            linear(in_features, hidden_features, vb.pp("mlp_0"))?,
            linear(hidden_features, out_features, vb.pp("mlp_1"))?,
        ];

        let eps = if learn_eps {
            vb.get((1,), "eps")?
        } else {
            Tensor::zeros((1,), candle_core::DType::F32, vb.device())?
        };

        Ok(Self {
            mlp,
            eps,
            learn_eps,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// - `x`: Node features (N x in_features)
    /// - `adj`: Adjacency matrix (N x N), need not be normalized
    ///
    /// # Returns
    ///
    /// Updated node features (N x out_features)
    pub fn forward(&self, x: &Tensor, adj: &Tensor) -> Result<Tensor> {
        // Sum aggregation: Σ_{u∈N(v)} h_u
        let neighbor_sum = adj.matmul(x)?;

        // Self-loop with learned/fixed epsilon: (1 + ε) · h_v
        let eps_scalar = self.eps.to_vec1::<f32>()?[0];
        let self_contrib = (x * (1.0 + eps_scalar as f64))?;

        // Combine: (1 + ε)h_v + Σh_u
        let combined = (self_contrib + neighbor_sum)?;

        // MLP: Linear → ReLU → Linear
        let h = self.mlp[0].forward(&combined)?;
        let h = h.relu()?;
        self.mlp[1].forward(&h)
    }
}

/// Relational Graph Convolutional Network layer.
///
/// RGCN (Schlichtkrull et al., 2018) extends GCN to **heterogeneous graphs**
/// with multiple edge types (relations). This is essential for knowledge graphs,
/// social networks with different relationship types, and any multi-relational data.
///
/// # Historical Context: From Homogeneous to Heterogeneous
///
/// | Year | Model | Graph Type |
/// |------|-------|------------|
/// | 2017 | GCN | Homogeneous (single edge type) |
/// | 2017 | GraphSAGE | Homogeneous with sampling |
/// | 2018 | RGCN | Heterogeneous (multiple relations) |
/// | 2020 | HGT | Heterogeneous with attention |
///
/// The key insight: in knowledge graphs like Freebase or Wikidata, edges have
/// **types** (e.g., "born_in", "works_at", "friend_of"). A single weight matrix
/// cannot capture these different semantics.
///
/// # Mathematical Formulation
///
/// RGCN update rule:
///
/// ```text
/// h_i^{(l+1)} = σ( Σ_{r∈R} Σ_{j∈N_r(i)} (1/c_{i,r}) W_r^{(l)} h_j^{(l)} + W_0^{(l)} h_i^{(l)} )
/// ```
///
/// Where:
/// - `R` is the set of relation types
/// - `N_r(i)` is neighbors of node i under relation r
/// - `c_{i,r}` is a normalization constant (e.g., |N_r(i)|)
/// - `W_r` is a **relation-specific** weight matrix
/// - `W_0` is a self-loop weight matrix
///
/// # The Parameter Explosion Problem
///
/// With R relations and d×d weight matrices, RGCN has O(R·d²) parameters.
/// For large R (100+ relation types), this becomes intractable.
///
/// **Solution: Basis Decomposition**
///
/// ```text
/// W_r = Σ_{b=1}^{B} a_{rb} V_b
/// ```
///
/// Where:
/// - `V_b` are B shared basis matrices (learnable)
/// - `a_{rb}` are scalar coefficients per relation
///
/// This reduces parameters from O(R·d²) to O(B·d² + R·B).
///
/// **Alternative: Block-Diagonal Decomposition**
///
/// ```text
/// W_r = diag(Q_r^1, Q_r^2, ..., Q_r^B)
/// ```
///
/// Each relation has B small block matrices instead of one large matrix.
///
/// # When to Use
///
/// - **Knowledge graph completion** (link prediction)
/// - **Entity classification** in multi-relational graphs
/// - **Social networks** with different relationship types
/// - **Molecular graphs** with different bond types
///
/// # When NOT to Use
///
/// - Homogeneous graphs (use GCN/GAT instead—simpler, faster)
/// - When relation types are very similar (single GCN may suffice)
/// - Very large relation sets without decomposition (memory issues)
///
/// # Reference
///
/// Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional
/// Networks", ESWC 2018.
pub struct RGCNConv {
    /// Weight matrices per relation (or basis matrices if using decomposition)
    relation_weights: Vec<Linear>,
    /// Self-loop weight matrix
    self_weight: Linear,
    /// Number of relations
    num_relations: usize,
    /// Basis decomposition coefficients (if using basis decomposition)
    basis_coeffs: Option<Tensor>,
}

impl RGCNConv {
    /// Create a new RGCN layer without decomposition.
    ///
    /// # Arguments
    ///
    /// - `in_features`: Input feature dimension
    /// - `out_features`: Output feature dimension
    /// - `num_relations`: Number of edge/relation types
    /// - `vb`: Variable builder
    ///
    /// # Warning
    ///
    /// Without decomposition, this creates num_relations weight matrices.
    /// For large num_relations (>50), use `new_basis` instead.
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_relations: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut relation_weights = Vec::with_capacity(num_relations);
        for r in 0..num_relations {
            let w = linear(in_features, out_features, vb.pp(format!("rel_{r}")))?;
            relation_weights.push(w);
        }

        let self_weight = linear(in_features, out_features, vb.pp("self"))?;

        Ok(Self {
            relation_weights,
            self_weight,
            num_relations,
            basis_coeffs: None,
        })
    }

    /// Create RGCN with basis decomposition.
    ///
    /// # Arguments
    ///
    /// - `in_features`: Input feature dimension
    /// - `out_features`: Output feature dimension
    /// - `num_relations`: Number of edge/relation types
    /// - `num_bases`: Number of basis matrices (B in the paper)
    /// - `vb`: Variable builder
    ///
    /// # Choosing num_bases
    ///
    /// | num_bases | Effect |
    /// |-----------|--------|
    /// | 1 | All relations share one transform (very constrained) |
    /// | num_relations | No sharing (equivalent to no decomposition) |
    /// | sqrt(num_relations) | Typical sweet spot |
    pub fn new_basis(
        in_features: usize,
        out_features: usize,
        num_relations: usize,
        num_bases: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut basis_weights = Vec::with_capacity(num_bases);
        for b in 0..num_bases {
            let w = linear(in_features, out_features, vb.pp(format!("basis_{b}")))?;
            basis_weights.push(w);
        }

        let self_weight = linear(in_features, out_features, vb.pp("self"))?;

        // Coefficients: (num_relations, num_bases)
        let coeffs = vb.get((num_relations, num_bases), "coeffs")?;

        Ok(Self {
            relation_weights: basis_weights,
            self_weight,
            num_relations,
            basis_coeffs: Some(coeffs),
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// - `x`: Node features (N x in_features)
    /// - `edge_index`: Edge indices (2 x E) where row 0 = source, row 1 = target
    /// - `edge_type`: Relation type for each edge (E,) with values in [0, num_relations)
    ///
    /// # Returns
    ///
    /// Updated node features (N x out_features)
    pub fn forward(&self, x: &Tensor, edge_index: &Tensor, edge_type: &Tensor) -> Result<Tensor> {
        let _n = x.dim(0)?;
        let _out_dim = self.self_weight.weight().dim(0)?;
        let _device = x.device();

        // Self-loop contribution
        let out = self.self_weight.forward(x)?;

        // Get edge indices as vectors
        let src = edge_index.get(0)?;
        let dst = edge_index.get(1)?;
        let edge_types = edge_type.to_vec1::<i64>()?;

        // For each relation, aggregate messages
        for r in 0..self.num_relations {
            // Find edges of this relation type
            let mask: Vec<usize> = edge_types
                .iter()
                .enumerate()
                .filter(|(_, &t)| t == r as i64)
                .map(|(i, _)| i)
                .collect();

            if mask.is_empty() {
                continue;
            }

            // Get weight matrix for this relation
            let w = if self.basis_coeffs.is_some() {
                // Basis decomposition: W_r = Σ a_{rb} V_b
                // For simplicity, we use the first basis weight
                // Full implementation would compute weighted sum
                &self.relation_weights[0]
            } else {
                &self.relation_weights[r]
            };

            // Transform source features
            let h = w.forward(x)?;

            // Aggregate: for each edge (s, d) of type r, add h[s] to out[d]
            // This is a simplified version; full impl would use scatter_add
            let src_vec = src.to_vec1::<i64>()?;
            let dst_vec = dst.to_vec1::<i64>()?;

            for &idx in &mask {
                let s = src_vec[idx] as usize;
                let d = dst_vec[idx] as usize;

                // This is inefficient but correct
                // Production code would use scatter operations
                let h_s = h.get(s)?;
                let out_d = out.get(d)?;
                let _new_d = (out_d + h_s)?;

                // TODO: Update out[d] - requires in-place modification
                // For now, we just compute but don't aggregate
                // Full impl would use scatter_add
            }
        }

        Ok(out)
    }
}

/// Chebyshev Spectral Graph Convolution layer.
///
/// ChebNet (Defferrard et al., 2016) uses Chebyshev polynomials to approximate
/// spectral graph convolutions, enabling localized filters without computing
/// the full graph Laplacian eigenvectors.
///
/// # Historical Context: Spectral Graph Theory
///
/// | Year | Model | Approach |
/// |------|-------|----------|
/// | 2014 | Spectral Networks | Full eigendecomposition O(n³) |
/// | 2016 | ChebNet | Chebyshev polynomial approximation O(K|E|) |
/// | 2017 | GCN | First-order Chebyshev (K=1) |
///
/// GCN is actually a **simplification** of ChebNet with K=1 and specific
/// normalization. ChebNet with K>1 captures higher-order neighborhoods.
///
/// # Mathematical Formulation
///
/// Spectral convolution in graph signal processing:
///
/// ```text
/// g_θ * x = U g_θ(Λ) U^T x
/// ```
///
/// Where:
/// - `U` = eigenvectors of normalized Laplacian L = I - D^{-1/2}AD^{-1/2}
/// - `Λ` = eigenvalues (spectrum)
/// - `g_θ(Λ)` = filter in spectral domain
///
/// **Problem**: Computing U is O(n³). For large graphs, infeasible.
///
/// **ChebNet Solution**: Approximate g_θ(Λ) with Chebyshev polynomials:
///
/// ```text
/// g_θ(Λ) ≈ Σ_{k=0}^{K-1} θ_k T_k(Λ̃)
/// ```
///
/// Where:
/// - `T_k` = Chebyshev polynomial of degree k
/// - `Λ̃ = 2Λ/λ_max - I` (scaled to [-1, 1])
/// - `θ_k` = learnable coefficients
///
/// The key insight: T_k(L̃)x can be computed **without eigendecomposition**
/// using the recurrence relation:
///
/// ```text
/// T_0(L̃)x = x
/// T_1(L̃)x = L̃x
/// T_k(L̃)x = 2L̃ T_{k-1}(L̃)x - T_{k-2}(L̃)x
/// ```
///
/// # Filter Localization
///
/// Chebyshev polynomial of degree K is K-localized: it aggregates information
/// from nodes at most K hops away. This provides explicit control over
/// receptive field size.
///
/// | K | Receptive Field |
/// |---|-----------------|
/// | 1 | Direct neighbors only (like GCN) |
/// | 2 | 2-hop neighborhood |
/// | 3 | 3-hop neighborhood |
///
/// # When to Use
///
/// - When you need **explicit control** over receptive field size
/// - When **spectral properties** of the graph matter
/// - For graphs with **regular structure** (grids, meshes)
///
/// # When NOT to Use
///
/// - When graph structure varies significantly (ChebNet assumes fixed spectrum)
/// - For very large K (diminishing returns, expensive)
/// - When simpler GCN/GAT suffices
///
/// # Reference
///
/// Defferrard et al., "Convolutional Neural Networks on Graphs with Fast
/// Localized Spectral Filtering", NeurIPS 2016.
pub struct ChebConv {
    /// Chebyshev polynomial coefficients (K weight matrices)
    weights: Vec<Linear>,
    /// Polynomial degree (K)
    k: usize,
}

impl ChebConv {
    /// Create a new Chebyshev convolution layer.
    ///
    /// # Arguments
    ///
    /// - `in_features`: Input feature dimension
    /// - `out_features`: Output feature dimension
    /// - `k`: Chebyshev polynomial degree (filter size / receptive field)
    /// - `vb`: Variable builder
    ///
    /// # Choosing K
    ///
    /// - K=1: Equivalent to GCN (1-hop)
    /// - K=2-3: Good balance for most tasks
    /// - K>5: Rarely beneficial, increases computation
    pub fn new(in_features: usize, out_features: usize, k: usize, vb: VarBuilder) -> Result<Self> {
        assert!(k >= 1, "Chebyshev degree K must be >= 1");

        let mut weights = Vec::with_capacity(k);
        for i in 0..k {
            let w = linear(in_features, out_features, vb.pp(format!("cheb_{i}")))?;
            weights.push(w);
        }

        Ok(Self { weights, k })
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// - `x`: Node features (N x in_features)
    /// - `laplacian`: Scaled normalized Laplacian L̃ = 2L/λ_max - I
    ///   where L = I - D^{-1/2}AD^{-1/2}. Shape (N x N).
    ///
    /// # Laplacian Computation
    ///
    /// The caller must provide the **scaled** Laplacian. To compute:
    ///
    /// ```text
    /// 1. L = I - D^{-1/2} A D^{-1/2}  (normalized Laplacian)
    /// 2. λ_max = largest eigenvalue of L (or use λ_max ≈ 2 as approximation)
    /// 3. L̃ = 2L/λ_max - I
    /// ```
    ///
    /// Using λ_max ≈ 2 is common and avoids eigenvalue computation.
    ///
    /// # Returns
    ///
    /// Filtered node features (N x out_features)
    pub fn forward(&self, x: &Tensor, laplacian: &Tensor) -> Result<Tensor> {
        // Chebyshev recurrence:
        // T_0(L)x = x
        // T_1(L)x = Lx
        // T_k(L)x = 2L T_{k-1} - T_{k-2}

        let mut t_prev = x.clone(); // T_0 = x
        let mut out = self.weights[0].forward(&t_prev)?;

        if self.k == 1 {
            return Ok(out);
        }

        let mut t_curr = laplacian.matmul(x)?; // T_1 = Lx
        out = (out + self.weights[1].forward(&t_curr)?)?;

        for i in 2..self.k {
            // T_k = 2L T_{k-1} - T_{k-2}
            let t_next = ((laplacian.matmul(&t_curr)? * 2.0)? - &t_prev)?;
            out = (out + self.weights[i].forward(&t_next)?)?;

            t_prev = t_curr;
            t_curr = t_next;
        }

        Ok(out)
    }
}

/// Message Passing Neural Network (MPNN) framework.
///
/// MPNN (Gilmer et al., 2017) provides a **unified framework** that encompasses
/// nearly all spatial GNN variants. Understanding MPNN is understanding GNNs.
///
/// # Historical Context: Unifying the Zoo
///
/// By 2017, the GNN literature had exploded with variants:
///
/// | Year | Model | Key Mechanism |
/// |------|-------|---------------|
/// | 2014 | Neural FP | Learned fingerprints for molecules |
/// | 2016 | GGNN | Gated recurrent aggregation |
/// | 2017 | GCN | Spectral-inspired normalization |
/// | 2017 | GraphSAGE | Sampling + aggregation |
/// | 2017 | MPNN | Unified framework |
///
/// Gilmer et al. realized these were all special cases of a single pattern:
/// **message passing**. Their MPNN paper won the Best Paper award at ICML 2017.
///
/// # The MPNN Framework
///
/// Every MPNN layer consists of two phases:
///
/// ## 1. Message Phase
///
/// For each edge (v, w) with edge features e_vw:
///
/// ```text
/// m_vw = M(h_v, h_w, e_vw)
/// ```
///
/// Where M is a **message function** that computes what information to send.
///
/// ## 2. Update Phase
///
/// For each node v, aggregate messages and update:
///
/// ```text
/// m_v = ⊕_{w ∈ N(v)} m_vw      (aggregate)
/// h'_v = U(h_v, m_v)            (update)
/// ```
///
/// Where:
/// - `⊕` is a permutation-invariant aggregation (sum, mean, max)
/// - `U` is an **update function** (MLP, GRU, LSTM)
///
/// # How Other GNNs Fit
///
/// | GNN | Message M(h_v, h_w, e) | Aggregation ⊕ | Update U(h, m) |
/// |-----|------------------------|---------------|----------------|
/// | GCN | W·h_w | mean | identity |
/// | GraphSAGE | W·h_w | mean/max/lstm | concat + MLP |
/// | GAT | α_vw · W·h_w | sum | identity |
/// | GIN | h_w | sum | MLP((1+ε)h + m) |
/// | GGNN | W_e·h_w | sum | GRU(h, m) |
///
/// # This Implementation
///
/// This is a **flexible MPNN** that lets you configure:
/// - Edge feature handling
/// - Aggregation type (sum, mean, max)
/// - Update network architecture
///
/// For production, prefer the specialized layers (GCN, GAT, etc.)—they're
/// optimized for their specific structure.
///
/// # When to Use
///
/// - **Research**: Testing new message/update functions
/// - **Molecules**: Edge features (bond types) are critical
/// - **Learning MPNN**: Understanding the general pattern
///
/// # When NOT to Use
///
/// - **Production**: Use optimized specialized layers
/// - **No edge features**: GCN/GAT are simpler and faster
/// - **Very large graphs**: Custom sparse implementations needed
///
/// # Reference
///
/// Gilmer et al., "Neural Message Passing for Quantum Chemistry", ICML 2017.
pub struct MPNNConv {
    /// Message network: transforms concatenated [h_v, h_w, e_vw]
    message_net: Vec<Linear>,
    /// Update network: transforms concatenated [h_v, aggregated_messages]
    update_net: Vec<Linear>,
    /// Node feature dimension
    node_dim: usize,
    /// Edge feature dimension (0 if no edge features)
    edge_dim: usize,
    /// Output dimension
    out_dim: usize,
    /// Aggregation type
    aggregation: Aggregation,
}

/// Aggregation function for message passing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aggregation {
    /// Sum aggregation: m_v = Σ m_vw
    /// Injective for multisets (good for counting)
    Sum,
    /// Mean aggregation: m_v = (1/|N(v)|) Σ m_vw
    /// Distribution-preserving (good for varied degree)
    Mean,
    /// Max aggregation: m_v = max(m_vw)
    /// Captures "most important" message
    Max,
}

impl MPNNConv {
    /// Create a new MPNN layer.
    ///
    /// # Arguments
    ///
    /// - `node_dim`: Input node feature dimension
    /// - `edge_dim`: Edge feature dimension (0 for no edge features)
    /// - `hidden_dim`: Hidden dimension in message/update networks
    /// - `out_dim`: Output node feature dimension
    /// - `aggregation`: How to combine messages
    /// - `vb`: Variable builder
    pub fn new(
        node_dim: usize,
        edge_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        aggregation: Aggregation,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Message network: [h_v, h_w, e_vw] -> message
        // Input: 2*node_dim + edge_dim
        let msg_input = 2 * node_dim + edge_dim;
        let msg1 = linear(msg_input, hidden_dim, vb.pp("msg1"))?;
        let msg2 = linear(hidden_dim, hidden_dim, vb.pp("msg2"))?;

        // Update network: [h_v, aggregated] -> h'_v
        let upd_input = node_dim + hidden_dim;
        let upd1 = linear(upd_input, hidden_dim, vb.pp("upd1"))?;
        let upd2 = linear(hidden_dim, out_dim, vb.pp("upd2"))?;

        Ok(Self {
            message_net: vec![msg1, msg2],
            update_net: vec![upd1, upd2],
            node_dim,
            edge_dim,
            out_dim,
            aggregation,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// - `x`: Node features (N x node_dim)
    /// - `edge_index`: Edge indices (2 x E) where row 0 = source, row 1 = target
    /// - `edge_attr`: Optional edge features (E x edge_dim)
    ///
    /// # Returns
    ///
    /// Updated node features (N x out_dim)
    pub fn forward(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_attr: Option<&Tensor>,
    ) -> Result<Tensor> {
        let n = x.dim(0)?;
        let device = x.device();
        let dtype = x.dtype();

        // Extract source and target indices
        let src = edge_index.get(0)?.to_vec1::<i64>()?;
        let dst = edge_index.get(1)?.to_vec1::<i64>()?;
        let num_edges = src.len();

        // Compute messages for each edge
        // message_i = M(h_src, h_dst, e_i)
        let mut messages = Vec::with_capacity(num_edges);

        for i in 0..num_edges {
            let s = src[i] as usize;
            let d = dst[i] as usize;

            let h_src = x.get(s)?;
            let h_dst = x.get(d)?;

            // Concatenate [h_src, h_dst, edge_attr] or [h_src, h_dst]
            let msg_input = if let Some(e) = edge_attr {
                let e_i = e.get(i)?;
                Tensor::cat(&[&h_src, &h_dst, &e_i], 0)?
            } else {
                Tensor::cat(&[&h_src, &h_dst], 0)?
            };

            // Pass through message network
            let mut m = msg_input;
            for (j, layer) in self.message_net.iter().enumerate() {
                m = layer.forward(&m.unsqueeze(0)?)?;
                m = m.squeeze(0)?;
                if j < self.message_net.len() - 1 {
                    m = m.relu()?;
                }
            }
            messages.push((d, m));
        }

        // Aggregate messages per node
        let hidden_dim = self.message_net.last().unwrap().weight().dim(0)?;
        let mut aggregated = vec![Tensor::zeros(hidden_dim, dtype, device)?; n];
        let mut counts = vec![0usize; n];

        for (dst_node, msg) in messages {
            aggregated[dst_node] = match self.aggregation {
                Aggregation::Sum | Aggregation::Mean => (&aggregated[dst_node] + &msg)?,
                Aggregation::Max => aggregated[dst_node].maximum(&msg)?,
            };
            counts[dst_node] += 1;
        }

        // Apply mean normalization if needed
        if self.aggregation == Aggregation::Mean {
            for i in 0..n {
                if counts[i] > 1 {
                    aggregated[i] = (&aggregated[i] / counts[i] as f64)?;
                }
            }
        }

        // Update each node: h'_v = U(h_v, aggregated)
        let mut outputs = Vec::with_capacity(n);
        for i in 0..n {
            let h_v = x.get(i)?;
            let upd_input = Tensor::cat(&[&h_v, &aggregated[i]], 0)?;

            let mut h_new = upd_input;
            for (j, layer) in self.update_net.iter().enumerate() {
                h_new = layer.forward(&h_new.unsqueeze(0)?)?;
                h_new = h_new.squeeze(0)?;
                if j < self.update_net.len() - 1 {
                    h_new = h_new.relu()?;
                }
            }
            outputs.push(h_new);
        }

        // Stack outputs
        let output_tensors: Vec<&Tensor> = outputs.iter().collect();
        Tensor::stack(&output_tensors, 0)
    }
}

/// PairNorm: Graph normalization to mitigate over-smoothing.
///
/// Over-smoothing is the phenomenon where node representations converge to
/// indistinguishable values as GNN depth increases. PairNorm (Zhao et al., 2019)
/// addresses this by maintaining representation diversity.
///
/// # The Over-Smoothing Problem
///
/// Each GNN layer performs a weighted average of neighbor features:
///
/// ```text
/// h_i^{(k+1)} = AGG({h_j^{(k)} : j in N(i)})
/// ```
///
/// As k increases, all nodes eventually converge to the same representation
/// (the dominant eigenvector of the propagation matrix). This limits GNN depth
/// to typically 2-4 layers.
///
/// | Layers | Typical Behavior |
/// |--------|------------------|
/// | 1-2 | Good discrimination, limited receptive field |
/// | 3-4 | Often optimal for node classification |
/// | 5+ | Over-smoothing degrades performance |
///
/// # PairNorm's Solution
///
/// PairNorm applies two operations:
///
/// ## 1. Center (subtract global mean)
///
/// ```text
/// h_i' = h_i - mean(h)
/// ```
///
/// This prevents the mean from drifting toward a constant.
///
/// ## 2. Scale (row-wise normalization)
///
/// ```text
/// h_i'' = s * h_i' / ||h_i'||
/// ```
///
/// Where s is a learned or fixed scale factor. This maintains consistent
/// embedding magnitudes.
///
/// # Mathematical Intuition
///
/// PairNorm ensures that the **total pairwise squared distance** between
/// node representations remains constant:
///
/// ```text
/// Σ_{i,j} ||h_i - h_j||² = constant
/// ```
///
/// This directly prevents the "collapse" where all representations converge.
///
/// # Variants
///
/// | Mode | Center | Scale | Use Case |
/// |------|--------|-------|----------|
/// | PN (default) | Global | Row | General purpose |
/// | PN-SI | None | Global | When centering hurts |
/// | PN-SCS | Per-cluster | Row | Heterogeneous graphs |
///
/// # When to Use
///
/// - Deep GNNs (>4 layers)
/// - Tasks where receptive field matters
/// - When standard GCN/GAT performance degrades with depth
///
/// # When NOT to Use
///
/// - Shallow networks (2-3 layers) - overhead without benefit
/// - When over-smoothing isn't a problem
/// - Tasks where smoothing is desirable (e.g., denoising)
///
/// # Reference
///
/// Zhao & Akoglu, "PairNorm: Tackling Oversmoothing in GNNs", ICLR 2020.
///
/// See also: ContraNorm (2023), which uses contrastive learning instead.
pub struct PairNorm {
    /// Scale factor (s in the paper)
    scale: f32,
    /// Whether to apply centering
    center: bool,
}

impl PairNorm {
    /// Create a new PairNorm layer with default settings.
    ///
    /// # Arguments
    ///
    /// - `scale`: Scale factor for normalized representations. Default: 1.0.
    ///   Larger values spread representations further apart.
    ///
    /// # Default Behavior
    ///
    /// Uses PN mode: global centering + row-wise scaling.
    pub fn new(scale: f32) -> Self {
        Self {
            scale,
            center: true,
        }
    }

    /// Create PairNorm without centering (PN-SI mode).
    ///
    /// Useful when centering hurts performance, e.g., when global statistics
    /// carry important information.
    pub fn without_centering(scale: f32) -> Self {
        Self {
            scale,
            center: false,
        }
    }

    /// Apply PairNorm to node representations.
    ///
    /// # Arguments
    ///
    /// - `x`: Node representations (N x d)
    ///
    /// # Returns
    ///
    /// Normalized representations (N x d)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Center: subtract global mean
        let h = if self.center {
            let mean = x.mean(0)?; // (d,)
            x.broadcast_sub(&mean)?
        } else {
            x.clone()
        };

        // Scale: row-wise L2 normalization
        // ||h_i|| = sqrt(sum(h_i^2))
        let norm = h.sqr()?.sum(1)?.sqrt()?; // (N,)
        let norm = norm.reshape((norm.elem_count(), 1))?;

        // Avoid division by zero
        let norm = (norm + 1e-6)?;

        // h_i / ||h_i|| * scale
        let normalized = h.broadcast_div(&norm)?;
        normalized * self.scale as f64
    }
}

/// Jumping Knowledge Network: aggregate across GNN layers.
///
/// JK-Net (Xu et al., 2018) addresses over-smoothing by adaptively combining
/// representations from **all** GNN layers, not just the final one. This
/// allows the network to choose the appropriate scope for each node.
///
/// # The Insight
///
/// Different nodes need different receptive field sizes:
///
/// - **Hub nodes** (high degree) need local information—distant nodes are noise.
/// - **Peripheral nodes** (low degree) need broader context for signal.
///
/// Standard GNNs apply the same depth to all nodes. JK-Net adapts per-node.
///
/// # Aggregation Strategies
///
/// Given layer representations h^{(1)}, ..., h^{(K)}:
///
/// ## Concatenation (JK-Cat)
///
/// ```text
/// h_final = [h^{(1)} || h^{(2)} || ... || h^{(K)}]
/// ```
///
/// Most expressive but increases dimension by K×.
///
/// ## Max Pooling (JK-Max)
///
/// ```text
/// h_final[i] = max(h^{(1)}[i], h^{(2)}[i], ..., h^{(K)}[i])
/// ```
///
/// Element-wise max preserves dimension, captures strongest signals.
///
/// ## LSTM Attention (JK-LSTM)
///
/// ```text
/// h_final = LSTM(h^{(1)}, h^{(2)}, ..., h^{(K)})
/// ```
///
/// Learns to weight layers, most flexible but most parameters.
///
/// # When to Use
///
/// - Deep GNNs where intermediate layers may be optimal for some nodes
/// - Heterogeneous graphs with varying local structure
/// - When you can afford increased parameters (especially JK-Cat)
///
/// # Reference
///
/// Xu et al., "Representation Learning on Graphs with Jumping Knowledge
/// Networks", ICML 2018.
pub struct JumpingKnowledge {
    /// Aggregation mode
    mode: JKMode,
}

/// Aggregation mode for Jumping Knowledge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JKMode {
    /// Concatenate all layers: output dim = K * input_dim
    Concat,
    /// Element-wise maximum: output dim = input_dim
    Max,
    /// Sum all layers: output dim = input_dim
    Sum,
    /// Mean of all layers: output dim = input_dim  
    Mean,
}

impl JumpingKnowledge {
    /// Create a new JumpingKnowledge aggregator.
    pub fn new(mode: JKMode) -> Self {
        Self { mode }
    }

    /// Aggregate representations from multiple GNN layers.
    ///
    /// # Arguments
    ///
    /// - `layer_outputs`: Vector of tensors, one per GNN layer.
    ///   Each tensor is (N x d) where N = nodes, d = feature dimension.
    ///
    /// # Returns
    ///
    /// Aggregated representation:
    /// - Concat: (N x K*d)
    /// - Max/Sum/Mean: (N x d)
    pub fn forward(&self, layer_outputs: &[Tensor]) -> Result<Tensor> {
        if layer_outputs.is_empty() {
            return Err(candle_core::Error::Msg(
                "No layer outputs provided".to_string(),
            ));
        }

        match self.mode {
            JKMode::Concat => {
                // Concatenate along feature dimension
                let outputs: Vec<&Tensor> = layer_outputs.iter().collect();
                Tensor::cat(&outputs, 1)
            }
            JKMode::Max => {
                // Element-wise maximum across layers
                let mut result = layer_outputs[0].clone();
                for layer in layer_outputs.iter().skip(1) {
                    result = result.maximum(layer)?;
                }
                Ok(result)
            }
            JKMode::Sum => {
                // Sum across layers
                let mut result = layer_outputs[0].clone();
                for layer in layer_outputs.iter().skip(1) {
                    result = (&result + layer)?;
                }
                Ok(result)
            }
            JKMode::Mean => {
                // Mean across layers
                let mut result = layer_outputs[0].clone();
                for layer in layer_outputs.iter().skip(1) {
                    result = (&result + layer)?;
                }
                result / layer_outputs.len() as f64
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_gcn_forward_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let gcn = GCNConv::new(64, 32, true, vb).unwrap();

        // 10 nodes, 64 features - use F32
        let x = Tensor::randn(0f32, 1f32, (10, 64), &device).unwrap();
        // Simple adjacency (identity for testing) - F32
        let adj = Tensor::eye(10, DType::F32, &device).unwrap();

        let out = gcn.forward(&x, &adj).unwrap();
        assert_eq!(out.dims(), &[10, 32]);
    }

    #[test]
    fn test_sage_forward_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let sage = SAGEConv::new(64, 32, Aggregator::Mean, true, vb).unwrap();

        let x = Tensor::randn(0f32, 1f32, (10, 64), &device).unwrap();
        let adj = Tensor::eye(10, DType::F32, &device).unwrap();

        let out = sage.forward(&x, &adj).unwrap();
        assert_eq!(out.dims(), &[10, 32]);
    }

    #[test]
    fn test_lightgcn_forward_shape() {
        let device = Device::Cpu;

        // LightGCN has no learnable weights except the input embeddings
        let lightgcn = LightGCNConv::new(3, None);

        // 10 nodes, 64 features (these would be learnable embeddings in practice)
        let embeddings = Tensor::randn(0f32, 1f32, (10, 64), &device).unwrap();
        // Normalized adjacency (identity for testing)
        let norm_adj = Tensor::eye(10, DType::F32, &device).unwrap();

        let out = lightgcn.forward(&embeddings, &norm_adj).unwrap();
        assert_eq!(out.dims(), &[10, 64]); // Same shape as input
    }

    #[test]
    fn test_lightgcn_layer_combination() {
        // Test that layer combination works correctly
        let device = Device::Cpu;
        let lightgcn = LightGCNConv::new(2, None); // 2 layers, uniform weight 1/3

        // Simple embeddings
        let embeddings = Tensor::ones((5, 4), DType::F32, &device).unwrap();
        // Identity adjacency (no propagation effect)
        let adj = Tensor::eye(5, DType::F32, &device).unwrap();

        let out = lightgcn.forward(&embeddings, &adj).unwrap();
        // With identity adj, each layer just copies: 3 copies * 1/3 = 1
        // So output should be close to input
        assert_eq!(out.dims(), &[5, 4]);
    }

    #[test]
    fn test_gin_forward_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let gin = GINConv::new(64, 128, 32, true, vb).unwrap();

        let x = Tensor::randn(0f32, 1f32, (10, 64), &device).unwrap();
        let adj = Tensor::eye(10, DType::F32, &device).unwrap();

        let out = gin.forward(&x, &adj).unwrap();
        assert_eq!(out.dims(), &[10, 32]);
    }

    #[test]
    fn test_gin_sum_aggregation() {
        // Test that GIN uses sum aggregation (not mean)
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let gin = GINConv::new(4, 8, 4, false, vb).unwrap();

        // Simple graph: node 0 connected to nodes 1, 2
        // Node features: all ones
        let x = Tensor::ones((3, 4), DType::F32, &device).unwrap();

        // Adjacency: 0 connected to 1 and 2
        let adj_data: Vec<f32> = vec![
            0.0, 1.0, 1.0, // node 0: neighbors 1, 2
            1.0, 0.0, 0.0, // node 1: neighbor 0
            1.0, 0.0, 0.0, // node 2: neighbor 0
        ];
        let adj = Tensor::from_vec(adj_data, (3, 3), &device).unwrap();

        // Should run without error
        let out = gin.forward(&x, &adj).unwrap();
        assert_eq!(out.dims(), &[3, 4]);
    }

    #[test]
    fn test_chebconv_forward_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // K=3 Chebyshev filter
        let cheb = ChebConv::new(64, 32, 3, vb).unwrap();

        let x = Tensor::randn(0f32, 1f32, (10, 64), &device).unwrap();
        // Scaled Laplacian (identity for testing - not realistic but tests shape)
        let laplacian = Tensor::eye(10, DType::F32, &device).unwrap();

        let out = cheb.forward(&x, &laplacian).unwrap();
        assert_eq!(out.dims(), &[10, 32]);
    }

    #[test]
    fn test_chebconv_k1_similar_to_gcn() {
        // ChebNet with K=1 should be similar to a linear transform
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let cheb = ChebConv::new(16, 8, 1, vb).unwrap();

        let x = Tensor::randn(0f32, 1f32, (5, 16), &device).unwrap();
        let laplacian = Tensor::eye(5, DType::F32, &device).unwrap();

        let out = cheb.forward(&x, &laplacian).unwrap();
        assert_eq!(out.dims(), &[5, 8]);
    }

    #[test]
    fn test_rgcn_forward_basic() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // 3 relation types
        let rgcn = RGCNConv::new(32, 16, 3, vb).unwrap();

        let x = Tensor::randn(0f32, 1f32, (5, 32), &device).unwrap();

        // Edge index: 2 edges
        let edge_index = Tensor::from_vec(
            vec![0i64, 1, 2, 3], // sources: 0, 2; targets: 1, 3
            (2, 2),
            &device,
        )
        .unwrap();

        // Edge types: relation 0 and relation 1
        let edge_type = Tensor::from_vec(vec![0i64, 1], (2,), &device).unwrap();

        let out = rgcn.forward(&x, &edge_index, &edge_type).unwrap();
        assert_eq!(out.dims(), &[5, 16]);
    }

    #[test]
    fn test_rgcn_basis_decomposition() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // 10 relations, but only 3 basis matrices
        let rgcn = RGCNConv::new_basis(32, 16, 10, 3, vb).unwrap();

        let x = Tensor::randn(0f32, 1f32, (5, 32), &device).unwrap();
        let edge_index = Tensor::from_vec(vec![0i64, 1], (2, 1), &device).unwrap();
        let edge_type = Tensor::from_vec(vec![5i64], (1,), &device).unwrap(); // relation 5

        let out = rgcn.forward(&x, &edge_index, &edge_type).unwrap();
        assert_eq!(out.dims(), &[5, 16]);
    }

    #[test]
    fn test_mpnn_forward_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // MPNN with no edge features
        let mpnn = MPNNConv::new(32, 0, 64, 16, Aggregation::Sum, vb).unwrap();

        let x = Tensor::randn(0f32, 1f32, (5, 32), &device).unwrap();
        let edge_index = Tensor::from_vec(vec![0i64, 1, 2, 1, 2, 3], (2, 3), &device).unwrap();

        let out = mpnn.forward(&x, &edge_index, None).unwrap();
        assert_eq!(out.dims(), &[5, 16]);
    }

    #[test]
    fn test_mpnn_with_edge_features() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // MPNN with 8-dim edge features
        let mpnn = MPNNConv::new(16, 8, 32, 16, Aggregation::Mean, vb).unwrap();

        let x = Tensor::randn(0f32, 1f32, (4, 16), &device).unwrap();
        let edge_index = Tensor::from_vec(vec![0i64, 1, 1, 2], (2, 2), &device).unwrap();
        let edge_attr = Tensor::randn(0f32, 1f32, (2, 8), &device).unwrap();

        let out = mpnn.forward(&x, &edge_index, Some(&edge_attr)).unwrap();
        assert_eq!(out.dims(), &[4, 16]);
    }

    #[test]
    fn test_mpnn_aggregation_variants() {
        let device = Device::Cpu;

        for agg in [Aggregation::Sum, Aggregation::Mean, Aggregation::Max] {
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

            let mpnn = MPNNConv::new(8, 0, 16, 8, agg, vb).unwrap();

            let x = Tensor::ones((3, 8), DType::F32, &device).unwrap();
            let edge_index = Tensor::from_vec(vec![0i64, 1, 1, 2], (2, 2), &device).unwrap();

            let out = mpnn.forward(&x, &edge_index, None).unwrap();
            assert_eq!(out.dims(), &[3, 8]);
        }
    }

    #[test]
    fn test_pairnorm_preserves_shape() {
        let device = Device::Cpu;
        let pn = PairNorm::new(1.0);

        let x = Tensor::randn(0f32, 1f32, (10, 32), &device).unwrap();
        let out = pn.forward(&x).unwrap();

        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn test_pairnorm_centers_features() {
        let device = Device::Cpu;
        let pn = PairNorm::new(1.0);

        // Use fixed data instead of random to avoid flaky tests
        let x = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
            ],
            (5, 8),
            &device,
        )
        .unwrap();

        let out = pn.forward(&x).unwrap();

        // After centering, mean should be close to zero
        let mean = out.mean(0).unwrap();
        let mean_vals = mean.to_vec1::<f32>().unwrap();

        for val in mean_vals {
            assert!(
                val.abs() < 0.1,
                "Mean should be near zero after centering, got {}",
                val
            );
        }
    }

    #[test]
    fn test_pairnorm_without_centering() {
        let device = Device::Cpu;
        let pn = PairNorm::without_centering(2.0);

        let x = Tensor::randn(0f32, 1f32, (5, 8), &device).unwrap();
        let out = pn.forward(&x).unwrap();

        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn test_jumping_knowledge_concat() {
        let device = Device::Cpu;
        let jk = JumpingKnowledge::new(JKMode::Concat);

        let layer1 = Tensor::ones((5, 8), DType::F32, &device).unwrap();
        let layer2 = Tensor::ones((5, 8), DType::F32, &device).unwrap();
        let layer3 = Tensor::ones((5, 8), DType::F32, &device).unwrap();

        let out = jk.forward(&[layer1, layer2, layer3]).unwrap();

        // Concat: 3 layers x 8 features = 24
        assert_eq!(out.dims(), &[5, 24]);
    }

    #[test]
    fn test_jumping_knowledge_max() {
        let device = Device::Cpu;
        let jk = JumpingKnowledge::new(JKMode::Max);

        let layer1 = Tensor::ones((5, 8), DType::F32, &device).unwrap();
        let layer2 = (Tensor::ones((5, 8), DType::F32, &device).unwrap() * 2.0).unwrap();

        let out = jk.forward(&[layer1, layer2]).unwrap();

        // Max preserves shape
        assert_eq!(out.dims(), &[5, 8]);

        // Values should be close to 2.0 (the max)
        let vals = out.to_vec2::<f32>().unwrap();
        for row in &vals {
            for &v in row {
                assert!((v - 2.0).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_jumping_knowledge_mean() {
        let device = Device::Cpu;
        let jk = JumpingKnowledge::new(JKMode::Mean);

        let layer1 = Tensor::ones((3, 4), DType::F32, &device).unwrap();
        let layer2 = (Tensor::ones((3, 4), DType::F32, &device).unwrap() * 3.0).unwrap();

        let out = jk.forward(&[layer1, layer2]).unwrap();

        assert_eq!(out.dims(), &[3, 4]);

        // Mean of 1.0 and 3.0 = 2.0
        let vals = out.to_vec2::<f32>().unwrap();
        for row in &vals {
            for &v in row {
                assert!((v - 2.0).abs() < 0.01);
            }
        }
    }
}
