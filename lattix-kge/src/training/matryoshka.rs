/// Matryoshka Representation Learning (MRL) loss wrapper.
///
/// Based on Kusupati et al. (2022) and 2025 temporal extensions.
///
/// This loss encourages the first `k` dimensions of an embedding to be
/// discriminative, allowing for "resizable" vectors.
pub struct MatryoshkaLoss {
    /// Dimension granularities (e.g., [64, 128, 256, 512, 768]).
    pub granularities: Vec<usize>,
    /// Weights for each granularity.
    pub weights: Vec<f32>,
}

impl MatryoshkaLoss {
    pub fn new(granularities: Vec<usize>) -> Self {
        let n = granularities.len();
        Self {
            granularities,
            weights: vec![1.0 / n as f32; n],
        }
    }

    /// Compute total loss as a weighted sum across granularities.
    pub fn compute<F>(&self, mut loss_fn: F) -> f32 
    where F: FnMut(usize) -> f32 {
        let mut total_loss = 0.0;
        for (i, &dim) in self.granularities.iter().enumerate() {
            total_loss += self.weights[i] * loss_fn(dim);
        }
        total_loss
    }
}
