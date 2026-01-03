use crate::tensor::{core::Tensor, init::rand_norm_tensor, ops::{add, dot}, nn::gelu};

pub struct FeedForward {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let w1 = rand_norm_tensor(vec![d_model, d_ff], 0.0, 0.02);
        let b1 = Tensor::n_tensor(0.0, vec![d_ff]);

        let w2 = rand_norm_tensor(vec![d_ff, d_model], 0.0, 0.02);
        let b2 = Tensor::n_tensor(0.0, vec![d_model]);

        Self { w1, b1, w2, b2 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x is [seq, d_model]
        let h = add(&dot(x, &self.w1), &self.b1, true);
        // Apply GELU Transform
        let h_gelu = gelu(&h);
        
        add(&dot(&h_gelu, &self.w2), &self.b2, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedforward_shapes() {
        let d_model = 64;
        let d_ff = 256;
        let ff = FeedForward::new(d_model, d_ff);

        assert_eq!(ff.w1.shape(), &[d_model, d_ff]);
        assert_eq!(ff.b1.shape(), &[d_ff]);
        assert_eq!(ff.w2.shape(), &[d_ff, d_model]);
        assert_eq!(ff.b2.shape(), &[d_model]);
    }

    #[test]
    fn test_feedforward_forward() {
        let d_model = 64;
        let d_ff = 256;
        let seq_len = 10;
        let ff = FeedForward::new(d_model, d_ff);

        let input = Tensor::n_tensor(1.0, vec![seq_len, d_model]);
        let output = ff.forward(&input);

        assert_eq!(output.shape(), &[seq_len, d_model]);
    }

    #[test]
    fn test_feedforward_values() {
        // Create a deterministic FeedForward layer for testing values
        let d_model = 2;
        let d_ff = 4;
        
        // Manually construct weights to have predictable values
        // w1: all 0.1
        let w1 = Tensor::n_tensor(0.1, vec![d_model, d_ff]);
        // b1: all 0.0
        let b1 = Tensor::n_tensor(0.0, vec![d_ff]);
        // w2: all 0.2
        let w2 = Tensor::n_tensor(0.2, vec![d_ff, d_model]);
        // b2: all 0.0
        let b2 = Tensor::n_tensor(0.0, vec![d_model]);

        let ff = FeedForward { w1, b1, w2, b2 };

        // Input: [1, 2] -> all 1.0
        let input = Tensor::n_tensor(1.0, vec![1, d_model]);
        
        // Expected calculation:
        // 1. First Linear: x * w1 + b1
        //    x = [1.0, 1.0] (1x2)
        //    w1 = [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]] (2x4)
        //    dot(x, w1) = [0.2, 0.2, 0.2, 0.2]
        //    add b1 (0.0) -> [0.2, 0.2, 0.2, 0.2]
        
        // 2. GELU: gelu(0.2)
        //    Using approximation or exact value. 
        //    gelu(0.2) ≈ 0.2 * 0.5 * (1.0 + erf(0.2 / sqrt(2)))
        //    erf(0.1414) ≈ 0.1585
        //    gelu(0.2) ≈ 0.1 * (1.1585) ≈ 0.11585
        //    Let's just check it's positive and transformed.
        
        let output = ff.forward(&input);
        
        // Check shape
        assert_eq!(output.shape(), &[1, d_model]);

        let data = output.data();
        // Verify symmetry: since input and weights are symmetric, output elements should be identical
        assert!((data[0] - data[1]).abs() < 1e-6);
        
        // Verify value is non-zero (transformation happened)
        assert!(data[0] > 0.0);
    }
}
