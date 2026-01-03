use crate::{tensor::{core::Tensor, ops::{dot}}};

pub struct MultiHeadAttention {
    num_heads: usize,
    d_model: usize,
    d_k: usize,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let d_k = d_model / num_heads;

        let wq = Tensor::rand_norm(vec![d_model, d_model], 0.0, 0.02);
        let wk = Tensor::rand_norm(vec![d_model, d_model], 0.0, 0.02);
        let wv = Tensor::rand_norm(vec![d_model, d_model], 0.0, 0.02);
        let wo = Tensor::rand_norm(vec![d_model, d_model], 0.0, 0.02);

        Self { num_heads, d_model, d_k, wq, wk, wv, wo}
    }

    pub fn forward(&self, x: Tensor) -> Tensor {
        // PROJECT Q,K,V
        let q = dot(&x, &self.wq);
        let k = dot(&x, &self.wk);
        let v = dot(&x, &self.wv);


        // Split into heads
        
        let seq_len = x.shape()[0];
        //Split q k and v into heads

        let q_heads = q.reshape(vec![seq_len, self.num_heads, self.d_k])
                     .transpose(vec![1, 0, 2]);

        let k_heads = k.reshape(vec![seq_len, self.num_heads, self.d_k])
                     .transpose(vec![1, 0, 2]);

        let v_heads = v.reshape(vec![seq_len, self.num_heads, self.d_k])
                     .transpose(vec![1, 0, 2]);
        let mut out_data: Vec<f32>= Vec::with_capacity(q_heads.data().len());
        for i in 0..self.num_heads{
            let q_h = q_heads.slice_axis(0, i);
            let k_h = k_heads.slice_axis(0, i);
            let v_h = v_heads.slice_axis(0, i);

            let scores_h = dot(&q_h, &k_h.transpose(vec![1, 0])).div_scalar((self.d_k as f32).sqrt());
            let weights_h = &scores_h.softmax_axis(1);
            out_data.extend_from_slice(dot(weights_h, &v_h).data());
        }
            
        // Merge heads

        let stacked = Tensor::new(out_data, q_heads.shape().to_vec());
        
        let pre_merged = stacked.transpose(vec![1, 0, 2]);
        let merged = pre_merged.reshape(vec![seq_len, self.d_model]);


        // Apply W_O
        dot(&merged, &self.wo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multihead_attention_shapes() {
        let d_model = 64;
        let num_heads = 8;
        let seq_len = 10;
        
        let mha = MultiHeadAttention::new(d_model, num_heads);
        
        // Input shape: [seq_len, d_model]
        let input = Tensor::rand_norm(vec![seq_len, d_model], 0.0, 1.0);
        
        let output = mha.forward(input);
        
        assert_eq!(output.shape(), &[seq_len, d_model]);
    }

    #[test]
    fn test_multihead_attention_shapes_small() {
        let d_model = 16;
        let num_heads = 4;
        let seq_len = 5;
        
        let mha = MultiHeadAttention::new(d_model, num_heads);
        
        let input = Tensor::rand_norm(vec![seq_len, d_model], 0.0, 1.0);
        
        let output = mha.forward(input);
        
        assert_eq!(output.shape(), &[seq_len, d_model]);
    }

    #[test]
    fn test_head_merge_logic() {
        // Simulate the logic used in forward() to merge heads
        // num_heads = 2, seq_len = 2, d_k = 2
        // stacked shape: [num_heads, seq_len, d_k] -> [2, 2, 2]
        
        // Head 0 data:
        // S0: [1.0, 2.0]
        // S1: [3.0, 4.0]
        
        // Head 1 data:
        // S0: [5.0, 6.0]
        // S1: [7.0, 8.0]
        
        // In memory (stacked), this is Head 0 then Head 1:
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let stacked = Tensor::new(data, vec![2, 2, 2]);
        
        // Transpose to [seq_len, num_heads, d_k] -> [2, 2, 2]
        let pre_merged = stacked.transpose(vec![1, 0, 2]);
        
        // Expected data after transpose:
        // S0 from H0 ([1, 2]), S0 from H1 ([5, 6])
        // S1 from H0 ([3, 4]), S1 from H1 ([7, 8])
        let expected_transposed = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
        assert_eq!(pre_merged.data(), &expected_transposed);
        
        // Reshape to [seq_len, d_model] where d_model = num_heads * d_k = 4
        let merged = pre_merged.reshape(vec![2, 4]);
        
        assert_eq!(merged.shape(), &[2, 4]);
        assert_eq!(merged.data(), &expected_transposed); // Data shouldn't change on reshape
        
        // Verify rows correspond to concatenated heads for each sequence position
        // Row 0 (S0): [1, 2, 5, 6] -> [H0_S0, H1_S0]
        assert_eq!(merged.row(0), vec![1.0, 2.0, 5.0, 6.0]);
        // Row 1 (S1): [3, 4, 7, 8] -> [H0_S1, H1_S1]
        assert_eq!(merged.row(1), vec![3.0, 4.0, 7.0, 8.0]);
    }
}
