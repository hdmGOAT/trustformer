use crate::{
    tensor::{core::Tensor, ops::add},
    transformer::{feedforward::FeedForward, layernorm::{LayerNorm}, multihead::MultiHeadAttention},
};

pub struct TransformerBlock {
    ln1: LayerNorm,
    attn: MultiHeadAttention,
    ln2: LayerNorm,
    ffn: FeedForward,
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_heads: usize, ffn_hidden: usize) -> Self {
        Self {
            ln1: LayerNorm::new(d_model),
            attn: MultiHeadAttention::new(d_model, n_heads),
            ln2: LayerNorm::new(d_model),
            ffn: FeedForward::new(d_model, ffn_hidden),
        }
    }


    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x_norm1 = self.ln1.forward(x);

        let attn_out = self.attn.forward(&x_norm1);

        let x1 = add(x, &attn_out, false);

        let x_norm2 = self.ln2.forward(&x1);

        let ffn_out = self.ffn.forward(&x_norm2);

        add(&x1, &ffn_out, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::core::Tensor;

    const EPS: f32 = 1e-6;

    #[test]
    fn transformerblock_output_shape() {
        let d_model = 4;
        let n_heads = 2;
        let ffn_hidden = 8;

        let block = TransformerBlock::new(d_model, n_heads, ffn_hidden);

        let x = Tensor::new(vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        ], vec![3, d_model]);

        let y = block.forward(&x);

        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn transformerblock_residual_finiteness() {
        let d_model = 4;
        let n_heads = 2;
        let ffn_hidden = 8;

        let block = TransformerBlock::new(d_model, n_heads, ffn_hidden);

        let x = Tensor::new(vec![0.5; 12], vec![3, d_model]);

        let y = block.forward(&x);

        assert!(y.is_finite());
    }

    #[test]
    fn transformerblock_zero_input() {
        let d_model = 4;
        let n_heads = 2;
        let ffn_hidden = 8;

        let block = TransformerBlock::new(d_model, n_heads, ffn_hidden);

        let x = Tensor::new(vec![0.0; 12], vec![3, d_model]);
        let y = block.forward(&x);

        assert_eq!(y.shape(), x.shape());

        for &val in y.data() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn transformerblock_small_sequence() {
        let d_model = 2;
        let n_heads = 1;
        let ffn_hidden = 4;

        let block = TransformerBlock::new(d_model, n_heads, ffn_hidden);

        let x = Tensor::new(vec![1.0, 2.0], vec![1, d_model]);
        let y = block.forward(&x);

        assert_eq!(y.shape(), x.shape());
        assert!(y.is_finite());
    }

    #[test]
    fn test_transformer_block_forward_shape() {
        let d_model = 8;
        let n_heads = 2;
        let ffn_hidden = 16;

        let block = TransformerBlock::new(d_model, n_heads, ffn_hidden);

        // Input tensor: sequence length 4, embedding dim d_model
        let x = Tensor::rand_uni(vec![4, d_model], 0.0, 1.0);

        let out = block.forward(&x);

        // Output should have the same shape as input (residual connection)
        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn test_transformer_block_forward_values_finite() {
        let d_model = 8;
        let n_heads = 2;
        let ffn_hidden = 16;

        let block = TransformerBlock::new(d_model, n_heads, ffn_hidden);

        let x = Tensor::rand_uni(vec![4, d_model], -1.0, 1.0);
        let out = block.forward(&x);

        // All output values should be finite
        assert!(out.is_finite());
    }

    #[test]
    fn test_transformer_block_residual_effect() {
        let d_model = 8;
        let n_heads = 2;
        let ffn_hidden = 16;

        let block = TransformerBlock::new(d_model, n_heads, ffn_hidden);

        let x = Tensor::rand_uni(vec![4, d_model], 0.0, 1.0);
        let out = block.forward(&x);

        // Output should not be exactly equal to input unless weights are degenerate
        let equal = x.data().iter().zip(out.data()).all(|(&a, &b)| (a - b).abs() < EPS);
        assert!(!equal, "Output should differ from input due to attention + FFN");
    }
}
