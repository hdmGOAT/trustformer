use crate::{embeddings::core::Embeddings, tensor::{core::Tensor, linear::Linear}, transformer::{block::TransformerBlock, layernorm::LayerNorm}};

pub struct Decoder {
    emb: Embeddings,
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    pub lm_head: Linear
}

impl Decoder {
    pub fn new(
        vocab_size: usize,
        max_seq_len: usize,
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        ffn_hidden: usize
    ) -> Self {
        let emb = Embeddings::new(max_seq_len, vocab_size, d_model);

        let mut blocks = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            blocks.push(TransformerBlock::new(d_model, n_heads, ffn_hidden));
        }

        let ln_f = LayerNorm::new(d_model);
        let lm_head = Linear::new(d_model, vocab_size);

        Self { emb, blocks, ln_f, lm_head }
    }

    pub fn forward(&self, tokens: &[usize]) -> Tensor {
        let mut x = self.emb.forward(tokens);

        for block in &self.blocks {
            x = block.forward(&x);
        }

        x = self.ln_f.forward(&x);

        self.lm_head.forward(&x)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn test_decoder_forward_shape() {
        let vocab_size = 50;
        let max_seq_len = 16;
        let d_model = 32;
        let n_layers = 2;
        let n_heads = 4;
        let ffn_hidden = 64;

        let model = Decoder::new(vocab_size, max_seq_len, d_model, n_layers, n_heads, ffn_hidden);

        let tokens = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens);

        // logits shape = [seq_len, vocab_size]
        assert_eq!(logits.shape(), &[tokens.len(), vocab_size]);
    }

    #[test]
    fn test_decoder_forward_empty_sequence() {
        let vocab_size = 20;
        let max_seq_len = 10;
        let d_model = 8;
        let n_layers = 1;
        let n_heads = 2;
        let ffn_hidden = 16;

        let model = Decoder::new(vocab_size, max_seq_len, d_model, n_layers, n_heads, ffn_hidden);

        let tokens: Vec<usize> = vec![];
        let logits = model.forward(&tokens);

        assert_eq!(logits.shape(), &[0, vocab_size]);
        assert!(logits.data().is_empty());
    }

    #[test]
    fn test_decoder_forward_known_behavior() {
        // sanity: zero embeddings → lm_head bias dominates
        let vocab_size = 5;
        let max_seq_len = 4;
        let d_model = 3;
        let n_layers = 1;
        let n_heads = 1;
        let ffn_hidden = 4;

        let mut model = Decoder::new(vocab_size, max_seq_len, d_model, n_layers, n_heads, ffn_hidden);

        // zero embeddings
        for i in 0..model.emb.token.weight.data().len() {
            model.emb.token.weight.data_mut()[i] = 0.0;
        }
        for i in 0..model.emb.position.weight.data().len() {
            model.emb.position.weight.data_mut()[i] = 0.0;
        }
        for i in 0..model.lm_head.weight.data().len() {
            model.lm_head.weight.data_mut()[i] = 0.0;
        }

        // bias = 1.0
        for i in 0..model.lm_head.bias.data().len() {
            model.lm_head.bias.data_mut()[i] = 1.0;
        }

        let tokens = vec![0, 1, 2];
        let logits = model.forward(&tokens);

        // every output should equal bias (1.0)
        for &v in logits.data() {
            assert!((v - 1.0).abs() < EPS);
        }
    }

    #[test]
    fn test_decoder_forward_max_seq_len() {
        let vocab_size = 10;
        let max_seq_len = 5;
        let d_model = 4;
        let n_layers = 2;
        let n_heads = 2;
        let ffn_hidden = 8;

        let model = Decoder::new(vocab_size, max_seq_len, d_model, n_layers, n_heads, ffn_hidden);

        let tokens = vec![0, 1, 2, 3, 4];
        let logits = model.forward(&tokens);

        assert_eq!(logits.shape(), &[tokens.len(), vocab_size]);
    }

    #[test]
    fn test_decoder_forward_sequence_too_long_panics() {
        let vocab_size = 10;
        let max_seq_len = 5;
        let d_model = 4;
        let n_layers = 1;
        let n_heads = 2;
        let ffn_hidden = 8;

        let model = Decoder::new(vocab_size, max_seq_len, d_model, n_layers, n_heads, ffn_hidden);

        let tokens = vec![0, 1, 2, 3, 4, 5];

        let result = std::panic::catch_unwind(|| {
            let _ = model.forward(&tokens);
        });

        assert!(result.is_err(), "Should panic if sequence > max_seq_len");
    }

    #[test]
    fn test_decoder_minimal_generation() {
        let vocab_size = 10;
        let max_seq_len = 6;
        let d_model = 8;
        let n_layers = 2;
        let n_heads = 2;
        let ffn_hidden = 16;

        let model = Decoder::new(vocab_size, max_seq_len, d_model, n_layers, n_heads, ffn_hidden);

        // generate one token at a time (greedy simulation)
        let mut ctx = vec![0];
        for _ in 0..5 {
            let logits = model.forward(&ctx);
            assert_eq!(logits.shape(), &[ctx.len(), vocab_size]);

            // pick next token: argmax over last row
            let last_logits = logits.row(ctx.len() - 1);
            let next = last_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            ctx.push(next);
        }

        assert!(ctx.len() == 6);
    }
}
