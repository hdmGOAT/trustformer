use crate::{embeddings::{position::PositionEmbedding, token::TokenEmbedding}, tensor::{core::Tensor, ops::add}};

pub struct Embeddings {
    pub token: TokenEmbedding,
    pub position: PositionEmbedding
}

impl Embeddings {
    pub fn new (max_seq_len: usize, vocab_size: usize, d_model: usize) -> Self {
        Self { 
            token: TokenEmbedding::new(vocab_size, d_model), 
            position: PositionEmbedding::new(max_seq_len, d_model) 
        }
    }

    pub fn forward (&self, tokens: &[usize]) -> Tensor {
        let tok_emb = self.token.forward(tokens);
        let pos_emb = self.position.forward(tokens.len());

        add(&tok_emb, &pos_emb, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_shape() {
        let max_seq_len = 10;
        let vocab_size = 50;
        let d_model = 4;
        let emb = Embeddings::new(max_seq_len, vocab_size, d_model);

        let tokens = vec![0, 1, 2, 3, 4];
        let out = emb.forward(&tokens);

        assert_eq!(out.shape(), &[tokens.len(), d_model]);
        assert_eq!(out.data().len(), tokens.len() * d_model);
    }

    #[test]
    fn test_forward_empty_tokens() {
        let max_seq_len = 10;
        let vocab_size = 50;
        let d_model = 3;
        let emb = Embeddings::new(max_seq_len, vocab_size, d_model);

        let tokens = vec![];
        let out = emb.forward(&tokens);

        assert_eq!(out.shape(), &[0, d_model]);
        assert!(out.data().is_empty());
    }

    #[test]
    fn test_forward_token_pos_addition() {
        let max_seq_len = 6;
        let vocab_size = 10;
        let d_model = 3;
        let emb = Embeddings::new(max_seq_len, vocab_size, d_model);

        let tokens = vec![0, 1, 2];
        let out = emb.forward(&tokens);

        let tok_out = emb.token.forward(&tokens);
        let pos_out = emb.position.forward(tokens.len());

        // Ensure output equals elementwise addition
        for i in 0..tokens.len() {
            let row_out = out.row(i);
            let tok_row = tok_out.row(i);
            let pos_row = pos_out.row(i);
            for j in 0..d_model {
                assert!((row_out[j] - (tok_row[j] + pos_row[j])).abs() < 1e-6);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Sequence must not exceed max")]
    fn test_forward_seq_too_long() {
        let max_seq_len = 5;
        let vocab_size = 10;
        let d_model = 4;
        let emb = Embeddings::new(max_seq_len, vocab_size, d_model);

        // Position embedding should panic
        let tokens = vec![0, 1, 2, 3, 4, 5];
        let _ = emb.forward(&tokens);
    }

    #[test]
    fn test_forward_max_seq_len() {
        let max_seq_len = 6;
        let vocab_size = 20;
        let d_model = 2;
        let emb = Embeddings::new(max_seq_len, vocab_size, d_model);

        let tokens = vec![0, 1, 2, 3, 4, 5];
        let out = emb.forward(&tokens);

        assert_eq!(out.shape(), &[tokens.len(), d_model]);
        assert_eq!(out.data().len(), tokens.len() * d_model);
    }
}
