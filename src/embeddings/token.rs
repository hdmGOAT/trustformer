use crate::tensor::core::Tensor;

pub struct TokenEmbedding {
    weight: Tensor
}

impl TokenEmbedding {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Self {weight: Tensor::rand_norm(vec![vocab_size, d_model], 0.0, 0.2)}
    } 


    pub fn forward(&self, tokens: &[usize]) -> Tensor {
        if tokens.is_empty() {
            let d_model = self.weight.shape()[1];
            return Tensor::new(vec![], vec![0, d_model]);
        }

        let d_model = self.weight.shape()[1];
        let mut data = Vec::with_capacity(tokens.len() * d_model);

        for &tok in tokens {
            let row = self.weight.row(tok);
            data.extend_from_slice(&row);
        }

        Tensor::new(data, vec![tokens.len(), d_model])
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_initialization() {
        let vocab_size = 10;
        let d_model = 4;
        let emb = TokenEmbedding::new(vocab_size, d_model);

        let shape = emb.weight.shape();
        assert_eq!(shape, &[vocab_size, d_model]);

        assert_eq!(emb.weight.data().len(), vocab_size * d_model);
    }

    #[test]
    fn test_forward_non_empty_tokens() {
        let vocab_size = 5;
        let d_model = 3;
        let emb = TokenEmbedding::new(vocab_size, d_model);

        let tokens = vec![0, 2, 4];
        let out = emb.forward(&tokens);

        assert_eq!(out.shape(), &[tokens.len(), d_model]);

        for (i, &tok) in tokens.iter().enumerate() {
            let row_out = out.row(i);
            let row_weight = emb.weight.row(tok);
            assert_eq!(row_out, row_weight);
        }
    }

    #[test]
    fn test_forward_empty_tokens() {
        let vocab_size = 5;
        let d_model = 3;
        let emb = TokenEmbedding::new(vocab_size, d_model);

        let tokens: Vec<usize> = vec![];
        let out = emb.forward(&tokens);

        assert_eq!(out.shape(), &[0, d_model]);
        assert!(out.data().is_empty());
    }

    #[test]
    #[should_panic]
    fn test_forward_out_of_bounds_token() {
        let vocab_size = 5;
        let d_model = 3;
        let emb = TokenEmbedding::new(vocab_size, d_model);

        let tokens = vec![0, 5];
        let _ = emb.forward(&tokens);
    }
}
