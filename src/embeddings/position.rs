use crate::tensor::core::Tensor;

pub struct PositionEmbedding {
    pub weight: Tensor
}

impl PositionEmbedding {
    pub fn new (max_seq_len: usize, d_model: usize) -> Self {
        Self { weight: Tensor::rand_norm(vec![max_seq_len, d_model],0.0, 0.2)}
    }

    pub fn forward (&self, seq_len: usize) -> Tensor {

        let shape = self.weight.shape();
        let max_seq_len = shape[0];
        let d_model = shape[1];
        assert!(seq_len <= max_seq_len, "Sequence must not exceed max");
        
        let mut data = Vec::with_capacity(seq_len * d_model);
        for i in 0..seq_len{
            let row = self.weight.row(i);
            data.extend_from_slice(&row);
        }

        Tensor::new(data, vec![seq_len, d_model])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_shape() {
        let max_seq_len = 10;
        let d_model = 4;
        let pos_emb = PositionEmbedding::new(max_seq_len, d_model);

        let seq_len = 5;
        let out = pos_emb.forward(seq_len);

        assert_eq!(out.shape(), &[seq_len, d_model]);

        assert_eq!(out.data().len(), seq_len * d_model);
    }

    #[test]
    fn test_forward_empty_seq() {
        let max_seq_len = 10;
        let d_model = 3;
        let pos_emb = PositionEmbedding::new(max_seq_len, d_model);

        let seq_len = 0;
        let out = pos_emb.forward(seq_len);

        assert_eq!(out.shape(), &[0, d_model]);
        assert!(out.data().is_empty());
    }

    #[test]
    fn test_forward_max_seq_len() {
        let max_seq_len = 8;
        let d_model = 2;
        let pos_emb = PositionEmbedding::new(max_seq_len, d_model);

        let seq_len = 8;
        let out = pos_emb.forward(seq_len);

        assert_eq!(out.shape(), &[seq_len, d_model]);
        assert_eq!(out.data().len(), seq_len * d_model);
    }

    #[test]
    #[should_panic(expected = "Sequence must not exceed max")]
    fn test_forward_too_long() {
        let max_seq_len = 5;
        let d_model = 4;
        let pos_emb = PositionEmbedding::new(max_seq_len, d_model);

        let _ = pos_emb.forward(6);
    }

    #[test]
    fn test_forward_row_matches_weight() {
        let max_seq_len = 6;
        let d_model = 3;
        let pos_emb = PositionEmbedding::new(max_seq_len, d_model);

        let seq_len = 4;
        let out = pos_emb.forward(seq_len);

        for i in 0..seq_len {
            let weight_row = pos_emb.weight.row(i);
            let out_row = out.row(i);
            assert_eq!(weight_row, out_row);
        }
    }
}
