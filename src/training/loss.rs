use crate::tensor::core::Tensor;

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss
    }

    pub fn gather_1d(logits: &Tensor, targets: &Tensor) -> Tensor {
        // logits is [seq_len, vocab_size], targets is [seq_len]
        assert_eq!(logits.shape()[0], targets.shape()[0], "Sequence length mismatch");
        let seq_len = targets.data().len();
        let mut gathered = Vec::with_capacity(seq_len);
        for i in 0..seq_len{
            let row = logits.row(i);
            let target_idx = targets.get(&[i]) as usize;
            gathered.push(row[target_idx]);
        }

        Tensor::new(gathered, vec![seq_len])
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::core::Tensor;

    #[test]
    fn test_gather_basic() {
        // logits: 3 positions, vocab size 4
        let logits = Tensor::new(
            vec![
                0.1, 0.2, 0.6, 0.1,   // first token
                0.05, 0.1, 0.1, 0.75, // second token
                0.7, 0.1, 0.1, 0.1    // third token
            ],
            vec![3, 4],
        );

        let targets = Tensor::new(vec![2.0, 3.0, 0.0], vec![3]);

        let gathered = CrossEntropyLoss::gather_1d(&logits, &targets);
        let expected = vec![0.6, 0.75, 0.7];

        for (a, b) in gathered.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(gathered.shape(), &[3]);
    }

    #[test]
    fn test_gather_seq_len_1() {
        let logits = Tensor::new(vec![0.1, 0.9, 0.0], vec![1, 3]);
        let targets = Tensor::new(vec![1.0], vec![1]);

        let gathered = CrossEntropyLoss::gather_1d(&logits, &targets);
        assert_eq!(gathered.data(), &[0.9]);
        assert_eq!(gathered.shape(), &[1]);
    }

    #[test]
    #[should_panic(expected = "Sequence length mismatch")]
    fn test_gather_mismatched_length() {
        let logits = Tensor::new(vec![0.1, 0.9, 0.0, 0.2, 0.3, 0.5], vec![2, 3]);
        let targets = Tensor::new(vec![0.0], vec![1]);

        // should panic because seq_len != targets.len()
        CrossEntropyLoss::gather_1d(&logits, &targets);
    }

    #[test]
    #[should_panic]
    fn test_gather_target_out_of_bounds() {
        let logits = Tensor::new(vec![0.1, 0.2, 0.3], vec![1, 3]);
        let targets = Tensor::new(vec![3.0], vec![1]); // index 3 is out-of-bounds
        CrossEntropyLoss::gather_1d(&logits, &targets);
    }
}
