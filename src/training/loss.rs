use crate::tensor::core::Tensor;

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss
    }

    pub fn gather_1d(&self, logits: &Tensor, targets: &Tensor) -> Tensor {
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

    pub fn forward(&self, logits: &Tensor, targets: &Tensor) -> Tensor {
        let log_probs = logits.softmax_axis(1).log();

        let picked = self.gather_1d(&log_probs, targets);

        let seq_len = picked.shape()[0] as f32;
        let loss = -picked.data().iter().sum::<f32>() / seq_len;

        Tensor::new(vec![loss], vec![1])
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::core::Tensor;

    /* ---------------- gather_1d tests ---------------- */

    #[test]
    fn test_gather_basic() {
        let logits = Tensor::new(
            vec![
                0.1, 0.2, 0.6, 0.1,
                0.05, 0.1, 0.1, 0.75,
                0.7, 0.1, 0.1, 0.1,
            ],
            vec![3, 4],
        );

        let targets = Tensor::new(vec![2.0, 3.0, 0.0], vec![3]);
        let loss_fn = CrossEntropyLoss::new();

        let gathered = loss_fn.gather_1d(&logits, &targets);
        let expected = [0.6, 0.75, 0.7];

        for (a, b) in gathered.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        assert_eq!(gathered.shape(), &[3]);
    }

    #[test]
    fn test_gather_seq_len_1() {
        let logits = Tensor::new(vec![0.1, 0.9, 0.0], vec![1, 3]);
        let targets = Tensor::new(vec![1.0], vec![1]);

        let loss_fn = CrossEntropyLoss::new();
        let gathered = loss_fn.gather_1d(&logits, &targets);

        assert_eq!(gathered.data(), &[0.9]);
        assert_eq!(gathered.shape(), &[1]);
    }

    #[test]
    #[should_panic(expected = "Sequence length mismatch")]
    fn test_gather_mismatched_length() {
        let logits = Tensor::new(vec![0.1, 0.9, 0.0, 0.2, 0.3, 0.5], vec![2, 3]);
        let targets = Tensor::new(vec![0.0], vec![1]);

        let loss_fn = CrossEntropyLoss::new();
        loss_fn.gather_1d(&logits, &targets);
    }

    #[test]
    #[should_panic]
    fn test_gather_target_out_of_bounds() {
        let logits = Tensor::new(vec![0.1, 0.2, 0.3], vec![1, 3]);
        let targets = Tensor::new(vec![3.0], vec![1]);

        let loss_fn = CrossEntropyLoss::new();
        loss_fn.gather_1d(&logits, &targets);
    }

    /* ---------------- cross entropy tests ---------------- */

    #[test]
    fn test_cross_entropy_simple_known_case() {
        let logits = Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 3]);
        let targets = Tensor::new(vec![1.0], vec![1]);

        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.forward(&logits, &targets);

        // softmax = [1/3, 1/3, 1/3]
        // loss = -log(1/3)
        let expected = (3.0f32).ln();

        assert!((loss.data()[0] - expected).abs() < 1e-5);
        assert_eq!(loss.shape(), &[1]);
    }

    #[test]
    fn test_cross_entropy_uniform_distribution() {
        let logits = Tensor::new(
            vec![
                0.0, 0.0,
                0.0, 0.0,
            ],
            vec![2, 2],
        );
        let targets = Tensor::new(vec![0.0, 1.0], vec![2]);

        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.forward(&logits, &targets);

        // softmax = [0.5, 0.5]
        // loss per token = -log(0.5) = ln(2)
        // mean over tokens keeps ln(2)
        let expected = (2.0f32).ln();

        assert!((loss.data()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_confident_correct_prediction() {
        // Use moderately confident logits instead of extreme
        let logits = Tensor::new(vec![6.0, 0.0, 0.0], vec![1, 3]);
        let targets = Tensor::new(vec![0.0], vec![1]);

        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.forward(&logits, &targets);

        // Loss is small and positive
        assert!(loss.data()[0] > 0.0);
        assert!(loss.data()[0] < 0.01);
    }

    #[test]
    fn test_cross_entropy_multiple_tokens_average() {
        let logits = Tensor::new(
            vec![
                2.0, 0.0,
                0.0, 2.0,
            ],
            vec![2, 2],
        );
        let targets = Tensor::new(vec![0.0, 1.0], vec![2]);

        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.forward(&logits, &targets);

        // Each token has loss ~0.127, average is same
        assert!(loss.data()[0] > 0.05);
        assert!(loss.data()[0] < 0.3);
    }

    #[test]
    fn test_cross_entropy_no_nan_or_inf() {
        let logits = Tensor::new(vec![1000.0, -1000.0], vec![1, 2]);
        let targets = Tensor::new(vec![0.0], vec![1]);

        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.forward(&logits, &targets);

        assert!(!loss.data()[0].is_nan());
        assert!(!loss.data()[0].is_infinite());
    }
}
