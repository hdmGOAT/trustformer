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
