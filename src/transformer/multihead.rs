use crate::{tensor::{core::Tensor, nn::softmax, ops::{dot}}};

pub struct MultiHeadAttention {
    num_heads: usize,
    d_model: usize,
    d_k: usize,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    eps: f32
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let d_k = d_model / num_heads;

        let wq = Tensor::rand_norm(vec![d_model, d_model], 0.0, 0.02);
        let wk = Tensor::rand_norm(vec![d_model, d_model], 0.0, 0.02);
        let wv = Tensor::rand_norm(vec![d_model, d_model], 0.0, 0.02);
        let wo = Tensor::rand_norm(vec![d_model, d_model], 0.0, 0.02);

        Self { num_heads, d_model, d_k, wq, wk, wv, wo, eps: 1e-5}
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

        // Compute attention per head
        let mut out_h: Vec<Tensor>= Vec::with_capacity(self.num_heads);
        for i in 0..self.num_heads{
            let q_h = q_heads.slice_axis(0, i);
            let k_h = k_heads.slice_axis(0, i);
            let v_h = v_heads.slice_axis(0, i);

            let scores_h = dot(&q_h, &k_h.transpose(vec![1, 0])).div_scalar(self.d_k as f32);
            let weights_h = softmax(&scores_h);
            out_h.push(dot(&weights_h, &v_h));
        }
            
        // Merge heads
        // Apply W_O
        todo!()
    }
}
