use crate::tensor::{core::Tensor, ops::{add, div, mul, sub}};

pub fn residual_layernorm(x: &Tensor, f_x: &Tensor, ln: &LayerNorm) -> Tensor {
    let added = add(x, f_x, false);
    ln.forward(&added)
}

pub struct LayerNorm {
    gamma: Tensor,
    beta: Tensor,
    eps: f32
}

impl LayerNorm {
    pub fn new(hidden_dim: usize) -> Self {
        Self { 
            gamma: Tensor::n_tensor(1.0, vec![hidden_dim]), 
            beta: Tensor::n_tensor(0.0, vec![hidden_dim]), 
            eps: 1e-5
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mean = x.mean_last_dim();
        let variance = x.var_last_dim();
        let centered = sub(x, &mean, true);

        let eps_tensor = Tensor::n_tensor(self.eps, variance.shape().to_vec());
        let var_eps = add(&variance, &eps_tensor, true);

        let denom = var_eps.sqrt();

        let norm = div(&centered, &denom, self.eps, true);

        let prod = mul(&norm, &self.gamma, true);
        add(&prod, &self.beta, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::core::Tensor;

    #[test]
    fn test_layernorm_forward_basic() {
        let ln = LayerNorm::new(3);
        let x = Tensor::new(vec![
            1.0, 2.0, 3.0, 
            4.0, 5.0, 6.0
        ], vec![2, 3]); 

        let y = ln.forward(&x);

        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_layernorm_zero_mean_unit_var() {
        let ln = LayerNorm::new(3);
        let x = Tensor::new(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ], vec![2, 3]);

        let y = ln.forward(&x);

        let mean = y.mean_last_dim();
        let var = y.var_last_dim();

        for m in mean.data() {
            assert!((*m).abs() < 1e-4, "Mean not zero: {}", m);
        }

        for v in var.data() {
            assert!((v - 1.0).abs() < 1e-4, "Variance not 1: {}", v);
        }
    }

    #[test]
    fn test_layernorm_with_batch() {
        let ln = LayerNorm::new(4);
        let x = Tensor::new(vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        ], vec![3, 4]);

        let y = ln.forward(&x);

        assert_eq!(y.shape(), x.shape());

        let mean = y.mean_last_dim();
        let var = y.var_last_dim();

        for m in mean.data() {
            assert!((*m).abs() < 1e-4);
        }
        for v in var.data() {
            assert!((v - 1.0).abs() < 1e-4);
        }
    }
}
