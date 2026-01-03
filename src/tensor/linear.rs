use crate::tensor::{core::Tensor, ops::{add, dot}};

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            weight: Tensor::rand_norm(vec![out_dim, in_dim],0.0, 0.02 ),
            bias: Tensor::n_tensor(0.0, vec![out_dim]),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let y = dot(x, &self.weight.transpose(vec![1, 0]));

        add(&y, &self.bias, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_initialization() {
        let in_dim = 5;
        let out_dim = 3;
        let linear = Linear::new(in_dim, out_dim);

        assert_eq!(linear.weight.shape(), &[out_dim, in_dim]);
        assert_eq!(linear.bias.shape(), &[out_dim]);
    }

    #[test]
    fn test_linear_forward() {
        let in_dim = 2;
        let out_dim = 2;
        let mut linear = Linear::new(in_dim, out_dim);
        
        // Manually set weights and bias for predictable output
        // Weight: [[1.0, 2.0], [3.0, 4.0]]
        // Bias: [0.5, 0.5]
        linear.weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![out_dim, in_dim]);
        linear.bias = Tensor::new(vec![0.5, 0.5], vec![out_dim]);

        // Input: [[1.0, 1.0]] (Batch size 1)
        let input = Tensor::new(vec![1.0, 1.0], vec![1, in_dim]);
        
        let output = linear.forward(&input);
        
        // Expected output:
        // y = x * W^T + b
        // [1, 1] * [[1, 3], [2, 4]] + [0.5, 0.5]
        // = [1*1 + 1*2, 1*3 + 1*4] + [0.5, 0.5]
        // = [3, 7] + [0.5, 0.5]
        // = [3.5, 7.5]
        
        assert_eq!(output.shape(), &[1, out_dim]);
        let data = output.data();
        assert!((data[0] - 3.5).abs() < 1e-6);
        assert!((data[1] - 7.5).abs() < 1e-6);
    }

    #[test]
    fn test_linear_forward_batch() {
        let in_dim = 2;
        let out_dim = 2;
        let mut linear = Linear::new(in_dim, out_dim);
        
        // Weight: [[1.0, 1.0], [1.0, 1.0]]
        linear.weight = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![out_dim, in_dim]);
        linear.bias = Tensor::new(vec![0.0, 0.0], vec![out_dim]);

        // Input: [[1.0, 2.0], [3.0, 4.0]] (Batch size 2)
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, in_dim]);
        
        let output = linear.forward(&input);
        
        // Expected output:
        // x * W^T
        // [[1, 2], [3, 4]] * [[1, 1], [1, 1]]
        // Row 0: [1*1 + 2*1, 1*1 + 2*1] = [3, 3]
        // Row 1: [3*1 + 4*1, 3*1 + 4*1] = [7, 7]
        
        assert_eq!(output.shape(), &[2, out_dim]);
        let data = output.data();
        assert!((data[0] - 3.0).abs() < 1e-6);
        assert!((data[1] - 3.0).abs() < 1e-6);
        assert!((data[2] - 7.0).abs() < 1e-6);
        assert!((data[3] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_forward_dims() {
        let in_dim = 3;
        let out_dim = 2;
        let mut linear = Linear::new(in_dim, out_dim);
        
        // Weight: [2, 3]
        // [[1, 1, 1], [2, 2, 2]]
        linear.weight = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], vec![out_dim, in_dim]);
        linear.bias = Tensor::new(vec![0.0, 0.0], vec![out_dim]);

        // Input: [1, 3]
        // [[1, 2, 3]]
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, in_dim]);
        
        let output = linear.forward(&input);
        
        // Expected:
        // [1, 3] * [3, 2] (W^T)
        // W^T = [[1, 2], [1, 2], [1, 2]]
        // [1*1 + 2*1 + 3*1, 1*2 + 2*2 + 3*2]
        // [6, 12]
        
        assert_eq!(output.shape(), &[1, out_dim]);
        let data = output.data();
        assert!((data[0] - 6.0).abs() < 1e-6);
        assert!((data[1] - 12.0).abs() < 1e-6);
    }
}

