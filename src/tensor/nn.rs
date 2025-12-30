use crate::tensor::core::Tensor;
use crate::utils::float::{safe_exp, safe_div};
use crate::tensor::reductions::stable_sum;
use std::f32;

pub fn softmax(t: &Tensor) -> Tensor {
    let max_val = t.data().iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = t
        .data()
        .iter()
        .map(|&x| safe_exp(x - max_val))
        .collect();

    let sum = stable_sum(&exps);
    let normalized: Vec<f32> = exps
        .iter()
        .map(|&x| safe_div(x, sum, 1e-8).unwrap_or(0.0))
        .collect();

    Tensor::new(normalized, t.shape().to_vec())
}

pub fn normalize(t: &Tensor) -> Tensor {
    let max_val = t.data().iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_val = t.data().iter().copied().fold(f32::INFINITY, f32::min);
    let range = max_val - min_val;

    let normalized: Vec<f32> = t
        .data()
        .iter()
        .map(|&x| safe_div(x - min_val, range, 1e-8).unwrap_or(0.0))
        .collect();

    Tensor::new(normalized, t.shape().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::float::approx_eq;

    #[test]
    fn test_softmax_basic() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let s = softmax(&t);
        let sum: f32 = s.data().iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-6));

        for &val in s.data() {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_normalize_basic() {
        let t = Tensor::new(vec![0.0, 5.0, 10.0], vec![3, 1]);
        let n = normalize(&t);
        let expected = [0.0, 0.5, 1.0];
        for (a, b) in n.data().iter().zip(expected.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_normalize_single_value() {
        let t = Tensor::new(vec![42.0], vec![1]);
        let n = normalize(&t);
        assert!(approx_eq(n.data()[0], 0.0, 1e-6)); // single value maps to 0
    }
}
