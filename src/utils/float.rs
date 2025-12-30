use core::f32;

use crate::tensor::{ core::Tensor};

const MAX_EXP: f32 = 88.0;
const MIN_EXP: f32 = -88.0;

pub fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

pub fn tensor_allclose(a: &Tensor, b: &Tensor, eps: f32) -> bool {
    a.shape() == b.shape()
        && a.stride() == b.stride()
        && a.data()
            .iter()
            .zip(b.data().iter())
            .all(|(a, b)| approx_eq(*a, *b, eps))
}

pub fn clamp(val: f32, min: f32, max: f32) -> f32 {
    if val > max {
        return max;
    } else if val < min {
        return min;
    }
    val
}

pub fn safe_exp(val: f32) -> f32 {
    clamp(val, MIN_EXP, MAX_EXP).exp()
}

pub fn safe_div(a: f32, b: f32, eps: f32) -> Option<f32> {
    if b.abs() < eps {
        None
    } else {
        Some(a / b)
    }
}

pub fn stable_sum(values: &[f32]) -> f32 {
    let mut sum: f32 = 0.0;
    let mut comp: f32 = 0.0;

    for &val in values {
        let y = val - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

pub fn tensor_is_finite(t: &Tensor) -> bool {
    t.data().iter().all(|&a| a.is_finite())
}


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


pub fn tensor_max(t: &Tensor) -> Option<f32> {
    t.data().iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap())
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
    use std::f32;

    #[test]
    fn test_approx_eq_basic() {
        assert!(approx_eq(1.0, 1.5, 1.0));
        assert!(!approx_eq(1.0, 1.5, 0.4));
    }

    #[test]
    fn test_approx_eq_very_small() {
        assert!(approx_eq(1.00000001, 1.0, 0.00001));
        assert!(!approx_eq(1.0001, 1.0, 0.00001));
    }

    #[test]
    fn test_tensor_all_close_basic() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let b = a.clone();
        assert!(tensor_allclose(&a, &b, 1e-6));
    }

    #[test]
    fn test_tensor_all_close_fail() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let b = Tensor::new(vec![1.0, 2.0, 3.1], vec![3, 1]);
        assert!(!tensor_allclose(&a, &b, 0.01));
    }

    #[test]
    fn test_clamp_basic() {
        assert_eq!(clamp(50.0, 1.0, 100.0), 50.0);
        assert_eq!(clamp(0.0, 1.0, 100.0), 1.0);
        assert_eq!(clamp(200.0, 1.0, 100.0), 100.0);
    }

    #[test]
    fn test_safe_exp_stability() {
        let large_val = 1000.0;
        let small_val = -1000.0;
        assert!(approx_eq(safe_exp(MAX_EXP), safe_exp(large_val), 1e-6));
        assert!(approx_eq(safe_exp(MIN_EXP), safe_exp(small_val), 1e-6));
    }

    #[test]
    fn test_safe_div_basic() {
        assert_eq!(safe_div(10.0, 2.0, 1e-8), Some(5.0));
        assert_eq!(safe_div(1.0, 0.0, 1e-8), None);
        assert_eq!(safe_div(1.0, 1e-10, 1e-8), None);
    }

    #[test]
    fn test_stable_sum_basic() {
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(stable_sum(&vals), 10.0);
    }

    #[test]
    fn test_stable_sum_precision() {
        let n = 1_000_000;
        let vals: Vec<f32> = vec![0.000001; n]; 
        let naive: f32 = vals.iter().sum();
        let stable = stable_sum(&vals);
        let exact = 1.0_f32;

        let naive_error = (naive - exact).abs();
        let stable_error = (stable - exact).abs();

        assert!(stable_error <= naive_error, "stable sum should be more accurate than naive sum");
        assert!(approx_eq(stable, exact, 1e-6));
    }

    #[test]
    fn test_tensor_is_finite() {
        let finite_tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
        assert!(tensor_is_finite(&finite_tensor));

        let inf_tensor = Tensor::new(vec![1.0, f32::INFINITY], vec![2, 1]);
        assert!(!tensor_is_finite(&inf_tensor));

        let nan_tensor = Tensor::new(vec![1.0, f32::NAN], vec![2, 1]);
        assert!(!tensor_is_finite(&nan_tensor));
    }

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
    fn test_tensor_max_basic() {
        let t = Tensor::new(vec![1.0, 5.0, 3.0], vec![3, 1]);
        assert_eq!(tensor_max(&t), Some(5.0));

        let empty = Tensor::new(vec![], vec![0]);
        assert_eq!(tensor_max(&empty), None);
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
