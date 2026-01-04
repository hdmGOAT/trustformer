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

#[cfg(test)]
mod tests {
    use super::*;

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
}
