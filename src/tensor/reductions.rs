use crate::tensor::core::Tensor;
use std::f32;

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

pub fn is_finite(t: &Tensor) -> bool {
    t.data().iter().all(|&a| a.is_finite())
}

pub fn max(t: &Tensor) -> Option<f32> {
    t.data().iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::float::approx_eq;
    use std::f32;

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
        assert!(is_finite(&finite_tensor));

        let inf_tensor = Tensor::new(vec![1.0, f32::INFINITY], vec![2, 1]);
        assert!(!is_finite(&inf_tensor));

        let nan_tensor = Tensor::new(vec![1.0, f32::NAN], vec![2, 1]);
        assert!(!is_finite(&nan_tensor));
    }

    #[test]
    fn test_tensor_max_basic() {
        let t = Tensor::new(vec![1.0, 5.0, 3.0], vec![3, 1]);
        assert_eq!(max(&t), Some(5.0));

        let empty = Tensor::new(vec![], vec![0]);
        assert_eq!(max(&empty), None);
    }
}
