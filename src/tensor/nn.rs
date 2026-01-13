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
pub fn log(t: &Tensor) -> Tensor {
    let eps = 1e-8;

    let data: Vec<f32> = t
        .data()
        .iter()
        .map(|&x| {
            let safe_x = if x < eps { eps } else { x };
            safe_x.ln()
        })
        .collect();

    Tensor::new(data, t.shape().to_vec())
}

pub fn softmax_axis(t: &Tensor, axis: isize) -> Tensor {
    let shape = t.shape();
    let rank = shape.len();

    let axis = if axis < 0 {
        (rank as isize + axis) as usize
    } else {
        axis as usize
    };

    assert!(axis < rank);

    let axis_dim = shape[axis];
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();

    let mut out = vec![0.0; t.data().len()];

    for o in 0..outer {
        for i in 0..inner {
            let mut max_val = f32::NEG_INFINITY;

            for a in 0..axis_dim {
                let idx = o * axis_dim * inner + a * inner + i;
                max_val = max_val.max(t.data()[idx]);
            }

            let mut sum = 0.0;
            for a in 0..axis_dim {
                let idx = o * axis_dim * inner + a * inner + i;
                let e = safe_exp(t.data()[idx] - max_val);
                out[idx] = e;
                sum += e;
            }

            let denom = sum + 1e-8;
            for a in 0..axis_dim {
                let idx = o * axis_dim * inner + a * inner + i;
                out[idx] = safe_div(out[idx], denom, 1e-8).unwrap_or(0.0);
            }
        }
    }

    Tensor::new(out, shape.to_vec())
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

pub fn gelu(t: &Tensor) -> Tensor {
    let data: Vec<f32> = t.data().iter().map(|&x| {
        0.5 * x * (1.0 + (f32::consts::SQRT_2 / f32::consts::PI.sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }).collect();
    Tensor::new(data, t.shape().to_vec())
}

#[cfg(test)]
mod tests {
    use std::f32::consts::E;

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

    #[test]
    fn test_softmax_axis_1_rows() {
        // Shape [2, 3] -> 2 rows, 3 columns
        // [ [0.0, 0.0, 0.0], 
        //   [1.0, 1.0, 1.0] ]
        let data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let t = Tensor::new(data, vec![2, 3]);
        
        // Axis 1 means we apply softmax ACROSS the columns (per row)
        let s = softmax_axis(&t, 1);

        // Since elements in row 0 are equal, probs should be 0.333...
        assert!(approx_eq(s.data()[0], 1.0/3.0, 1e-6));
        assert!(approx_eq(s.data()[1], 1.0/3.0, 1e-6));
        assert!(approx_eq(s.data()[2], 1.0/3.0, 1e-6));
        
        // Row 0 sum should be 1.0
        let row0_sum: f32 = s.data()[0..3].iter().sum();
        assert!(approx_eq(row0_sum, 1.0, 1e-6));

        // Row 1 sum should be 1.0
        let row1_sum: f32 = s.data()[3..6].iter().sum();
        assert!(approx_eq(row1_sum, 1.0, 1e-6));
    }

    #[test]
    fn test_softmax_axis_0_cols() {
        // Shape [2, 2]
        // [ 0.0, 10.0 ]
        // [ 0.0, 10.0 ]
        let data = vec![0.0, 10.0, 0.0, 10.0];
        let t = Tensor::new(data, vec![2, 2]);

        // Axis 0 means we apply softmax DOWN the columns
        let s = softmax_axis(&t, 0);

        // Column 0: softmax([0, 0]) -> [0.5, 0.5]
        assert!(approx_eq(s.data()[0], 0.5, 1e-6)); // index (0,0)
        assert!(approx_eq(s.data()[2], 0.5, 1e-6)); // index (1,0)

        // Column 1: softmax([10, 10]) -> [0.5, 0.5]
        assert!(approx_eq(s.data()[1], 0.5, 1e-6)); // index (0,1)
        assert!(approx_eq(s.data()[3], 0.5, 1e-6)); // index (1,1)
    }

    #[test]
    fn test_softmax_axis_negative_indexing() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        
        // Axis -1 on a rank 2 tensor == Axis 1
        let s_neg = softmax_axis(&t, -1);
        let s_pos = softmax_axis(&t, 1);

        for (a, b) in s_neg.data().iter().zip(s_pos.data().iter()) {
            assert!(approx_eq(*a, *b, 1e-8));
        }
    }

    #[test]
    fn test_softmax_axis_3d_middle() {
        // Shape [2, 2, 2]. Axis 1.
        // We want to ensure the stride logic (outer/inner) works correctly.
        // We will make the data such that along axis 1, values are distinct, 
        // but along other axes they are identical, to verify direction.
        
        // Flat: [0, 0, 10, 10,  0, 0, 10, 10]
        // Batch 0:
        //   Row 0: [0, 0]
        //   Row 1: [10, 10]
        // We expect softmax along axis 1 (vertical in the 2x2 slice).
        // For slice (0,0): Inputs are 0 (from row 0) and 10 (from row 1).
        // Softmax(0, 10) -> roughly [0.0, 1.0]
        
        let data = vec![
            0.0, 0.0,   // Batch 0, Row 0
            10.0, 10.0, // Batch 0, Row 1
            0.0, 0.0,   // Batch 1, Row 0
            10.0, 10.0  // Batch 1, Row 1
        ]; 
        let t = Tensor::new(data, vec![2, 2, 2]);
        let s = softmax_axis(&t, 1);

        // Check Batch 0, Col 0 (Indices 0 and 2)
        let val_low = s.data()[0]; // Input 0.0
        let val_high = s.data()[2]; // Input 10.0

        // exp(0) / (exp(0) + exp(10)) is very small
        // exp(10) / (exp(0) + exp(10)) is near 1
        assert!(val_low < 0.001);
        assert!(val_high > 0.999);
        assert!(approx_eq(val_low + val_high, 1.0, 1e-6));
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Test with large numbers to ensure safe_exp / max-subtraction is working
        // Softmax([1000, 1001, 1002]) should equal Softmax([0, 1, 2])
        let t_large = Tensor::new(vec![1000.0, 1001.0, 1002.0], vec![1, 3]);
        let t_small = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]);

        let s_large = softmax_axis(&t_large, 1);
        let s_small = softmax_axis(&t_small, 1);

        for (a, b) in s_large.data().iter().zip(s_small.data().iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }

        // Ensure we don't get NaNs
        assert!(!s_large.data()[0].is_nan());
    }

    // --- Edge Case Tests ---

    #[test]
    fn test_normalize_constant_tensor() {
        // If all values are the same, max - min = 0.
        // safe_div should handle this (likely returning 0 or handling epsilon).
        let t = Tensor::new(vec![5.0, 5.0, 5.0], vec![3]);
        let n = normalize(&t);
        
        // Based on your implementation: x - min = 0. range = 0. safe_div(0, 0) -> 0.
        for &val in n.data() {
            assert!(approx_eq(val, 0.0, 1e-6));
        }
    }

    #[test]
    #[should_panic]
    fn test_softmax_axis_out_of_bounds() {
        let t = Tensor::new(vec![1.0, 2.0], vec![2]);
        // Rank is 1. Axis 1 is out of bounds. Should panic/assert.
        softmax_axis(&t, 1);
    }

    #[test]
    fn test_gelu_basic() {
        let t = Tensor::new(vec![0.0, 1.0, -1.0], vec![3]);
        let g = gelu(&t);
        
        // GELU(0) = 0
        assert!(approx_eq(g.data()[0], 0.0, 1e-6));
        
        // GELU(1) approx 0.8413
        assert!(approx_eq(g.data()[1], 0.84119, 1e-4));
        
        // GELU(-1) approx -0.1587
        assert!(approx_eq(g.data()[2], -0.1588, 1e-4));
    }

    #[test]
    fn test_log_basic() {
        let t = Tensor::new(vec![1.0, E], vec![2]);
        let l = log(&t);

        assert!(approx_eq(l.data()[0], 0.0, 1e-6)); // ln(1) = 0
        assert!(approx_eq(l.data()[1], 1.0, 1e-6)); // ln(e) = 1
    }

    #[test]
    fn test_log_preserves_shape() {
        let t = Tensor::new(vec![0.2, 0.3, 0.5], vec![3, 1]);
        let l = log(&t);

        assert_eq!(l.shape(), &[3, 1]);
    }

    #[test]
    fn test_log_numerical_safety_zero() {
        let t = Tensor::new(vec![0.0, 1e-12], vec![2]);
        let l = log(&t);

        // Should not be -inf or NaN
        for &val in l.data() {
            assert!(!val.is_nan());
            assert!(!val.is_infinite());
        }
    }

    #[test]
    fn test_log_after_softmax() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let s = softmax_axis(&t, 1);
        let l = log(&s);

        // log(softmax) should sum to something < 0
        let sum: f32 = l.data().iter().sum();
        assert!(sum < 0.0);
    }
}
