use crate::{tensor::core::Tensor, utils::float::safe_div};

pub fn add(a: &Tensor, b: &Tensor, broadcast: bool) -> Tensor {
    if !broadcast {
        assert_eq!(a.shape(), b.shape());
        return Tensor::new(a.data().iter().zip(b.data()).map(|(x, y)| x + y).collect(), a.shape().to_vec());
    }
    broadcast_op(a, b, |x, y| x + y)
}

pub fn sub(a: &Tensor, b: &Tensor, broadcast: bool) -> Tensor {
    if !broadcast {
        assert_eq!(a.shape(), b.shape());
        return Tensor::new(a.data().iter().zip(b.data()).map(|(x, y)| x - y).collect(), a.shape().to_vec());
    }
    broadcast_op(a, b, |x, y| x - y)
}

pub fn mul(a: &Tensor, b: &Tensor, broadcast: bool) -> Tensor {
    if !broadcast {
        assert_eq!(a.shape(), b.shape());
        return Tensor::new(a.data().iter().zip(b.data()).map(|(x, y)| x * y).collect(), a.shape().to_vec());
    }
    broadcast_op(a, b, |x, y| x * y)
}

pub fn div(a: &Tensor, b: &Tensor, eps: f32, broadcast: bool) -> Tensor {
    if !broadcast {
        return Tensor::new(a.data().iter().zip(b.data()).map(|(x, y)| safe_div(*x, *y, eps).unwrap()).collect(), a.shape().to_vec());
    }
    broadcast_op(a, b, |x, y| safe_div(x, y, eps).unwrap())
}

fn broadcast_op<F>(a: &Tensor, b: &Tensor, f: F) -> Tensor
where F: Fn(f32, f32) -> f32
{
    let a_shape = a.shape();
    let b_shape = b.shape();
    let a_stride = a.stride();
    let b_stride = b.stride();

    let rank_a = a_shape.len();
    let rank_b = b_shape.len();
    let max_rank = rank_a.max(rank_b);
    
    let offset_a = max_rank - rank_a;
    let offset_b = max_rank - rank_b;

    let mut shape = Vec::with_capacity(max_rank);
    for i in 0..max_rank {
        let dim_a = if i < offset_a { 1 } else { a_shape[i - offset_a] };
        let dim_b = if i < offset_b { 1 } else { b_shape[i - offset_b] };
        
        if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
             panic!("Operands could not be broadcast together with shapes {:?} {:?}", a_shape, b_shape);
        }
        shape.push(if dim_a == 1 { dim_b } else { dim_a });
    }

    let mut result = vec![0.0; shape.iter().product()];
    let r_stride = Tensor::compute_stride(&shape);

    for (idx, r) in result.iter_mut().enumerate() {
        let mut temp_idx = idx;
        let mut a_idx = 0;
        let mut b_idx = 0;

        for (i, &stride) in r_stride.iter().enumerate() {
            let coord = temp_idx / stride;
            temp_idx %= stride;

            if i >= offset_a {
                let real_idx = i - offset_a;
                if a_shape[real_idx] != 1 {
                    a_idx += coord * a_stride[real_idx];
                }
            }

            if i >= offset_b {
                let real_idx = i - offset_b;
                if b_shape[real_idx] != 1 {
                    b_idx += coord * b_stride[real_idx];
                }
            }
        }

        *r = f(a.data()[a_idx], b.data()[b_idx]);
    }

    Tensor::new_with_stride(result, shape, r_stride)
}

pub fn dot(a: &Tensor, b: &Tensor) -> Tensor {
    match (a.shape().len(), b.shape().len()) {
        (1, 1) => {
            assert_eq!(a.shape()[0], b.shape()[0]);
            let sum = a.data().iter().zip(b.data()).map(|(x,y)| x*y).sum::<f32>();
            Tensor::new(vec![sum], vec![1])
        },
        (2, 2) => {
            let (m, k_a) = (a.shape()[0], a.shape()[1]);
            let (k_b, n) = (b.shape()[0], b.shape()[1]);
            assert_eq!(k_a, k_b, "Inner dimensions must match for dot product");

            let mut result = vec![0.0; m * n];

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..k_a {
                        sum += a.data()[i*k_a + k] * b.data()[k*n + j];
                    }
                    result[i*n + j] = sum;
                }
            }

            Tensor::new(result, vec![m, n])
        },
        _ => panic!("Dot product currently supports only 1D or 2D tensors"),
    }
}
#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_add_tensors_rank2(){
        let a = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let sum = add(&a, &b, false);
        let expected_sum = Tensor::new(vec![5.0, 5.0, 5.0, 5.0], vec![2, 2]);
        assert_eq!(sum.data(), expected_sum.data());
    }

    #[test]
    #[should_panic]
    fn test_add_tensors_mismatch(){
        let a = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);

        let _sum = add(&a, &b, false);
    }

    #[test]
    fn test_sub_tensors_rank2(){
        let a = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let dif = sub(&a, &b, false);
        let expected_dif = Tensor::new(vec![3.0, 1.0, -1.0, -3.0], vec![2, 2]);
        assert_eq!(dif.data(), expected_dif.data());
    }

    #[test]
    #[should_panic]
    fn test_dif_tensors_mismatch(){
        let a = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);

        let _sum = sub(&a, &b, false);
    }

    #[test]
    fn test_add_tensors_1d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);

        let sum = add(&a, &b, false);
        let expected = Tensor::new(vec![5.0, 7.0, 9.0], vec![3]);
        assert_eq!(sum.data(), expected.data());
    }

    #[test]
    fn test_sub_tensors_1d() {
        let a = Tensor::new(vec![5.0, 7.0, 9.0], vec![3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

        let dif = sub(&a, &b, false);
        let expected = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        assert_eq!(dif.data(), expected.data());
    }

    #[test]
    fn test_mul_tensors_rank2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let product = mul(&a, &b, false);
        let expected = Tensor::new(vec![5.0, 12.0, 21.0, 32.0], vec![2, 2]);
        assert_eq!(product.data(), expected.data());
    }

    #[test]
    fn test_mul_single_element() {
        let a = Tensor::new(vec![3.0], vec![1]);
        let b = Tensor::new(vec![7.0], vec![1]);

        let product = mul(&a, &b, false);
        let expected = Tensor::new(vec![21.0], vec![1]);
        assert_eq!(product.data(), expected.data());
    }

    #[test]
    fn test_div_tensors_rank2() {
        let a = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let b = Tensor::new(vec![2.0, 4.0, 5.0, 8.0], vec![2, 2]);

        let result = div(&a, &b, 1e-6, false);
        let expected = Tensor::new(vec![5.0, 5.0, 6.0, 5.0], vec![2, 2]);
        assert_eq!(result.data(), expected.data());
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_div_single_element() {
        let a = Tensor::new(vec![21.0], vec![1]);
        let b = Tensor::new(vec![7.0], vec![1]);

        let result = div(&a, &b, 1e-6, false);
        let expected = Tensor::new(vec![3.0], vec![1]);
        assert_eq!(result.data(), expected.data());
    }

    #[test]
    fn test_add_3d_tensor_broadcast() {
        let a = Tensor::new(vec![1.0,2.0,3.0,4.0,5.0,6.0], vec![2,1,3]);
        let b = Tensor::new(vec![6.0,5.0,4.0,3.0,2.0,1.0], vec![2,1,3]);

        let sum = add(&a, &b, true);
        let expected = Tensor::new(vec![7.0,7.0,7.0,7.0,7.0,7.0], vec![2,1,3]);
        assert_eq!(sum.data(), expected.data());
    }

    #[test]
    fn test_dot_vectors() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);

        let result = dot(&a, &b);
        let expected = Tensor::new(vec![32.0], vec![1]);
        assert_eq!(result.data(), expected.data());
        assert_eq!(result.shape(), expected.shape());
    }

    #[test]
    fn test_dot_2x2_matrices() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let result = dot(&a, &b);
        let expected = Tensor::new(vec![
            1.0*5.0 + 2.0*7.0, 1.0*6.0 + 2.0*8.0,
            3.0*5.0 + 4.0*7.0, 3.0*6.0 + 4.0*8.0
        ], vec![2, 2]);

        assert_eq!(result.data(), expected.data());
        assert_eq!(result.shape(), expected.shape());
    }

    #[test]
    fn test_dot_rectangular_matrices() {
        let a = Tensor::new(vec![
            1.0, 2.0, 
            3.0, 4.0,
            5.0, 6.0
        ], vec![3, 2]);

        let b = Tensor::new(vec![
            7.0, 8.0, 9.0,
            10.0,11.0,12.0
        ], vec![2, 3]);

        let result = dot(&a, &b);
        let expected = Tensor::new(vec![
            1.0*7.0 + 2.0*10.0, 1.0*8.0 + 2.0*11.0, 1.0*9.0 + 2.0*12.0,
            3.0*7.0 + 4.0*10.0, 3.0*8.0 + 4.0*11.0, 3.0*9.0 + 4.0*12.0,
            5.0*7.0 + 6.0*10.0, 5.0*8.0 + 6.0*11.0, 5.0*9.0 + 6.0*12.0
        ], vec![3, 3]);

        assert_eq!(result.data(), expected.data());
        assert_eq!(result.shape(), expected.shape());
    }

    #[test]
    #[should_panic]
    fn test_dot_mismatched_vectors() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let _ = dot(&a, &b);
    }

    #[test]
    #[should_panic]
    fn test_dot_mismatched_matrices() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let _ = dot(&a, &b);
    }

    #[test]
    #[should_panic]
    fn test_dot_high_rank_tensor() {
        let a = Tensor::new(vec![1.0,2.0,3.0,4.0], vec![2,2,1]);
        let b = Tensor::new(vec![1.0,2.0,3.0,4.0], vec![2,2,1]);
        let _ = dot(&a, &b);
    }
}
