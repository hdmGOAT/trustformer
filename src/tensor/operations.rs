use crate::tensor::core::Tensor;

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape(), b.shape(), "The tensors are not of similar shape");

    let len = a.data().len();
    let mut result = Vec::with_capacity(len);

    let a_data = a.data();
    let b_data = b.data();

    for i in 0..len {
        result.push(a_data[i] + b_data[i]);
    }

    Tensor::new(result, a.shape().to_vec())
}

pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape(), b.shape(), "The tensors are not of similar shape");

    let len = a.data().len();
    let mut result = Vec::with_capacity(len);

    let a_data = a.data();
    let b_data = b.data();

    for i in 0..len {
        result.push(a_data[i] - b_data[i]);
    }

    Tensor::new(result, a.shape().to_vec())
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape(), b.shape(), "The tensors are not of similar shape");

    let len = a.data().len();
    let mut result = Vec::with_capacity(len);

    let a_data = a.data();
    let b_data = b.data();

    for i in 0..len {
        result.push(a_data[i] * b_data[i]);
    }

    Tensor::new(result, a.shape().to_vec())
}

pub fn dot(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();
    assert_eq!(a_shape.len(), 2, "a must be 2D");
    assert_eq!(b_shape.len(), 2, "b must be 2D");

    let (m, k) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    assert_eq!(k, k2, "Inner dimensions must match");

    let mut data = vec![0.0; m * n];
    

    let a_stride = a.stride();
    let b_stride = b.stride();
    let a_stride0 = a_stride[0];
    let a_stride1 = a_stride[1];
    let b_stride0 = b_stride[0];
    let b_stride1 = b_stride[1];

    let a_data = a.data();
    let b_data = b.data();
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                let a_idx = i * a_stride0 + p * a_stride1;
                let b_idx = p * b_stride0 + j * b_stride1;
                sum += a_data[a_idx] * b_data[b_idx];
            }
            data[i * n + j] = sum;
        }
    }

    Tensor::new(data, vec![m, n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_add_tensors_rank2(){
        let a = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let sum = add(&a, &b);
        let expected_sum = Tensor::new(vec![5.0, 5.0, 5.0, 5.0], vec![2, 2]);
        assert_eq!(sum.data(), expected_sum.data());
    }


    #[test]
    #[should_panic]
    pub fn test_add_tensors_mismatch(){
        let a = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);

        let _sum = add(&a, &b);
    }


    #[test]
    pub fn test_sub_tensors_rank2(){
        let a = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let dif = sub(&a, &b);
        let expected_dif = Tensor::new(vec![3.0, 1.0, -1.0, -3.0], vec![2, 2]);
        assert_eq!(dif.data(), expected_dif.data());
    }


    #[test]
    #[should_panic]
    pub fn test_dif_tensors_mismatch(){
        let a = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);

        let _sum = sub(&a, &b);
    }

    #[test]
    pub fn test_add_tensors_1d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);

        let sum = add(&a, &b);
        let expected = Tensor::new(vec![5.0, 7.0, 9.0], vec![3]);
        assert_eq!(sum.data(), expected.data());
    }

    #[test]
    pub fn test_sub_tensors_1d() {
        let a = Tensor::new(vec![5.0, 7.0, 9.0], vec![3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

        let dif = sub(&a, &b);
        let expected = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        assert_eq!(dif.data(), expected.data());
    }

    #[test]
    pub fn test_dot_product_2x2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let product = dot(&a, &b);
        let expected = Tensor::new(vec![19.0, 22.0, 43.0, 50.0], vec![2, 2]);
        assert_eq!(product.data(), expected.data());
    }

    #[test]
    pub fn test_dot_product_rectangular() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

        let product = dot(&a, &b);
        let expected = Tensor::new(vec![58.0, 64.0, 139.0, 154.0], vec![2, 2]);
        assert_eq!(product.data(), expected.data());
    }

    #[test]
    #[should_panic]
    pub fn test_dot_product_mismatch_inner() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0], vec![3, 1]);

        let _ = dot(&a, &b);
    }

    #[test]
    pub fn test_add_empty_tensors() {
        let a = Tensor::new(vec![], vec![0]);
        let b = Tensor::new(vec![], vec![0]);

        let sum = add(&a, &b);
        assert!(sum.data().is_empty());
        assert_eq!(sum.shape(), &[0]);
    }

    #[test]
    pub fn test_sub_single_element() {
        let a = Tensor::new(vec![42.0], vec![1]);
        let b = Tensor::new(vec![2.0], vec![1]);

        let dif = sub(&a, &b);
        let expected = Tensor::new(vec![40.0], vec![1]);
        assert_eq!(dif.data(), expected.data());
    }

    #[test]
    pub fn test_mul_tensors_rank2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let product = mul(&a, &b);
        let expected = Tensor::new(vec![5.0, 12.0, 21.0, 32.0], vec![2, 2]);
        assert_eq!(product.data(), expected.data());
    }

    #[test]
    pub fn test_mul_single_element() {
        let a = Tensor::new(vec![3.0], vec![1]);
        let b = Tensor::new(vec![7.0], vec![1]);

        let product = mul(&a, &b);
        let expected = Tensor::new(vec![21.0], vec![1]);
        assert_eq!(product.data(), expected.data());
    }

    #[test]
    pub fn test_mul_empty_tensors() {
        let a = Tensor::new(vec![], vec![0]);
        let b = Tensor::new(vec![], vec![0]);

        let product = mul(&a, &b);
        assert!(product.data().is_empty());
        assert_eq!(product.shape(), &[0]);
    }

    #[test]
    pub fn test_add_3d_tensor() {
        let a = Tensor::new(vec![1.0,2.0,3.0,4.0,5.0,6.0], vec![2,1,3]);
        let b = Tensor::new(vec![6.0,5.0,4.0,3.0,2.0,1.0], vec![2,1,3]);

        let sum = add(&a, &b);
        let expected = Tensor::new(vec![7.0,7.0,7.0,7.0,7.0,7.0], vec![2,1,3]);
        assert_eq!(sum.data(), expected.data());
    }
}
