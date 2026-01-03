use crate::tensor::init::n_tensor;
use crate::tensor::init::rand_norm_tensor;
use crate::tensor::init::rand_uni_tensor;
use crate::tensor::reductions;
use crate::tensor::nn;

#[derive(Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    stride: Vec<usize>
} 



impl Tensor {
    pub fn shape(&self) -> &[usize]{
        &self.shape
    }

    pub fn data(&self) -> &[f32]{
        &self.data
    }


    pub fn stride(&self) -> &[usize]{
        &self.stride
    }

    pub fn n_tensor (n: f32, shape: Vec<usize>) -> Tensor {
        n_tensor(n, shape)
    }

    pub fn rand_norm (shape: Vec<usize>, mean: f32, std: f32) -> Tensor {
        rand_norm_tensor(shape, mean, std)
    }

    pub fn rand_uni (shape: Vec<usize>, low: f32, high: f32) -> Tensor {
        rand_uni_tensor(shape, low, high)
    }

    pub fn compute_stride(shape: &[usize])-> Vec<usize>{
        let mut stride = vec![0; shape.len()];

        let mut acc = 1;

        for i in (0..shape.len()).rev() {
            stride[i] = acc;
            acc *= shape[i];
        }

        stride
    }

    pub fn new_with_stride(data: Vec<f32>, shape: Vec<usize>, stride: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "Data length mismatch");
        Self { data, shape, stride }
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "Data length mismatch");
        let stride = Self::compute_stride(&shape);
        Self { data, shape, stride }
    }
    pub fn index (&self, idx: &[usize]) -> usize {
        assert_eq!(idx.len(), self.shape.len(), "Index rank mismatch");
        let flat_index: usize = idx.iter()
                                .zip(&self.stride)
                                .map(|(i, s)| i * s)
                                .sum();
        flat_index
    }

    pub fn mean_last_dim(&self) -> Tensor {
        let rank = self.shape.len();
        assert!(rank >= 1, "Tensor must have at least 1 dimension");

        let last_dim = self.shape[rank - 1];
        let outer_size: usize = self.shape[..rank-1].iter().product();

        let mut data = Vec::with_capacity(outer_size);

        for i in 0..outer_size {
            let start = i * last_dim;
            let end = start + last_dim;
            let slice = &self.data[start..end];
            let sum: f32 = slice.iter().sum();
            data.push(sum / last_dim as f32);
        }

        let mut shape = self.shape.clone();
        shape[rank-1] = 1;

        Tensor::new(data, shape)
    }

    pub fn sqrt(&self) -> Tensor {
        let data: Vec<f32> = self.data().iter().map(|&x| x.sqrt()).collect();
        Tensor::new(data, self.shape().to_vec())
    }

    pub fn var_last_dim(&self) -> Tensor {
        let rank = self.shape.len();
        assert!(rank >= 1, "Tensor must have at least 1 dimension");

        let last_dim = self.shape[rank - 1];
        let outer_size: usize = self.shape[..rank-1].iter().product();

        let mut data = Vec::with_capacity(outer_size);

        let mean_tensor = self.mean_last_dim();
        let mean_data = mean_tensor.data();

        for (i, &mean_val) in mean_data.iter().enumerate() {
            let start = i * last_dim;
            let end = start + last_dim;
            let slice = &self.data[start..end];

            let mut sum_sq_diff = 0.0;
            for &v in slice {
                let diff = v - mean_val;
                sum_sq_diff += diff * diff;
            }

            data.push(sum_sq_diff / last_dim as f32);
        }

        let mut shape = self.shape.clone();
        shape[rank-1] = 1;

        Tensor::new(data, shape)
    }

    pub fn is_finite(&self) -> bool {
        reductions::is_finite(self)
    }

    pub fn max(&self) -> Option<f32> {
        reductions::max(self)
    }

    pub fn softmax(&self) -> Tensor {
        nn::softmax(self)
    }

    pub fn softmax_axis(&self, axis: isize) -> Tensor {
        nn::softmax_axis(self, axis)
    }

    pub fn normalize(&self) -> Tensor {
        nn::normalize(self)
    }

    pub fn set (&mut self, idx: &[usize], val: f32) {
        let flat_index = self.index(idx);
        self.data[flat_index] = val;
    }

    pub fn get (&self, idx: &[usize]) -> f32{
        let flat_index = self.index(idx);
        self.data[flat_index]
    }

    pub fn row(&self, row: usize) -> Vec<f32> {
        assert_eq!(self.shape.len(), 2, "row() requires a rank-2 tensor");
        assert!(row < self.shape[0], "row index out of bounds");

        let cols = self.shape[1];
        let start = row * cols;
        let end = start + cols;

        self.data[start..end].to_vec()
    }

    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        assert!(scalar != 0.0);
        let inv = 1.0 / scalar;
        let data = self.data.iter().map(|x| x * inv).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn slice_axis(&self, axis: usize, index: usize) -> Tensor {
        let rank = self.shape.len();
        assert!(axis < rank, "Axis out of bounds");
        assert!(index < self.shape[axis], "Index out of bounds");

        let mut new_shape = self.shape.clone();
        new_shape.remove(axis);

        let new_size: usize = new_shape.iter().product();
        let mut new_data = Vec::with_capacity(new_size);

        let mut idx = vec![0; rank];
        idx[axis] = index;

        for _ in 0..new_size {
            let flat_index = self.index(&idx);
            new_data.push(self.data[flat_index]);

            for dim in (0..rank).rev() {
                if dim == axis {
                    continue;
                }
                idx[dim] += 1;
                if idx[dim] < self.shape[dim] {
                    break;
                }
                idx[dim] = 0;
            }
        }

        Tensor::new(new_data, new_shape)
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let old_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();

        assert!(
            old_size == new_size,
            "Cannot reshape tensor: size mismatch ({} != {})",
            old_size,
            new_size
        );

        Tensor::new(self.data.clone(), new_shape)
    }

    pub fn transpose(&self, perm: Vec<usize>) -> Tensor {
        let rank = self.shape.len();
        assert_eq!(perm.len(), rank, "Permutation length must equal tensor rank");

        let new_shape: Vec<usize> = perm.iter().map(|&i| self.shape[i]).collect();
        let new_stride = Tensor::compute_stride(&new_shape);
        let new_size: usize = new_shape.iter().product();
        
        let mut new_data = Vec::with_capacity(new_size);
        let mut idx = vec![0; rank];

        let permuted_old_strides: Vec<usize> = perm.iter().map(|&p| self.stride[p]).collect();

        for _ in 0..new_size {
            let mut old_flat_index = 0;
            for (dim, &count) in idx.iter().enumerate() {
                old_flat_index += count * permuted_old_strides[dim];
            }
            
            new_data.push(self.data[old_flat_index]);

            for dim in (0..rank).rev() {
                idx[dim] += 1;
                if idx[dim] < new_shape[dim] {
                    break;
                }
                idx[dim] = 0;
            }
        }

        Tensor::new_with_stride(new_data, new_shape, new_stride)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_initializations() {
        let tens = Tensor::new(vec![0.0, 0.1, 1.0, 1.1], vec![2, 2]);
        assert_eq!(tens.get(&[0,0]), 0.0);
        assert_eq!(tens.get(&[0,1]), 0.1);
        assert_eq!(tens.get(&[1,0]), 1.0);
        assert_eq!(tens.get(&[1,1]), 1.1);
    }

    #[test]
    fn test_tensor_rank1() {
        let tens = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
        assert_eq!(tens.get(&[0]), 10.0);
        assert_eq!(tens.get(&[1]), 20.0);
        assert_eq!(tens.get(&[2]), 30.0);
    }

    #[test]
    fn test_tensor_rank3() {
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let tens = Tensor::new(data, vec![2, 2, 2]);
        assert_eq!(tens.get(&[0,0,0]), 0.0);
        assert_eq!(tens.get(&[0,0,1]), 1.0);
        assert_eq!(tens.get(&[0,1,0]), 2.0);
        assert_eq!(tens.get(&[1,0,1]), 5.0);
        assert_eq!(tens.get(&[1,1,1]), 7.0);
    }

    #[test]
    #[should_panic]
    fn test_get_out_of_bounds() {
        let tens = Tensor::new(vec![1.0, 2.0], vec![2]);
        tens.get(&[2]);
    }

    #[test]
    #[should_panic]
    fn test_get_wrong_rank() {
        let tens = Tensor::new(vec![1.0, 2.0], vec![2]);
        tens.get(&[0, 1]);
    }

    #[test]
    #[should_panic]
    fn test_constructor_shape_mismatch() {
        let _tens = Tensor::new(vec![1.0, 2.0, 3.0], vec![2, 2]);
    }

    #[test]
    fn test_empty_tensor() {
        let tens = Tensor::new(vec![], vec![0]);
        assert_eq!(tens.data.len(), 0);
    }

    #[test]
    fn test_set_and_get() {
        let mut tens = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);

        tens.set(&[0, 0], 1.1);
        tens.set(&[0, 1], 2.2);
        tens.set(&[1, 0], 3.3);
        tens.set(&[1, 1], 4.4);

        assert_eq!(tens.get(&[0,0]), 1.1);
        assert_eq!(tens.get(&[0,1]), 2.2);
        assert_eq!(tens.get(&[1,0]), 3.3);
        assert_eq!(tens.get(&[1,1]), 4.4);
    }

    #[test]
    fn test_overwrite_values() {
        let mut tens = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        tens.set(&[0,0], 10.0);
        tens.set(&[1,1], 20.0);
        assert_eq!(tens.get(&[0,0]), 10.0);
        assert_eq!(tens.get(&[1,1]), 20.0);
    }

    #[test]
    #[should_panic]
    fn test_set_out_of_bounds() {
        let mut tens = Tensor::new(vec![1.0, 2.0], vec![2]);
        tens.set(&[2], 5.0);
    }

    #[test]
    #[should_panic]
    fn test_set_wrong_rank() {
        let mut tens = Tensor::new(vec![1.0, 2.0], vec![2]);
        tens.set(&[0, 1], 5.0);
    }

    #[test]
    fn test_set_and_get_rank3() {
        let mut tens = Tensor::new(vec![0.0; 8], vec![2, 2, 2]);
        tens.set(&[1,1,1], 7.7);
        assert_eq!(tens.get(&[1,1,1]), 7.7);
        tens.set(&[0,0,0], 0.1);
        assert_eq!(tens.get(&[0,0,0]), 0.1);
    }

    #[test]
    fn test_tensor_methods() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
        
        assert!(t.is_finite());
        
        assert_eq!(t.max(), Some(3.0));
        
        let s = t.softmax();
        let sum: f32 = s.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        let n = t.normalize();
        assert!((n.data()[0] - 0.0).abs() < 1e-6);
        assert!((n.data()[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_row() {
        let t = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2]
        );

        assert_eq!(t.row(0), vec![1.0, 2.0]);
        assert_eq!(t.row(1), vec![3.0, 4.0]);
        assert_eq!(t.row(2), vec![5.0, 6.0]);
    }

    const EPS: f32 = 1e-6;

    #[test]
    fn mean_var_last_dim_1d() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        let mean = t.mean_last_dim();
        let var = t.var_last_dim();

        assert_eq!(mean.shape(), &[1]);
        assert_eq!(var.shape(), &[1]);

        assert!((mean.data()[0] - 2.5).abs() < EPS);
        assert!((var.data()[0] - 1.25).abs() < EPS);
    }

    #[test]
    fn mean_var_last_dim_2d() {
        let t = Tensor::new(
            vec![
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
            ],
            vec![3, 2],
        );

        let mean = t.mean_last_dim();
        let var = t.var_last_dim();

        assert_eq!(mean.shape(), &[3, 1]);
        assert_eq!(var.shape(), &[3, 1]);

        let expected_means = [1.5, 3.5, 5.5];
        let expected_vars = [0.25, 0.25, 0.25];

        for i in 0..3 {
            assert!((mean.data()[i] - expected_means[i]).abs() < EPS);
            assert!((var.data()[i] - expected_vars[i]).abs() < EPS);
        }
    }

    #[test]
    fn mean_var_last_dim_3d() {
        const EPS: f32 = 1e-6;

        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
                10.0, 11.0, 12.0
            ],
            vec![2, 2, 3],
        );

        let mean = t.mean_last_dim();
        let var = t.var_last_dim();

        assert_eq!(mean.shape(), &[2, 2, 1]);
        assert_eq!(var.shape(), &[2, 2, 1]);

        let expected_means = [2.0, 5.0, 8.0, 11.0];
        let expected_var = 2.0 / 3.0;

        for ((&mean_val, &var_val), &expected_mean) in
            mean.data()
                .iter()
                .zip(var.data())
                .zip(expected_means.iter())
        {
            assert!((mean_val - expected_mean).abs() < EPS);
            assert!((var_val - expected_var).abs() < EPS);
        }
    }

    #[test]
    fn variance_zero_for_constant_tensor() {
        let t = Tensor::new(vec![7.0; 12], vec![3, 4]);

        let var = t.var_last_dim();

        for &v in var.data() {
            assert!(v.abs() < EPS);
        }
    }

    #[test]
    fn mean_var_shapes_preserved() {
        let t = Tensor::new(vec![0.0; 24], vec![2, 3, 4]);

        let mean = t.mean_last_dim();
        let var = t.var_last_dim();

        assert_eq!(mean.shape(), &[2, 3, 1]);
        assert_eq!(var.shape(), &[2, 3, 1]);
    }

    #[test]
    fn mean_var_work_with_single_last_dim() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);

        let mean = t.mean_last_dim();
        let var = t.var_last_dim();

        assert_eq!(mean.shape(), &[3, 1]);
        assert_eq!(var.shape(), &[3, 1]);

        for i in 0..3 {
            assert!((mean.data()[i] - t.data()[i]).abs() < EPS);
            assert!(var.data()[i].abs() < EPS);
        }
    }

    #[test]
    fn mean_var_are_finite() {
        let t = Tensor::new(
            vec![
                1e6, 1e6 + 1.0,
                -1e6, -1e6 + 1.0
            ],
            vec![2, 2],
        );

        let mean = t.mean_last_dim();
        let var = t.var_last_dim();

        assert!(mean.is_finite());
        assert!(var.is_finite());
    }

    #[test]
    fn test_sqrt_1d() {
        let t = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![4]);
        let s = t.sqrt();
        let expected = [1.0, 2.0, 3.0, 4.0];

        for (a, &b) in s.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < EPS);
        }
        assert_eq!(s.shape(), &[4]);
    }

    #[test]
    fn test_sqrt_2d() {
        let t = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![2, 2]);
        let s = t.sqrt();
        let expected = [1.0, 2.0, 3.0, 4.0];

        for (a, &b) in s.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < EPS);
        }
        assert_eq!(s.shape(), &[2, 2]);
    }

    #[test]
    fn test_sqrt_3d() {
        let t = Tensor::new((1..9).map(|x| (x * x) as f32).collect(), vec![2, 2, 2]);
        let s = t.sqrt();
        let expected = (1..9).map(|x| x as f32).collect::<Vec<f32>>();

        for (a, &b) in s.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < EPS);
        }
        assert_eq!(s.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_sqrt_zero() {
        let t = Tensor::new(vec![0.0, 0.0], vec![2]);
        let s = t.sqrt();
        assert_eq!(s.data(), &[0.0, 0.0]);
    }

    #[test]
    fn test_sqrt_empty() {
        let t = Tensor::new(vec![], vec![0]);
        let s = t.sqrt();
        assert!(s.data().is_empty());
        assert_eq!(s.shape(), &[0]);
    }

    #[test]
    fn test_reshape_2d_to_1d() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = t.reshape(vec![4]);
        assert_eq!(r.shape(), &[4]);
        assert_eq!(r.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reshape_1d_to_2d() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
        let r = t.reshape(vec![2, 3]);
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_2d_to_3d() {
        let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);
        let r = t.reshape(vec![3, 1, 2]);
        assert_eq!(r.shape(), &[3, 1, 2]);
        assert_eq!(r.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_reshape_preserves_data() {
        let t = Tensor::new((1..13).map(|x| x as f32).collect(), vec![3, 4]);
        let r = t.reshape(vec![2, 2, 3]);
        assert_eq!(r.data(), t.data());
    }

    #[test]
    fn test_reshape_single_element() {
        let t = Tensor::new(vec![42.0], vec![1]);
        let r = t.reshape(vec![1, 1]);
        assert_eq!(r.shape(), &[1, 1]);
        assert_eq!(r.data(), &[42.0]);
    }

    #[test]
    #[should_panic]
    fn test_reshape_mismatch_panics() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let _ = t.reshape(vec![3]);
    }

    #[test]
    fn test_reshape_to_same_shape() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = t.reshape(vec![2, 2]);
        assert_eq!(r.shape(), t.shape());
        assert_eq!(r.data(), t.data());
    }

    #[test]
    fn test_reshape_zero_elements() {
        let t = Tensor::new(vec![], vec![0]);
        let r = t.reshape(vec![0]);
        assert_eq!(r.shape(), &[0]);
        assert!(r.data().is_empty());
    }

    #[test]
    fn test_transpose_2d() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        // Shape [2, 3]
        // [[1, 2, 3],
        //  [4, 5, 6]]
        
        let t_t = t.transpose(vec![1, 0]);
        // Shape [3, 2]
        // [[1, 4],
        //  [2, 5],
        //  [3, 6]]
        
        assert_eq!(t_t.shape(), &[3, 2]);
        assert_eq!(t_t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_3d() {
        let t = Tensor::new((0..24).map(|x| x as f32).collect(), vec![2, 3, 4]);
        // Permute to [4, 2, 3] -> perm [2, 0, 1]
        let t_t = t.transpose(vec![2, 0, 1]);
        assert_eq!(t_t.shape(), &[4, 2, 3]);
        
        // Check a specific value
        // Old index [1, 2, 3] -> value 1*12 + 2*4 + 3 = 12 + 8 + 3 = 23.
        // New index should be [3, 1, 2] (since new dims are old dims 2, 0, 1)
        // Let's verify: new_idx[0] = old_idx[2] = 3. new_idx[1] = old_idx[0] = 1. new_idx[2] = old_idx[1] = 2.
        
        assert_eq!(t.get(&[1, 2, 3]), 23.0);
        assert_eq!(t_t.get(&[3, 1, 2]), 23.0);
    }

    #[test]
    fn test_slice_axis_2d() {
        // 2x3 tensor
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        // Shape: [[1,2,3],
        //         [4,5,6]]

        // Slice axis 0 (rows)
        let row0 = t.slice_axis(0, 0);
        assert_eq!(row0.shape(), &[3]);
        assert_eq!(row0.data(), &[1.0, 2.0, 3.0]);

        let row1 = t.slice_axis(0, 1);
        assert_eq!(row1.shape(), &[3]);
        assert_eq!(row1.data(), &[4.0, 5.0, 6.0]);

        // Slice axis 1 (columns)
        let col0 = t.slice_axis(1, 0);
        assert_eq!(col0.shape(), &[2]);
        assert_eq!(col0.data(), &[1.0, 4.0]);

        let col2 = t.slice_axis(1, 2);
        assert_eq!(col2.shape(), &[2]);
        assert_eq!(col2.data(), &[3.0, 6.0]);
    }

    #[test]
    fn test_slice_axis_3d() {
        // 2x2x3 tensor
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0,   // first row, first depth
                4.0, 5.0, 6.0,   // first row, second depth
                7.0, 8.0, 9.0,   // second row, first depth
                10.0, 11.0, 12.0 // second row, second depth
            ],
            vec![2, 2, 3],
        );

        // Shape: [2,2,3]

        // Slice first axis (0)
        let slice0 = t.slice_axis(0, 0);
        assert_eq!(slice0.shape(), &[2, 3]);
        assert_eq!(slice0.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let slice1 = t.slice_axis(0, 1);
        assert_eq!(slice1.shape(), &[2, 3]);
        assert_eq!(slice1.data(), &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        // Slice second axis (1)
        let slice_axis1_0 = t.slice_axis(1, 0);
        assert_eq!(slice_axis1_0.shape(), &[2, 3]);
        assert_eq!(slice_axis1_0.data(), &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);

        // Slice third axis (2)
        let slice_axis2_1 = t.slice_axis(2, 1);
        assert_eq!(slice_axis2_1.shape(), &[2, 2]);
        assert_eq!(slice_axis2_1.data(), &[2.0, 5.0, 8.0, 11.0]);
    }

    #[test]
    #[should_panic]
    fn test_slice_axis_out_of_bounds_axis() {
        let t = Tensor::new(vec![1.0,2.0,3.0,4.0], vec![2,2]);
        t.slice_axis(2, 0); // invalid axis
    }

    #[test]
    #[should_panic]
    fn test_slice_axis_out_of_bounds_index() {
        let t = Tensor::new(vec![1.0,2.0,3.0,4.0], vec![2,2]);
        t.slice_axis(0, 5); // invalid index
    }


    #[test]
    fn test_div_scalar_1d() {
        let t = Tensor::new(vec![2.0, 4.0, 6.0], vec![3]);
        let r = t.div_scalar(2.0);

        let expected = [1.0, 2.0, 3.0];
        for (a, &b) in r.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        assert_eq!(r.shape(), &[3]);
    }

    #[test]
    fn test_div_scalar_2d() {
        let t = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]);
        let r = t.div_scalar(2.0);

        let expected = [1.0, 2.0, 3.0, 4.0];
        for (a, &b) in r.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        assert_eq!(r.shape(), &[2, 2]);
    }

    #[test]
    fn test_div_scalar_3d() {
        let t = Tensor::new((1..9).map(|x| x as f32).collect(), vec![2, 2, 2]);
        let r = t.div_scalar(2.0);

        for (i, &v) in t.data().iter().enumerate() {
            assert!((r.data()[i] - v / 2.0).abs() < 1e-6);
        }

        assert_eq!(r.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_div_scalar_identity() {
        let t = Tensor::new(vec![1.5, -2.5, 3.5], vec![3]);
        let r = t.div_scalar(1.0);

        for (a, &b) in r.data().iter().zip(t.data().iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_div_scalar_negative_values() {
        let t = Tensor::new(vec![-2.0, 2.0, -4.0, 4.0], vec![4]);
        let r = t.div_scalar(2.0);

        let expected = [-1.0, 1.0, -2.0, 2.0];
        for (a, &b) in r.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_div_scalar_empty_tensor() {
        let t = Tensor::new(vec![], vec![0]);
        let r = t.div_scalar(2.0);

        assert!(r.data().is_empty());
        assert_eq!(r.shape(), &[0]);
    }
}

