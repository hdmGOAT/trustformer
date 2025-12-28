struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    stride: Vec<usize>
} 



impl Tensor {
    fn to_stride(shape: &[usize])-> Vec<usize>{
        let mut stride = vec![0; shape.len()];

        let mut acc = 1;

        for i in (0..shape.len()).rev() {
            stride[i] = acc;
            acc *= shape[i];
        }

        stride
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "Data length mismatch");
        let stride = Self::to_stride(&shape);
        Self { data, shape, stride }
    }
    pub fn index (&self, idx: &[usize]) -> f32 {
        assert_eq!(idx.len(), self.shape.len(), "Index rank mismatch");
        let flat_index: usize = idx.iter()
                                .zip(&self.stride)
                                .map(|(i, s)| i * s)
                                .sum();
        self.data[flat_index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_initializations() {
        let tens = Tensor::new(vec![0.0, 0.1, 1.0, 1.1], vec![2, 2]);

        assert_eq!(tens.index(&[0,0]), 0.0);
        assert_eq!(tens.index(&[0,1]), 0.1);
        assert_eq!(tens.index(&[1,0]), 1.0);
        assert_eq!(tens.index(&[1,1]), 1.1);
    }

    #[test]
    fn test_tensor_rank1() {
        let tens = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
        assert_eq!(tens.index(&[0]), 10.0);
        assert_eq!(tens.index(&[1]), 20.0);
        assert_eq!(tens.index(&[2]), 30.0);
    }

    #[test]
    fn test_tensor_rank3() {
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let tens = Tensor::new(data, vec![2, 2, 2]);

        assert_eq!(tens.index(&[0,0,0]), 0.0);
        assert_eq!(tens.index(&[0,0,1]), 1.0);
        assert_eq!(tens.index(&[0,1,0]), 2.0);
        assert_eq!(tens.index(&[1,0,1]), 5.0);
        assert_eq!(tens.index(&[1,1,1]), 7.0);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds() {
        let tens = Tensor::new(vec![1.0, 2.0], vec![2]);
        tens.index(&[2]);
    }

    #[test]
    #[should_panic]
    fn test_index_wrong_rank() {
        let tens = Tensor::new(vec![1.0, 2.0], vec![2]);
        tens.index(&[0, 1]);
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
}
