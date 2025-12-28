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
        let stride = Self::to_stride(&shape);
        Self { data, shape, stride }
    }
    pub fn index (&self, idx: &[usize]) -> f32 {
        let flat_index: usize = idx.iter()
                                .zip(&self.shape)
                                .map(|(i, s)| i * s)
                                .sum();
        self.data[flat_index]
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test_tensor_initializations(){
        let mut tens = Tensor::new(vec![0.0, 0.1, 1.0, 1.1], vec![2, 2]);
        
        assert_eq!(tens.index(&[0,0]), 0.0)
    }
}
