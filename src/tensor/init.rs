use rand::Rng;
use rand_distr::Distribution;
use rand_distr::Normal;

use crate::tensor::core::Tensor;

pub fn n_tensor (n: f32, shape: Vec<usize>) -> Tensor {
    Tensor::new(vec![n; shape.iter().product()], shape)
}

pub fn rand_uni_tensor (shape: Vec<usize>, low: f32, high: f32) -> Tensor {
    assert!(low < high, "Low must be < high");

    let len: usize = shape.iter().product();
    let mut rng = rand::rng();


    let data: Vec<f32> = (0..len)
        .map(|_| rng.random_range(low..high))
        .collect();
    
    Tensor::new(data, shape)
}

pub fn rand_norm_tensor (shape: Vec<usize>, mean: f32, std: f32) -> Tensor {
    assert!(std > 0.0, "std must be > 0");

    let len: usize = shape.iter().product();
    let normal = Normal::new(mean, std).unwrap();
    let mut rng = rand::rng();


    let data: Vec<f32> = (0..len)
        .map(|_| normal.sample(&mut rng))
        .collect();
    
    Tensor::new(data, shape)
}

#[cfg(test)]
mod tests {
    use crate::utils::float::{ tensor_allclose};

    use super::*;

    #[test]
    fn test_init_n_works () {
        let zero_tens = n_tensor(1.0, vec![1]);
        assert!(tensor_allclose(&zero_tens, &Tensor::new(vec![1.0], vec![1]), 0.001))
    }

    #[test]
    fn test_rand_uniform_shape() {
        let t = rand_uni_tensor(vec![4, 5], -1.0, 1.0);
        assert_eq!(t.shape(), &[4, 5]);
        assert_eq!(t.data().len(), 20);

        for &v in t.data() {
            assert!((-1.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_rand_normal_shape() {
        let t = rand_norm_tensor(vec![3, 3], 0.0, 1.0);
        assert_eq!(t.shape(), &[3, 3]);
        assert_eq!(t.data().len(), 9);
    }
}
