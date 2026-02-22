use ndarray::{Array, Array1, Array2, ArrayBase, ArrayView2, Axis};
use rand::prelude::*;

#[derive(Debug)]
pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
}

pub fn weight_init_value() -> f32 {
    let mut rng = rand::rng();
    rng.random_range(-1.0..1.00)
}

pub fn bias_init_value() -> f32 {
    let mut rng = rand::rng();
    rng.random_range(-1.0..1.00)
}

impl Layer {
    pub fn new(inputs: usize, num_neurons: usize) -> Self {
        let weights: Array2<f32> =
            Array::from_shape_fn((num_neurons, inputs), |_| weight_init_value());
        let biases: Array1<f32> = Array::from_shape_fn(inputs, |_| bias_init_value());
        
        Self {
            weights,
            biases
        }
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
    pub learning_rate: f32,
}

impl Network {
    pub fn new(shape: &[usize], learning_rate: f32) -> Self {
        let mut layers: Vec<Layer> = Vec::new();
        for i in 0..shape.len() - 1 {
            layers.push(Layer::new(shape[i], shape[i + 1]));
        }

        Self {
            layers,
            learning_rate
        }
    }
}
