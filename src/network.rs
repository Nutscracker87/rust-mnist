use ndarray::{Array, Array1, Array2, ArrayBase, ArrayView2, Axis};
use rand::prelude::*;

use crate::sample::Sample;

pub const MAX_EPOCH: usize = 30;
pub const MINI_BATCH_SIZE: usize = 32;

#[derive(Debug)]
pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
}

pub fn weight_init_value() -> f32 {
    let mut rng = rand::rng();
    rng.random_range(-1.0..1.00)
}

impl Layer {
    pub fn new(inputs: usize, num_neurons: usize) -> Self {
        let mut weights: Array2<f32> =
            Array::from_shape_fn((num_neurons, inputs), |_| weight_init_value());
        let mut biases: Array1<f32> = Array1::<f32>::zeros(num_neurons);

        Self { weights, biases }
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
            learning_rate,
        }
    }

    // Stochastic Gradient Descent (SGD)
    pub fn sgd(&self, training_data_set: &[Sample], eta: f32) {
        // let mut test = training_data_set.chunks(1);
        // self.train(test.next().unwrap());
        // split training data set to mini batches as a part of SGD
        for mini_batch in training_data_set.chunks(MINI_BATCH_SIZE) {
            self.train(mini_batch);
        }
    }

    pub fn train(&self, batch: &[Sample]) {
        for sample in batch {
            self.backprop(sample);
        }
        // let deltas = Array2::try_from(value)
        // self.forward();
    }

    // * **Forward:** Feed input through each layer; store each layer's output **a** and pre-sigmoid sum **z**. → `forward(network_input)` → `all_layers_outputs`, `all_weighted_sums`
    // * **Deltas:** Output layer: δ = (a − target) ⊙ σ'(z). Hidden layers (from last hidden backward): $\delta^l = (W^{l+1})^T \delta^{l+1} \odot \sigma'(z^l)$. → `compute_deltas(...)`
    // * **Gradients:** ∂C/∂b = δ; ∂C/∂w = δ ⊗ a_prev (outer product). Return gradients for all layers. → `compute_gradients(layers_deltas, all_layers_outputs)`
    pub fn backprop(&self, sample: &Sample) {
        let targets = sample.get_label().to_owned();

        // first layer have no activation method and always equal inputs
        let mut layer_activations: Vec<Array1<f32>> = Vec::with_capacity(self.layers.len() + 1);//vec![sample.get_image().to_owned()];
        let mut layers_z: Vec<Array1<f32>> = Vec::new();

        let mut current_layer_activations: Array1<f32> = sample.get_image().to_owned();

        // Forward step
        for layer in &self.layers {
            // println!("cur layer a: {:?}", current_layer_activations.shape());
            // println!("cur layer weights: {:?}", layer.weights);
            // println!("cur layer biases: {:?}", layer.biases);
            let z: Array1<f32> = layer.weights.dot(&current_layer_activations) + &layer.biases;
            let activations: Array1<f32> = z.mapv(Network::sigmoid);
            // On each iteration we are adding outputs from previous step (for first layer it's inputs): 
            // layer(0).activations = inputs; (input layer without neurons)
            // layer(1).activations = sigmoid(layer(0).z)
            // layer(2).activations = sigmoid(layer(1).z)
            // for last layer it will be out of loop
            layer_activations.push(current_layer_activations);
            current_layer_activations = activations;
            layers_z.push(z);
        }
        
        layer_activations.push(current_layer_activations);

        let mut deltas: Vec<Array1<f32>> = Vec::new();
        let mut grads: Vec<Array1<f32>> = Vec::new();

        // Calc deltas
        let last_layer_a: Array1<f32> = layer_activations.last().expect("Last layer activations should exists").to_owned();
        let last_layer_z: Array1<f32> = layers_z.last().expect("Last layer outputs should exists").to_owned();
        // δ = (a − target) ⊙ σ'(z)
        let last_layer_deltas = (last_layer_a - targets) * last_layer_z.mapv(Network::sigmoid_derivative);
        deltas.push(last_layer_deltas);

        println!("deltas {:?}", deltas);
        //println!("z: {:?}", layer_z.last().unwrap());
        // println!("Activations: {:?}", layer_activations.last().unwrap());
    }

    /// Sigmoid activation: σ(z) = 1 / (1 + e^(-z)). Maps any real number to (0, 1).
    /// Used so outputs are bounded and differentiable everywhere.
    pub fn sigmoid(z: f32) -> f32 {
        1.0 / (1.0 + (-z).exp())
    }

    pub fn sigmoid_derivative(z: f32) -> f32 {
        Network::sigmoid(z) * (1.0 - Network::sigmoid(z))
    }

    // pub fn backprop(&self) {

    // }
}
