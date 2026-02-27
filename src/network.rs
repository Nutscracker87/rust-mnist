use std::fmt::Debug;

use ndarray::{Array, Array1, Array2, ArrayView1, Axis};
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

    // Stochastic Gradient Descent (SGD) approach with mini batches
    pub fn run_training_epoch(&self, training_data_set: &[Sample]) {
        // let mut test = training_data_set.chunks(1);
        // self.train(test.next().unwrap());
        // split training data set to mini batches as a part of SGD
        for mini_batch in training_data_set.chunks(MINI_BATCH_SIZE) {
            self.process_batch(mini_batch);
        }
    }

    pub fn process_batch(&self, batch: &[Sample]) {
        let mut accumulated_grad_w: Vec<Array2<f32>> = Vec::new();
        let mut accumulated_grad_b: Vec<Array1<f32>> = Vec::new();

        for (idx, sample) in batch.into_iter().enumerate() {
            let (grads_w, grads_b) = self.backprop(sample);

            // Accumulate Gradients
            if idx == 0 {
                accumulated_grad_w = grads_w;
                accumulated_grad_b = grads_b;
            } else {
                for i in 0..self.layers.len() {
                    accumulated_grad_w[i] += &grads_w[i];
                    accumulated_grad_b[i] += &grads_b[i];
                }
            }
        }

        // Update weights and biases step
    }

    // * **Forward:** Feed input through each layer; store each layer's output **a** and pre-sigmoid sum **z**. → `forward(network_input)` → `all_layers_outputs`, `all_weighted_sums`
    // * **Deltas:** Output layer: δ = (a − target) ⊙ σ'(z). Hidden layers (from last hidden backward): $\delta^l = (W^{l+1})^T \delta^{l+1} \odot \sigma'(z^l)$. → `compute_deltas(...)`
    // * **Gradients:** ∂C/∂b = δ; ∂C/∂w = δ ⊗ a_prev (outer product). Return gradients for all layers. → `compute_gradients(layers_deltas, all_layers_outputs)`
    pub fn backprop(&self, sample: &Sample) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        // let targets = sample.get_label().to_owned();

        // first (input) layer have no the activation method and always equal inputs
        let mut layers_activations: Vec<Array1<f32>> = Vec::with_capacity(self.layers.len() + 1);
        let mut layers_z: Vec<Array1<f32>> = Vec::new();

        let mut current_layer_activations: Array1<f32> = sample.get_image().to_owned();

        //---------Forward step---
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
            layers_activations.push(current_layer_activations);
            current_layer_activations = activations;
            layers_z.push(z);
        }

        layers_activations.push(current_layer_activations);
        //------End of Forward Step----------

        let deltas: Vec<Array1<f32>> =
            self.compute_deltas(&layers_activations, layers_z, sample.get_label());

        // println!("deltas: {:?}", deltas.len());
        // println!("activations: {:?}", layers_activations.len());

        // returns (grads_w, grads_b)
        self.compute_gradients(deltas, &layers_activations)
    }

    pub fn compute_deltas(
        &self,
        layers_activations: &[Array1<f32>],
        layers_z: Vec<Array1<f32>>,
        targets: ArrayView1<f32>,
    ) -> Vec<Array1<f32>> {
        let mut deltas: Vec<Array1<f32>> = Vec::new();
        // Calc deltas
        let last_layer_a: ArrayView1<f32> = layers_activations
            .last()
            .expect("Last layer activations should exists")
            .view();
        let last_layer_z: ArrayView1<f32> = layers_z
            .last()
            .expect("Last layer outputs should exists")
            .view();
        // δ = (a − target) ⊙ σ'(z)
        let last_layer_deltas: Array1<f32> =
            (&last_layer_a - &targets) * last_layer_z.mapv(Network::sigmoid_derivative);

        deltas.push(last_layer_deltas.clone());

        let mut current_layer_deltas: Array1<f32> = last_layer_deltas;
        // hidden layers deltas: $\delta^l = (W^{l+1})^T \delta^{l+1} \odot \sigma'(z^l)$
        for i in (0..self.layers.len() - 1).rev() {
            let layer_to_right = self.layers[i+1].weights.view();
            current_layer_deltas = layer_to_right.t().dot(&current_layer_deltas)
             * layers_z[i].mapv(Network::sigmoid_derivative);

            // current_layer_deltas = d.view();
            deltas.push(current_layer_deltas.clone());
        }

        // since we added elements from the end to start - we need to reverse result
        deltas.reverse();
        deltas
    }

    // **Gradients:** ∂C/∂b = δ; ∂C/∂w = δ(layer) ⊗ a(layer-1) (outer product).
    // Outer Dot Product means: The first vector must be a Column (M x 1).The second vector must be a Row (1 x N)
    // The result is matrix (M x N) -> Array2
    // Return gradients for all weights of all layers. → `compute_gradients(layers_deltas, all_layers_outputs)`
    pub fn compute_gradients(
        &self,
        deltas: Vec<Array1<f32>>,
        activations: &[Array1<f32>],
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let mut all_gradients: Vec<Array2<f32>> = Vec::with_capacity(self.layers.len() - 1);

        // deltas have for one element less then activations - but we will not count last activation
        // because it's last layer output activation. The first layer activations - will be pure inputs
        for l in 0..self.layers.len() {
            // reshape arrays to matrix form N x 1
            let deltas_matrix = deltas[l].view().insert_axis(Axis(1));
            // reshape arrays to matrix form 1 x M
            let a = activations[l].view().insert_axis(Axis(0));
            // result will b e matrix N x M
            let grads = deltas_matrix.dot(&a);

            all_gradients.push(grads);
        }

        // the bias gradients aqual deltas
        let bias_gradients: Vec<Array1<f32>> = deltas;
        
        (all_gradients, bias_gradients)
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
