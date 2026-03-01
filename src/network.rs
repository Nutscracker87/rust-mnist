//! Feed-forward neural network with sigmoid activations, trained by backpropagation
//! and mini-batch stochastic gradient descent (SGD) for MNIST digit classification.

use std::fmt::Debug;

use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::prelude::*;

use crate::sample::Sample;

pub const MAX_EPOCH: usize = 30;
pub const MINI_BATCH_SIZE: usize = 32;

/// A single layer: linear transform (weights, biases) + optional activation applied elsewhere.
#[derive(Debug)]
pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
}

/// Initial weight value (uniform in [-1, 1]). Used for each matrix element.
pub fn weight_init_value() -> f32 {
    let mut rng = rand::rng();
    rng.random_range(-1.0..1.00)
}

impl Layer {
    /// Creates a layer with shape (num_neurons, inputs). Weights filled via `weight_init_value()`.
    pub fn new(inputs: usize, num_neurons: usize) -> Self {
        let weights: Array2<f32> =
            Array::from_shape_fn((num_neurons, inputs), |_| weight_init_value());
        let biases: Array1<f32> = Array1::<f32>::zeros(num_neurons);

        Self { weights, biases }
    }

    /// Gradient descent step: weights -= step * gradients_w (step is negative learning rate).
    pub fn update_weights(&mut self, step: f32, gradients_w: ArrayView2<f32>) {
        self.weights.scaled_add(step, &gradients_w);
    }

    /// Gradient descent step: biases -= step * gradients_b.
    pub fn update_biases(&mut self, step: f32, gradients_b: ArrayView1<f32>) {
        self.biases.scaled_add(step, &gradients_b);
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
    pub learning_rate: f32,
}

impl Network {
    /// Builds a network from layer sizes. E.g. [784, 36, 10] → input 784, hidden 36, output 10.
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

    /// One epoch: process the whole training set in mini-batches (SGD).
    /// Shuffling is done by the caller before each epoch.
    pub fn run_training_epoch(&mut self, training_data_set: &[Sample]) {
        for mini_batch in training_data_set.chunks(MINI_BATCH_SIZE) {
            self.process_batch(mini_batch);
        }
    }

    /// Process one mini-batch: accumulate gradients over samples, then apply one gradient step.
    /// Step formula: θ_new = θ_old - (η/|batch|) * Σ ∇L_i (average of batch gradients × η).
    pub fn process_batch(&mut self, batch: &[Sample]) {
        let mut accumulated_grad_w: Vec<Array2<f32>> = Vec::new();
        let mut accumulated_grad_b: Vec<Array1<f32>> = Vec::new();

        for (idx, sample) in batch.into_iter().enumerate() {
            let (grads_w, grads_b) = self.backprop(sample);

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

        let step = -self.learning_rate / (batch.len() as f32);
        for i in 0..self.layers.len() {
            self.layers[i].update_weights(step, accumulated_grad_w[i].view());
            self.layers[i].update_biases(step, accumulated_grad_b[i].view());
        }
    }

    /// Renders a 1D weight vector as a 2D grayscale grid using ANSI 256-color block chars (▀).
    /// Each character encodes two vertical pixels (top=foreground, bottom=background). Colors 232–255.
    pub fn display_weights(&self, weights: ArrayView1<f32>, height: usize, width: usize) {
        let image_to_process: Array2<f32> =
            Array2::from_shape_vec((height, width), weights.to_vec()).unwrap();
        for y in (0..height).step_by(2) {
            for x in 0..width {
                let top = image_to_process[[y, x]];
                let bottom = if y + 1 < height {
                    image_to_process[[y + 1, x]]
                } else {
                    0.0
                };
                let top_color = 232 + (top * 23.0) as u8;
                let bottom_color = 232 + (bottom * 23.0) as u8;
                print!("\x1b[38;5;{}m\x1b[48;5;{}m▀", top_color, bottom_color);
            }
            println!("\x1b[0m");
        }
    }

    /// For each layer, averages weights of neurons with activation > 0.3 and prints that average.
    pub fn display_active_weights(&self, layers_activations: &[Array1<f32>], target: &Sample) {
        println!(
            "=========== Target Digit: {} ===========",
            target.get_label_as_digit()
        );
        target.display_digit();
        for layer_idx in 0..self.layers.len() {
            let shape = self.layers[layer_idx].weights.row(0).dim();
            let mut neuron_average_weights: Array1<f32> = Array1::zeros(shape);
            let mut num_of_active_neurons = 0;

            for neuron_num in 0..self.layers[layer_idx].weights.nrows() {
                let activation = layers_activations[layer_idx + 1][neuron_num];
                if activation > 0.3 {
                    num_of_active_neurons += 1;
                    neuron_average_weights += &self.layers[layer_idx].weights.row(neuron_num);
                }
            }

            if num_of_active_neurons > 0 {
                neuron_average_weights /= num_of_active_neurons as f32;
            }

            self.print_neuron_weights(
                layer_idx,
                neuron_average_weights.view(),
                num_of_active_neurons,
            );
        }
    }

    /// Prints layer stats and the average-weight grid (grid shape inferred from length).
    pub fn print_neuron_weights(
        &self,
        layer: usize,
        neuron_average_weights: ArrayView1<f32>,
        num_of_active_neurons: usize,
    ) {
        let weights_num = neuron_average_weights.iter().count();
        let weights_activators_num = neuron_average_weights
            .iter()
            .filter(|&x| *x > 1.0 / (weights_num as f32))
            .count();
        println!(
            "Layer: {}, Activated Neurons: {}, Weights Activators: {}. Average Weights Distribution:",
            layer, num_of_active_neurons, weights_activators_num
        );
        let (height, width) = &Self::infer_grid_shape(neuron_average_weights.len());
        self.display_weights(neuron_average_weights, *height, *width);
    }

    /// Chooses (height, width) with height × width = n, as square as possible.
    fn infer_grid_shape(n: usize) -> (usize, usize) {
        if n == 0 {
            return (1, 0);
        }
        let mut h = (n as f32).sqrt().floor() as usize;
        if h == 0 {
            h = 1;
        }
        while n % h != 0 && h > 1 {
            h -= 1;
        }
        let w = n / h;
        (h, w)
    }

    /// Predicts digit by argmax over the last layer activations. Returns (digit, activations).
    pub fn predict(&self, sample: &Sample) -> (usize, Vec<Array1<f32>>) {
        let (layers_activations, _) = self.forward(sample);
        let (predicted, _) = layers_activations
            .last()
            .expect("Last layer activations should exists")
            .iter()
            .enumerate()
            .fold((0, 0.0), |(max_idx, max_val), (idx, &val)| {
                if val > max_val {
                    (idx, val)
                } else {
                    (max_idx, max_val)
                }
            });

        (predicted, layers_activations)
    }

    /// Forward pass: for each layer compute z = W·a_prev + b, then a = σ(z).
    /// Returns (activations per layer including input, pre-activations z per layer).
    /// Layout: activations[0] = input, activations[l+1] = σ(layer_l output).
    pub fn forward(&self, sample: &Sample) -> (Vec<Array1<f32>>, Vec<Array1<f32>>) {
        let mut layers_activations: Vec<Array1<f32>> = Vec::new();
        let mut layers_z: Vec<Array1<f32>> = Vec::new();

        let mut current_layer_activations: Array1<f32> = sample.get_image().to_owned();

        for layer in &self.layers {
            // z^l = W^l · a^{l-1} + b^l
            let z: Array1<f32> = layer.weights.dot(&current_layer_activations) + &layer.biases;
            // a^l = σ(z^l)
            // On each iteration we are adding outputs from previous step (for first layer it's inputs):
            // layer(0).activations = inputs; (input layer without neurons)
            // layer(1).activations = sigmoid(layer(0).z)
            // layer(2).activations = sigmoid(layer(1).z)
            // for last layer it will be out of loop
            let activations: Array1<f32> = z.mapv(Network::sigmoid);
            layers_activations.push(current_layer_activations);
            current_layer_activations = activations;
            layers_z.push(z);
        }

        layers_activations.push(current_layer_activations);

        (layers_activations, layers_z)
    }

    /// Backpropagation: compute gradients of the loss w.r.t. weights and biases.
    /// Steps: (1) forward → activations & z, (2) deltas δ, (3) gradients ∂C/∂w, ∂C/∂b.
    pub fn backprop(&self, sample: &Sample) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let (layers_activations, layers_z) = self.forward(sample);

        let deltas: Vec<Array1<f32>> =
            self.compute_deltas(&layers_activations, layers_z, sample.get_label());

        self.compute_gradients(deltas, &layers_activations)
    }

    /// Output layer: δ^L = (a^L − y) ⊙ σ'(z^L). Hidden: δ^l = (W^{l+1})ᵀ δ^{l+1} ⊙ σ'(z^l).
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
            let layer_to_right = self.layers[i + 1].weights.view();
            current_layer_deltas = layer_to_right.t().dot(&current_layer_deltas)
                * layers_z[i].mapv(Network::sigmoid_derivative);

            // current_layer_deltas = d.view();
            deltas.push(current_layer_deltas.clone());
        }
        // since we added elements from the end to start - we need to reverse result
        deltas.reverse();
        deltas
    }

    // Gradients: dC/db = delta; dC/dw = outer product delta * a_prev. Matrix (neurons x inputs).
    // Outer Dot Product means: The first vector must be a Column (M x 1).The second vector must be a Row (1 x N)
    // The result is matrix (M x N) -> Array2
    // Return gradients for all weights of all layers. → `compute_gradients(layers_deltas, all_layers_outputs)`
    pub fn compute_gradients(
        &self,
        deltas: Vec<Array1<f32>>,
        activations: &[Array1<f32>],
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let mut all_gradients: Vec<Array2<f32>> = Vec::new();

        // deltas have one element less then activations - but we will not count last activation
        // because it's last layer output - activation. The first layer activations - will be pure inputs
        for l in 0..self.layers.len() {
            // ∂C/∂w^l = δ^l (col) · (a^{l-1})ᵀ (row) → shape (neurons_l, inputs_l)
            let deltas_matrix = deltas[l].view().insert_axis(Axis(1));
            let a = activations[l].view().insert_axis(Axis(0));
            let grads = deltas_matrix.dot(&a);
            all_gradients.push(grads);
        }

        let bias_gradients: Vec<Array1<f32>> = deltas;

        (all_gradients, bias_gradients)
    }

    /// Sigmoid: σ(z) = 1 / (1 + e^{-z}). Output in (0, 1).
    pub fn sigmoid(z: f32) -> f32 {
        1.0 / (1.0 + (-z).exp())
    }

    /// σ'(z) = σ(z)(1 − σ(z)).
    pub fn sigmoid_derivative(z: f32) -> f32 {
        Network::sigmoid(z) * (1.0 - Network::sigmoid(z))
    }
}
