use std::fmt::Debug;

use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
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
        let weights: Array2<f32> =
            Array::from_shape_fn((num_neurons, inputs), |_| weight_init_value());
        let biases: Array1<f32> = Array1::<f32>::zeros(num_neurons);

        Self { weights, biases }
    }

    pub fn update_weights(&mut self, step: f32, gradients_w: ArrayView2<f32>) {
        self.weights.scaled_add(step, &gradients_w);
    }

    pub fn update_biases(&mut self, step: f32, gradients_b: ArrayView1<f32>) {
        self.biases.scaled_add(step, &gradients_b);
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
    pub fn run_training_epoch(&mut self, training_data_set: &[Sample]) {
        // let mut test = training_data_set.chunks(1);
        // self.train(test.next().unwrap());
        // split training data set to mini batches as a part of SGD
        for mini_batch in training_data_set.chunks(MINI_BATCH_SIZE) {
            self.process_batch(mini_batch);
        }
    }

    pub fn process_batch(&mut self, batch: &[Sample]) {
        //println!("--------Start processing mini batch ------------");
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
        for i in 0..self.layers.len() {
            self.layers[i].update_weights(
                -self.learning_rate / (batch.len() as f32),
                accumulated_grad_w[i].view(),
            );
            self.layers[i].update_biases(
                -self.learning_rate / (batch.len() as f32),
                accumulated_grad_b[i].view(),
            );
        }
        //println!("--------End processing mini batch ------------");
        // Update weights and biases step
    }

    pub fn display_weights(&self, weights: ArrayView1<f32>, height: usize, width: usize) {
        //let image: Array2<f32> = Array2::from_shape_vec((28, 28), pixels.to_vec()).unwrap();
        let image_to_process: Array2<f32> =
            Array2::from_shape_vec((height, width), weights.to_vec()).unwrap();
        // We iterate 2 rows at a time because 1 char = 2 vertical pixels
        for y in (0..height).step_by(2) {
            for x in 0..width {
                let top = image_to_process[[y, x]];
                let bottom = if y + 1 < height {
                    image_to_process[[y + 1, x]]
                } else {
                    0.0
                };

                // ANSI colors: 232-255 are grayscale levels
                let top_color = 232 + (top * 23.0) as u8;
                let bottom_color = 232 + (bottom * 23.0) as u8;

                // \x1b[38;5;...m sets foreground (top)
                // \x1b[48;5;...m sets background (bottom)
                print!("\x1b[38;5;{}m\x1b[48;5;{}m▀", top_color, bottom_color);
            }
            println!("\x1b[0m"); // Reset colors at end of line
        }
    }

    pub fn display_active_weights(&self, layers_activations: &[Array1<f32>], target: &Sample) {
        println!(
            "================= Target Digit: {} ===============",
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
                    // weights: ArrayView1<f32> = self.layers[layer_idx].weights.row(neuron_num);
                    // if layer_idx > 0 {
                    //     self.print_neuron_weights(
                    //         layer_idx,
                    //         self.layers[layer_idx].weights.row(neuron_num).view(),
                    //         num_of_active_neurons,
                    //     );
                    // }
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
            // if layer_idx == 0 {
            //     self.print_neuron_weights(
            //         layer_idx,
            //         neuron_average_weights.view(),
            //         num_of_active_neurons,
            //     );
            // }
        }
    }

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
        // if layer == 0 {
        //     self.display_weights(neuron_average_weights, 28, 28);
        // }

        // if layer == 1 {
        //     self.display_weights(neuron_average_weights, 6, 6);
        // }
    }

    /// Picks (height, width) so that height * width == n and the grid is as square as possible.
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

    pub fn predict(&self, sample: &Sample) -> (usize, Vec<Array1<f32>>) {
        let (layers_activations, _) = self.forward(sample);
        //current_layer_activations
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

    pub fn forward(&self, sample: &Sample) -> (Vec<Array1<f32>>, Vec<Array1<f32>>) {
        // first (input) layer have no the activation method and always equal inputs
        let mut layers_activations: Vec<Array1<f32>> = Vec::new();
        let mut layers_z: Vec<Array1<f32>> = Vec::new();

        let mut current_layer_activations: Array1<f32> = sample.get_image().to_owned();
        //---------Forward step---
        for layer in &self.layers {
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

        (layers_activations, layers_z)
    }
    // * **Forward:** Feed input through each layer; store each layer's output **a** and pre-sigmoid sum **z**. → `forward(network_input)` → `all_layers_outputs`, `all_weighted_sums`
    // * **Deltas:** Output layer: δ = (a − target) ⊙ σ'(z). Hidden layers (from last hidden backward): $\delta^l = (W^{l+1})^T \delta^{l+1} \odot \sigma'(z^l)$. → `compute_deltas(...)`
    // * **Gradients:** ∂C/∂b = δ; ∂C/∂w = δ ⊗ a_prev (outer product). Return gradients for all layers. → `compute_gradients(layers_deltas, all_layers_outputs)`
    pub fn backprop(&self, sample: &Sample) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let (layers_activations, layers_z) = self.forward(sample);

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

    // **Gradients:** ∂C/∂b = δ; ∂C/∂w = δ(layer) ⊗ a(layer-1) (outer product).
    // Outer Dot Product means: The first vector must be a Column (M x 1).The second vector must be a Row (1 x N)
    // The result is matrix (M x N) -> Array2
    // Return gradients for all weights of all layers. → `compute_gradients(layers_deltas, all_layers_outputs)`
    pub fn compute_gradients(
        &self,
        deltas: Vec<Array1<f32>>,
        activations: &[Array1<f32>],
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let mut all_gradients: Vec<Array2<f32>> = Vec::new();

        // deltas have for one element less then activations - but we will not count last activation
        // because it's last layer output activation. The first layer activations - will be pure inputs
        // println!("--------Start ----------");
        for l in 0..self.layers.len() {
            // reshape arrays to matrix form N x 1
            let deltas_matrix = deltas[l].view().insert_axis(Axis(1));
            // reshape arrays to matrix form 1 x M
            let a = activations[l].view().insert_axis(Axis(0));
            // println!("Will find dot product between deltas: {:?}", deltas_matrix);
            // println!("And activations: {:?}", a);
            // result will b e matrix N x M
            let grads = deltas_matrix.dot(&a);

            all_gradients.push(grads);
        }
        // println!("--------End ----------");
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
}
