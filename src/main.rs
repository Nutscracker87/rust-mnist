use std::time::Instant;

use rand::{RngExt, seq::SliceRandom};

use crate::{network::Network, sample::Sample};

mod data_loader;
pub mod network;
pub mod sample;

/// Train a small MLP on MNIST and report accuracy each epoch. Optionally print weight visualisations at the end.
fn main() {
    println!("Hello, my new neural network!");
    let mut data = data_loader::MnistData::new();
    let mut nn = Network::new(&[data_loader::INPUT_PIXELS, 36, 10], 3.0);
    let mut rng = rand::rng();
    let random_digit_for_visualisation = rng.random_range(0..10);

    test_nn_and_print_results(&nn, 0, &data.test_data_set, 0.0, 99);

    for i in 0..network::MAX_EPOCH {
        let epoch_start = Instant::now();
        // Shuffle each epoch so batches are not ordered by digit class.
        data.training_data_set.shuffle(&mut rng);
        nn.run_training_epoch(&data.training_data_set);

        let epoch_secs = epoch_start.elapsed().as_secs_f64();

        test_nn_and_print_results(
            &nn,
            i + 1,
            &data.test_data_set,
            epoch_secs,
            random_digit_for_visualisation as usize,
        );
    }

    // Runs the test set, counts correct predictions, and optionally prints weight visualisations for a chosen digit.
    fn test_nn_and_print_results(
        nn: &Network,
        epoch: usize,
        test_data_set: &Vec<Sample>,
        epoch_secs: f64,
        digit_to_display: usize,
    ) {
        let mut properly_predicted_count = 0;
        let visualise_times: usize = 2;
        let mut print_predicted_count = 0;
        let mut print_wrong_count = 0;
        for sample in test_data_set {
            let print =
                epoch == network::MAX_EPOCH && sample.get_label_as_digit() == digit_to_display;
            let (predicted_digit, layers_activations) = nn.predict(&sample);
            if predicted_digit == sample.get_label_as_digit() {
                if print == true && print_predicted_count < visualise_times {
                    println!("Properly predicted digit recognised as: {}", predicted_digit);
                    nn.display_active_weights(&layers_activations, sample);
                    print_predicted_count += 1;
                }
                properly_predicted_count += 1;
            } else {
                if print == true && print_wrong_count < visualise_times {
                    println!("Incorrectly predicted digit recognised as: {}", predicted_digit);
                    nn.display_active_weights(&layers_activations, sample);
                    print_wrong_count += 1;
                }
            }
        }

        println!(
            "  Epoch {}/{}: {}/{} correct ({:.2}s)",
            epoch,
            network::MAX_EPOCH,
            properly_predicted_count,
            &test_data_set.len(),
            epoch_secs
        );
    }
}

