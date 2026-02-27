use rand::seq::SliceRandom;

use crate::network::Network;

mod data_loader;
pub mod network;
pub mod sample;

fn main() {
    println!("Hello, my new neural network!");
    let mut data = data_loader::MnistData::new();
    let nn = Network::new(&[data_loader::INPUT_PIXELS, 32, 10], 3.0);
    let mut rng = rand::rng();

    for i in 0..network::MAX_EPOCH {
        // need to shuffle data for better learning, by default mnist is ordered sequentially by digit class.
        // we need to shuffle data for each epoch to get better results
        data.training_data_set.shuffle(&mut rng);
        nn.run_training_epoch(&data.training_data_set);
    }

    // let first_digit = data
    //     .training_data_set
    //     .first()
    //     .expect("Cant find first digit");

    // println!(
    //     "The first example label is: {}",
    //     first_digit.get_label_as_digit()
    // );
    // println!("The first example image is");
    // first_digit.dispaly_digit();

    // println!("NN initialized");
    // for layer in nn.layers {
    //     println!("Layer shape: {:?}", layer.weights.shape());
    //     println!("Layer values: {:?}", layer.weights);
    // }
}
