use crate::network::Network;

mod data_loader;
pub mod network;
pub mod sample;

fn main() {
    println!("Hello, my new neural network!");
    let data = data_loader::MnistData::new();
    let first_digit = data
        .training_data_set
        .first()
        .expect("Cant find first digit");
    let nn = Network::new(&[data_loader::INPUT_PIXELS, 32, 10], 3.0);

    println!(
        "The first example label is: {}",
        first_digit.get_label_as_digit()
    );
    println!("The first example image is");
    first_digit.dispaly_digit();

    println!("NN initialized");
    for layer in nn.layers {
        println!("Layer shape: {:?}", layer.weights.shape());
        println!("Layer values: {:?}", layer.weights);
    }
}
