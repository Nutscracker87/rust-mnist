pub mod network;
pub mod sample;
mod data_loader;

fn main() {
    println!("Hello, my new neural network!");
    let data = data_loader::MnistData::new();
    let first_digit = data.training_data_set.first().expect("Cant find first digit");
    println!("The first example label is: {}",  first_digit.get_label_as_digit());
    println!("The first example image is");
    first_digit.dispaly_digit();
}
