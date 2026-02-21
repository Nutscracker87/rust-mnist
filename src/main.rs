pub mod network;
mod data_loader;

fn main() {
    println!("Hello, my new neural network!");
    let data = data_loader::MnistData::new();
    let first_digit = data.training_data_set.first().expect("Cant find first digit");
    first_digit.display_digit_pixels();
}
