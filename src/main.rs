pub mod network;
mod data_loader;

fn main() {
    println!("Hello, my new neural network!");
    let data = data_loader::MnistData::new();
}
