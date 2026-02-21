use ndarray::Array2;


pub struct Layer {
    pub weights: Vec<Array2<f32>>,
}

pub struct Network {
    pub layers: Vec<Layer>,
    pub learning_rate: f32
}