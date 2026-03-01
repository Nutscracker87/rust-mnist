//! A single MNIST example: flattened image pixels and one-hot label.

use ndarray::{Array1, Array2, ArrayView1};

/// One training or test example: 784 normalized pixels and a 10-dim one-hot label.
pub struct Sample {
    image: Array1<f32>,
    label: Array1<f32>,
}

impl Sample {
    /// Build from raw pixel slice and one-hot label (e.g. from MNIST chunks).
    pub fn new(image: &[f32], label: &[u8]) -> Self {
        let labels_f32: Vec<f32> = label.iter().map(|&x| x as f32).collect();
        Sample {
            image: Array1::from(image.to_vec()).to_owned(),
            label: Array1::from_vec(labels_f32).to_owned(),
        }
    }

    pub fn get_image<'a>(&'a self) -> ArrayView1<'a, f32> {
        self.image.view()
    }

    pub fn get_label<'a>(&'a self) -> ArrayView1<'a, f32> {
        self.label.view()
    }

    /// Prints the 28×28 image in the terminal using block chars (▀) and ANSI grayscale (232–255).
    pub fn display_digit(&self) {
        let image_to_process: Array2<f32> =
            Array2::from_shape_vec((28, 28), self.image.to_vec()).unwrap();
        for y in (0..28).step_by(2) {
            for x in 0..28 {
                let top = image_to_process[[y, x]];
                let bottom = if y + 1 < 28 {
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

    /// Returns the digit class (0–9) as the index where the one-hot label is 1.
    pub fn get_label_as_digit(&self) -> usize {
        self.label
            .iter()
            .position(|&val| val > 0.5)
            .unwrap_or(0)
    }
}
