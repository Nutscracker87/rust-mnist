use ndarray::{Array1, Array2, ArrayView1};

pub struct Sample {
    image: Array1<f32>,
    label: Array1<f32>,
}

impl Sample {
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

    // mnist visualisation. A Simple Implementation (The "Block" Method)
    pub fn display_digit(&self) {
        //let image: Array2<f32> = Array2::from_shape_vec((28, 28), pixels.to_vec()).unwrap();
        let image_to_process: Array2<f32> =
            Array2::from_shape_vec((28, 28), self.image.to_vec()).unwrap();
        // We iterate 2 rows at a time because 1 char = 2 vertical pixels
        for y in (0..28).step_by(2) {
            for x in 0..28 {
                let top = image_to_process[[y, x]];
                let bottom = if y + 1 < 28 {
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

    pub fn get_label_as_digit(&self) -> usize {
        self.label
            .iter()
            .position(|&val| val > 0.5) // Returns Option<usize>
            .unwrap_or(0)
    }
}
