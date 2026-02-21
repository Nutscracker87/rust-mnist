use mnist::*;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, azip, s};

pub const TRAINING_SET_PATH: &str = "data";

pub struct Sample {
    image: Array1<f32>,
    label: Array1<f32>,
}

impl Sample {
    pub fn get_image<'a>(&'a self) -> ArrayView1<'a, f32> {
        self.image.view()
    }

    pub fn get_label<'a>(&'a self) -> ArrayView1<'a, f32> {
        self.label.view()
    }

    // mnist visualisation. A Simple Implementation (The "Block" Method)
    //pub fn display_high_res(pixels: ArrayView1<f32>) {
    pub fn display_digit_pixels(&self) {
        //let image: Array2<f32> = Array2::from_shape_vec((28, 28), pixels.to_vec()).unwrap();
        let image_to_process: Array2<f32> = Array2::from_shape_vec((28, 28), self.image.to_vec()).unwrap();
        // We iterate 2 rows at a time because 1 char = 2 vertical pixels
        for y in (0..28).step_by(2) {
            for x in 0..28 {
                let top = image_to_process[[y, x]];
                let bottom = if y + 1 < 28 { image_to_process[[y + 1, x]] } else { 0.0 };

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
}

pub struct MnistData {
    pub training_data_set: Vec<Sample>,
    pub test_data_set: Vec<Sample>,
}

impl MnistData {
    pub fn new() -> Self {
        let NormalizedMnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = MnistBuilder::new()
            .base_path(TRAINING_SET_PATH)
            // convert labels to vectors. For example digit 5 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            .label_format_one_hot()
            .training_set_length(50_000)
            .validation_set_length(10_000)
            .test_set_length(10_000)
            .finalize()
            .normalize();

        let training_data_set: Vec<Sample> = trn_img
            .chunks(784)
            .zip(trn_lbl.chunks(10))
            .map(|(img_pixels, label)| {
                let labels_f32: Vec<f32> = label.iter().map(|&x| x as f32).collect();
                Sample {
                    image: Array1::from(img_pixels.to_vec()).to_owned(),
                    label: Array1::from_vec(labels_f32).to_owned(),
                }
            })
            .collect();

        let test_data_set: Vec<Sample> = tst_img
            .chunks(784)
            .zip(tst_lbl.chunks(10))
            .map(|(img_pixels, label)| {
                let labels_f32: Vec<f32> = label.iter().map(|&x| x as f32).collect();
                Sample {
                    image: Array1::from(img_pixels.to_vec()).to_owned(),
                    label: Array1::from_vec(labels_f32).to_owned(),
                }
            })
            .collect();

        Self {
            training_data_set,
            test_data_set,
        }
    }
}
