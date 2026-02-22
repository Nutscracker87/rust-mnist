use mnist::*;
use crate::sample::Sample;

pub const TRAINING_SET_PATH: &str = "data";
pub const INPUT_PIXELS: usize = 784;
pub const ONE_HOT_OUTPUT_VECTOR_SIZE: usize = 10;

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
            .chunks(INPUT_PIXELS)
            .zip(trn_lbl.chunks(ONE_HOT_OUTPUT_VECTOR_SIZE))
            .map(|(digit_pixels, digit_label)| {
                Sample::new(digit_pixels, digit_label)
            })
            .collect();

        let test_data_set: Vec<Sample> = tst_img
            .chunks(INPUT_PIXELS)
            .zip(tst_lbl.chunks(ONE_HOT_OUTPUT_VECTOR_SIZE))
            .map(|(digit_pixels, digit_label)| {
                Sample::new(digit_pixels, digit_label)
            })
            .collect();

        Self {
            training_data_set,
            test_data_set,
        }
    }
}
