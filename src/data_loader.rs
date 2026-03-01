//! Loads MNIST from disk and exposes training/test sets as `Sample` collections.

use mnist::*;
use crate::sample::Sample;

pub const TRAINING_SET_PATH: &str = "data";
pub const INPUT_PIXELS: usize = 784;
pub const ONE_HOT_OUTPUT_VECTOR_SIZE: usize = 10;

/// Training and test datasets built from MNIST (normalized, one-hot labels).
pub struct MnistData {
    pub training_data_set: Vec<Sample>,
    pub test_data_set: Vec<Sample>,
}

impl MnistData {
    /// Loads MNIST from `TRAINING_SET_PATH`, normalizes pixels, and converts labels to one-hot.
    /// E.g. digit 5 → [0,0,0,0,0,1,0,0,0,0].
    pub fn new() -> Self {
        let NormalizedMnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = MnistBuilder::new()
            .base_path(TRAINING_SET_PATH)
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
