use mnist::*;
use ndarray::{Array1, Array2, Array3, s};

pub const TRAINING_SET_PATH: &str = "data";

pub struct Sample {
    image: Array1<f32>,
    label: Array1<f32>
}

pub struct MnistData {
    pub training_data_set: Vec<Sample>,
    pub test_data_set: Vec<Sample>
}

impl MnistData {
    pub fn new() {
        let Mnist {
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
            .finalize();

        let image_num = 0;
        // Can use an Array2 or Array3 here (Array3 for visualization)
        let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
            .expect("Error converting images to Array3 struct")
            .map(|x| *x as f32 / 256.0); // division by 256.0 is a trick to improve the logic resilience a little be. This way we will no work with 1.0
        println!("{:#.2?}\n", train_data.slice(s![image_num, .., ..]));

        // Convert the returned Mnist struct to Array2 format
        let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 10), trn_lbl)
            .expect("Error converting training labels to Array2 struct")
            .map(|x| *x as f32);
        println!(
            "The first digit is a {:?}",
            train_labels.slice(s![image_num, ..])
        );

        let _test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
            .expect("Error converting images to Array3 struct")
            .map(|x| *x as f32 / 256.);

        let _test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 10), tst_lbl)
            .expect("Error converting testing labels to Array2 struct")
            .map(|x| *x as f32);
    }
}
