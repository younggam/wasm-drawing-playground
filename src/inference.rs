use crate::state::{ModelType, Models};

use alloc::vec;
use burn::backend::{
    NdArray, Wgpu,
    wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup_async},
};

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::js_sys::Uint8ClampedArray;
use web_time::Instant;

pub enum ModelWithBackend {
    WithNdArrayBackend(Models<NdArray>),
    WithWgpuBackend(Models<Wgpu>),
}

#[wasm_bindgen]
pub struct StyleTransfer {
    model: ModelWithBackend,
}

#[wasm_bindgen]
impl StyleTransfer {
    /// Sets the backend to wgpu
    pub async fn new() -> Self {
        log::info!("Loading the model to the Wgpu backend");
        let start = Instant::now();

        let device = WgpuDevice::default();
        init_setup_async::<AutoGraphicsApi>(&device, Default::default()).await;
        let ret = Self {
            model: ModelWithBackend::WithWgpuBackend(Models::new(&device)),
        };

        log::debug!(
            "Model is loaded to the Wgpu backend in {:?}",
            start.elapsed()
        );
        ret
    }

    /// Runs inference on the image
    pub async fn inference(
        &self,
        model_type: usize,
        data: Uint8ClampedArray,
        width: usize,
        height: usize,
    ) -> Result<Uint8ClampedArray, JsValue> {
        log::info!("Running inference on the image");
        let start = Instant::now();

        let size = width * height;
        let mut vec = data.to_vec();
        let (mut i_r, mut i_g, mut i_b) = (0usize, size, 2 * size);
        let mut input = vec![0f32; 3 * size];
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) * 4;
                input[i_r] = vec[index] as f32;
                input[i_g] = vec[index + 1] as f32;
                input[i_b] = vec[index + 2] as f32;
                i_r += 1;
                i_g += 1;
                i_b += 1;
            }
        }

        let model_type = ModelType::from(model_type);
        let result = match self.model {
            ModelWithBackend::WithNdArrayBackend(ref model) => {
                model.forward(model_type, &input, width, height).await
            }
            ModelWithBackend::WithWgpuBackend(ref model) => {
                model.forward(model_type, &input, width, height).await
            }
        };

        log::debug!("Inference is completed in {:?}", start.elapsed());

        fn rgb_float_to_u8(val: f32) -> u8 {
            val.clamp(0.0, 255.0) as u8
        }

        let (mut i_r, mut i_g, mut i_b) = (0usize, size, 2 * size);
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) * 4;
                vec[index] = rgb_float_to_u8(result[i_r]);
                vec[index + 1] = rgb_float_to_u8(result[i_g]);
                vec[index + 2] = rgb_float_to_u8(result[i_b]);
                i_r += 1;
                i_g += 1;
                i_b += 1;
            }
        }
        data.copy_from(&vec);
        Ok(data)
    }
}
