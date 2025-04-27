use crate::state::{ModelType, Models};

use alloc::{vec, vec::Vec};
use burn::backend::{
    NdArray, Wgpu,
    wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup_async},
};
use burn::prelude::*;

use model::TransformerNet;

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
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        log::info!("Initializing the style transfer");
        let device = Default::default();
        Self {
            model: ModelWithBackend::WithNdArrayBackend(Models::new(&device)),
        }
    }

    /// Runs inference on the image
    pub async fn inference(
        &self,
        model_type: usize,
        data: Uint8ClampedArray,
        width: usize,
        height: usize,
    ) -> Vec<u8> {
        log::info!("Running inference on the image");
        let start = Instant::now();

        let size = width * height;
        let mut data = data.to_vec();
        let (mut i_r, mut i_g, mut i_b) = (0usize, size, 2 * size);
        let mut input = vec![0f32; 3 * size];
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) * 4;
                input[i_r] = data[index] as f32;
                input[i_g] = data[index + 1] as f32;
                input[i_b] = data[index + 2] as f32;
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
                data[index] = rgb_float_to_u8(result[i_r]);
                data[index + 1] = rgb_float_to_u8(result[i_g]);
                data[index + 2] = rgb_float_to_u8(result[i_b]);
                i_r += 1;
                i_g += 1;
                i_b += 1;
            }
        }
        data
    }

    /// Sets the backend to NdArray
    pub async fn set_backend_ndarray(&mut self) {
        log::info!("Loading the model to the NdArray backend");
        let start = Instant::now();

        let device = Default::default();
        self.model = ModelWithBackend::WithNdArrayBackend(Models::new(&device));

        log::debug!(
            "Model is loaded to the NdArray backend in {:?}",
            start.elapsed()
        );
    }

    /// Sets the backend to Wgpu
    pub async fn set_backend_wgpu(&mut self) {
        log::info!("Loading the model to the Wgpu backend");
        let start = Instant::now();

        let device = WgpuDevice::default();
        init_setup_async::<AutoGraphicsApi>(&device, Default::default()).await;
        self.model = ModelWithBackend::WithWgpuBackend(Models::new(&device));

        log::debug!(
            "Model is loaded to the Wgpu backend in {:?}",
            start.elapsed()
        );

        log::debug!("Warming up the model");
        let start = Instant::now();

        let _ = self
            .inference(
                0,
                Uint8ClampedArray::new_with_length(3 * 256 * 256),
                256,
                256,
            )
            .await;

        log::debug!("Warming up is completed in {:?}", start.elapsed());
    }
}
