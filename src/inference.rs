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
        preserve_color: bool,
    ) -> Result<Uint8ClampedArray, JsValue> {
        log::info!("Running inference on the image");
        let start = Instant::now();

        let size = width * height;
        let input_image = data.to_vec();
        let (mut i_b, mut i_g, mut i_r) = (0usize, size, 2 * size);
        let mut input = vec![0f32; 3 * size];
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) * 4;
                input[i_r] = input_image[index] as f32;
                input[i_g] = input_image[index + 1] as f32;
                input[i_b] = input_image[index + 2] as f32;
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

        let (mut i_b, mut i_g, mut i_r) = (0usize, size, 2 * size);
        let mut output_image = vec![0u8; 4 * size];
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) * 4;
                output_image[index] = rgb_float_to_u8(result[i_r]);
                output_image[index + 1] = rgb_float_to_u8(result[i_g]);
                output_image[index + 2] = rgb_float_to_u8(result[i_b]);
                output_image[index + 3] = input_image[index + 3];
                i_r += 1;
                i_g += 1;
                i_b += 1;
            }
        }
        if preserve_color {
            for y in 0..height {
                for x in 0..width {
                    let index = (y * width + x) * 4;
                    let dest_gray = 0.299 * output_image[index] as f32
                        + 0.587 * output_image[index + 1] as f32
                        + 0.114 * output_image[index + 2] as f32;

                    let (_, cr, cb) = rgb_to_ycrcb(
                        input_image[index],
                        input_image[index + 1],
                        input_image[index + 2],
                    );
                    let (r, g, b) = ycrcb_to_rgb(dest_gray, cr, cb);

                    output_image[index] = r.clamp(0.0,255.0).round() as u8;
                    output_image[index + 1] = g.clamp(0.0,255.0).round() as u8;
                    output_image[index + 2] = b.clamp(0.0,255.0).round() as u8;
                }
            }
        }
        data.copy_from(&output_image);
        Ok(data)
    }
}

fn rgb_to_ycrcb(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = r as f32;
    let g = g as f32;
    let b = b as f32;

    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cr = (r - y) * 0.713 + 128.0;
    let cb = (b - y) * 0.564 + 128.0;

    (y, cr, cb)
}

fn ycrcb_to_rgb(y: f32, cr: f32, cb: f32) -> (f32, f32, f32) {
    let r = y + 1.403 * (cr - 128.0);
    let g = y - 0.714 * (cr - 128.0) - 0.344 * (cb - 128.0);
    let b = y + 1.773 * (cb - 128.0);
    (r, g, b)
}
