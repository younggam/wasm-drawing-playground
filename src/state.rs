use alloc::vec::Vec;
use burn::prelude::Backend;
use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};
use model::TransformerNetwork;

const BAYANIHAN_ENCODED: &[u8] = include_bytes!("../pytorch/burn/bayanihan.bin");
const LAZY_ENCODED: &[u8] = include_bytes!("../pytorch/burn/lazy.bin");
const MOSAIC_ENCODED: &[u8] = include_bytes!("../pytorch/burn/mosaic.bin");
const STARRY_ENCODED: &[u8] = include_bytes!("../pytorch/burn/starry.bin");
const TOKYO_GHOUL_ENCODED: &[u8] = include_bytes!("../pytorch/burn/tokyo_ghoul.bin");
const UDNIE_ENCODED: &[u8] = include_bytes!("../pytorch/burn/udnie.bin");
const WAVE_ENCODED: &[u8] = include_bytes!("../pytorch/burn/wave.bin");

#[repr(usize)]
pub enum ModelType {
    Bayanihan = 0,
    Lazy = 1,
    Mosaic = 2,
    Starry = 3,
    TokyoGhoul = 4,
    Udnie = 5,
    Wave = 6,
}

impl From<usize> for ModelType {
    fn from(value: usize) -> Self {
        match value {
            0 => Self::Bayanihan,
            1 => Self::Lazy,
            2 => Self::Mosaic,
            3 => Self::Starry,
            4 => Self::TokyoGhoul,
            5 => Self::Udnie,
            6 => Self::Wave,
            _ => Self::Bayanihan,
        }
    }
}

pub struct Models<B: Backend> {
    pub bayanihan: TransformerNetwork<B>,
    pub lazy: TransformerNetwork<B>,
    pub mosaic: TransformerNetwork<B>,
    pub starry: TransformerNetwork<B>,
    pub tokyo_ghoul: TransformerNetwork<B>,
    pub udnie: TransformerNetwork<B>,
    pub wave: TransformerNetwork<B>,
}

impl<B: Backend> Models<B> {
    pub fn new(device: &B::Device) -> Self {
        let recorder = BinBytesRecorder::<FullPrecisionSettings, &'static [u8]>::default();
        Self {
            bayanihan: TransformerNetwork::init(device).load_record(
                recorder
                    .load(BAYANIHAN_ENCODED, device)
                    .expect("Failed to decode bayanihan"),
            ),
            lazy: TransformerNetwork::init(device).load_record(
                recorder
                    .load(LAZY_ENCODED, device)
                    .expect("Failed to decode lazy"),
            ),
            mosaic: TransformerNetwork::init(device).load_record(
                recorder
                    .load(MOSAIC_ENCODED, device)
                    .expect("Failed to decode mosaic"),
            ),
            starry: TransformerNetwork::init(device).load_record(
                recorder
                    .load(STARRY_ENCODED, device)
                    .expect("Failed to decode starry"),
            ),
            tokyo_ghoul: TransformerNetwork::init(device).load_record(
                recorder
                    .load(TOKYO_GHOUL_ENCODED, device)
                    .expect("Failed to decode tokyo ghoul"),
            ),
            udnie: TransformerNetwork::init(device).load_record(
                recorder
                    .load(UDNIE_ENCODED, device)
                    .expect("Failed to decode udnie"),
            ),
            wave: TransformerNetwork::init(device).load_record(
                recorder
                    .load(WAVE_ENCODED, device)
                    .expect("Failed to decode wave"),
            ),
        }
    }

    pub async fn forward(
        &self,
        model_type: ModelType,
        input: &[f32],
        width: usize,
        height: usize,
    ) -> Vec<f32> {
        match model_type {
            ModelType::Bayanihan => self.bayanihan.forward(input, width, height).await,
            ModelType::Lazy => self.lazy.forward(input, width, height).await,
            ModelType::Mosaic => self.mosaic.forward(input, width, height).await,
            ModelType::Starry => self.starry.forward(input, width, height).await,
            ModelType::TokyoGhoul => self.tokyo_ghoul.forward(input, width, height).await,
            ModelType::Udnie => self.udnie.forward(input, width, height).await,
            ModelType::Wave => self.wave.forward(input, width, height).await,
        }
    }
}
