use alloc::vec::Vec;
use burn::prelude::Backend;
use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};
use model::TransformerNet;

const CANDY_ENCODED: &[u8] = include_bytes!("../pytorch/burn/candy.bin");
const MOSAIC_ENCODED: &[u8] = include_bytes!("../pytorch/burn/mosaic.bin");
const RAIN_PRINCESS_ENCODED: &[u8] = include_bytes!("../pytorch/burn/rain_princess.bin");
const UDNIE_ENCODED: &[u8] = include_bytes!("../pytorch/burn/udnie.bin");

#[repr(usize)]
pub enum ModelType {
    Candy = 0,
    Mosaic = 1,
    RainPrincess = 2,
    Udnie = 3,
}

impl From<usize> for ModelType {
    fn from(value: usize) -> Self {
        match value {
            0 => Self::Candy,
            1 => Self::Mosaic,
            2 => Self::RainPrincess,
            3 => Self::Udnie,
            _ => Self::Candy,
        }
    }
}

pub struct Models<B: Backend> {
    pub candy: TransformerNet<B>,
    pub mosaic: TransformerNet<B>,
    pub rain_princess: TransformerNet<B>,
    pub udnie: TransformerNet<B>,
}

impl<B: Backend> Models<B> {
    pub fn new(device: &B::Device) -> Self {
        let recorder = BinBytesRecorder::<FullPrecisionSettings, &'static [u8]>::default();
        Self {
            candy: TransformerNet::init(device).load_record(
                recorder
                    .load(CANDY_ENCODED, device)
                    .expect("Failed to decode candy"),
            ),
            mosaic: TransformerNet::init(device).load_record(
                recorder
                    .load(MOSAIC_ENCODED, device)
                    .expect("Failed to decode mosaic"),
            ),
            rain_princess: TransformerNet::init(device).load_record(
                recorder
                    .load(RAIN_PRINCESS_ENCODED, device)
                    .expect("Failed to decode rain princess"),
            ),
            udnie: TransformerNet::init(device).load_record(
                recorder
                    .load(UDNIE_ENCODED, device)
                    .expect("Failed to decode udnie"),
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
            ModelType::Candy => self.candy.forward(input, width, height).await,
            ModelType::Mosaic => self.mosaic.forward(input, width, height).await,
            ModelType::RainPrincess => self.rain_princess.forward(input, width, height).await,
            ModelType::Udnie => self.udnie.forward(input, width, height).await,
        }
    }
}
