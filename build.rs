/// This build script does the following:
/// 1. Loads PyTorch weights into a model record.
/// 2. Saves the model record to a file using the `NamedMpkFileRecorder`.
use burn::{
    backend::NdArray,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

const MODEL_NAMES: &[&str] = &["candy", "mosaic", "rain_princess", "udnie"];

// Basic backend type (not used directly here).
type B = NdArray<f32>;

fn main() {
    let device = Default::default();
    // Save the model record to a file.
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();

    for model_name in MODEL_NAMES {
        // Load PyTorch weights into a model record.
        let record: model::TransformerNetRecord<B> =
            PyTorchFileRecorder::<FullPrecisionSettings>::default()
                .load(
                    LoadArgs::new(format!("pytorch/{model_name}.pt").into()),
                    &device,
                )
                .expect(&format!("Failed to decode {model_name}"));

        recorder
            .record(record, format!("pytorch/burn/{model_name}").into())
            .expect(&format!("Failed to save {model_name} record"));
    }
}
