use burn_import::{burn::graph::RecordType, onnx::ModelGen};

const INPUT_ONNX_FILES: [&str; 5] = [
    "src/model/candy-8.onnx",
    "src/model/mosaic-9.onnx",
    "src/model/pointilism-9.onnx",
    "src/model/rain-princess-9.onnx",
    "src/model/udnie-9.onnx",
];
const OUT_DIR: &str = "model/";

fn main() {
    // Re-run the build script if model files change.
    println!("cargo:rerun-if-changed=src/model");

    // Check if half_precision is enabled.
    let half_precision = cfg!(feature = "half_precision");

    for onnx_file in INPUT_ONNX_FILES {
        // Generate the model code from the ONNX file.
        ModelGen::new()
            .input(onnx_file)
            .out_dir(OUT_DIR)
            .record_type(RecordType::Bincode)
            .embed_states(true)
            .half_precision(half_precision)
            .run_from_script();
    }
}
