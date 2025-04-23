use wasm_bindgen::__rt::std;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());
    log::info!("Wasm init");
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, wasm-drawing-playground!");
}

#[wasm_bindgen]
pub fn image_something(data: &mut [u8]) {
    log::info!("image_something");
}
