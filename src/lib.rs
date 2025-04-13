mod utils;

use wasm_bindgen::prelude::*;
use web_sys::console;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn init() {
    utils::set_panic_hook();
    console::log_1(&"Wasm init".into());
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, wasm-drawing-playground!");
}

#[wasm_bindgen]
pub fn image_something(data: &mut [u8]) {
    console::log_1(&"image_something".into());
}
