import * as wasm from "wasm-drawing-playground";

let styleTransfer = new wasm.StyleTransfer();
console.log('style transfer ready');

self.onmessage = event => {
    const {type, modelType, pixels, width, height} = event.data;
    if (type === 'transfer' && styleTransfer != null) {
        console.log('style transfer start');
        styleTransfer.inference(modelType, pixels, width, height).then(data => postMessage({
            type: 'done',
            pixels: data.buffer,
            width: width,
            height: height,
        }, [data.buffer]))
    }
};