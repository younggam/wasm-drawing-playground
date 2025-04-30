import * as wasm from "wasm-drawing-playground";

let styleTransfer = new wasm.StyleTransfer();
console.log('style transfer ready');

self.onmessage = event => {
    const {type, styleName, pixels, width, height} = event.data;
    if (type === 'transfer' && styleTransfer != null) {
        console.log('style transfer start');
        styleTransfer.inference(nameToModel(styleName), pixels, width, height).then(data => postMessage({
            type: 'done',
            pixels: data.buffer,
            width: width,
            height: height,
        }, [data.buffer]))
    }
};

function nameToModel(styleName) {
    if (styleName === 'candy') return 0;
    if (styleName === 'mosaic') return 1;
    if (styleName === 'rain_princess') return 2;
    if (styleName === 'udnie') return 3;
    return 0;
}