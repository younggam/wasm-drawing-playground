import * as wasm from "wasm-drawing-playground";

let styleTransfer;
wasm.StyleTransfer.new().then(done => styleTransfer = done);

self.onmessage = event => {
    const {type, styleName, pixels, width, height, preserve} = event.data;
    if (type === 'transfer' && styleTransfer != null) {
        console.log('style transfer start');
        styleTransfer.inference(nameToModel(styleName), pixels, width, height, preserve).then(data => postMessage({
            type: 'done',
            pixels: data.buffer,
            width: width,
            height: height,
        }, [data.buffer]))
    }
};

function nameToModel(styleName) {
    if (styleName === 'bayanihan') return 0;
    if (styleName === 'lazy') return 1;
    if (styleName === 'mosaic') return 2;
    if (styleName === 'starry') return 3;
    if (styleName === 'tokyo_ghoul') return 4;
    if (styleName === 'udnie') return 5;
    if (styleName === 'wave') return 6;
    return 0;
}