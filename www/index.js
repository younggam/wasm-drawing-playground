const worker = new Worker(new URL('./worker.js', import.meta.url), {
    type: "module"
});

// global variables with default value
const canvas = document.querySelector("canvas"),
    toolBtns = document.querySelectorAll(".tool"),
    fillColor = document.querySelector("#fill-color"),
    preserveColor = document.querySelector("#preserve-color"),
    sizeSlider = document.querySelector("#size-slider"),
    colorBtns = document.querySelectorAll(".colors .option"),
    styleBtns = document.querySelectorAll(".style"),
    colorPicker = document.querySelector("#color-picker"),
    clearCanvas = document.querySelector(".clear-canvas"),
    saveImg = document.querySelector(".save-img"),
    convertImg = document.querySelector(".convert-img"),
    uploadImg = document.querySelector(".upload-img"),
    resizeCanvas = document.querySelector(".resize-canvas"),
    drawingBoard = document.querySelector(".drawing-board"),
    widthInput = document.querySelector("#widthInput"),
    heightInput = document.querySelector("#heightInput"),
    ctx = canvas.getContext("2d", {willReadFrequently: true});
let prevMouseX, prevMouseY, snapshot,
    isDrawing = false,
    isIn = false,
    selectedTool = "brush",
    brushWidth = 5,
    selectedColor = "#000",
    selectedStyle = "candy",
    recentColors = new Array(8).fill("#000");

// initialization

const setCanvasBackground = () => {
    // setting whole canvas background to white, so the downloaded img background will be white
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = selectedColor; // setting fillstyle back to the selectedColor, it'll be the brush color
}

canvas.width = canvas.offsetWidth;
canvas.height = canvas.offsetHeight;
setCanvasBackground();

worker.onmessage = event => {
    const {type, pixels, width, height} = event.data;
    if (type === 'done') {
        console.log('style transfer done');
        const imageData = new ImageData(new Uint8ClampedArray(pixels), width, height);
        ctx.putImageData(imageData, 0, 0);
    }
};

// draw func

const drawRect = (e) => {
    // if fillColor isn't checked draw a rect with border else draw rect with background
    if (!fillColor.checked) {
        // creating circle according to the mouse pointer
        return ctx.strokeRect(e.offsetX, e.offsetY, prevMouseX - e.offsetX, prevMouseY - e.offsetY);
    }
    ctx.fillRect(e.offsetX, e.offsetY, prevMouseX - e.offsetX, prevMouseY - e.offsetY);
}
const drawCircle = (e) => {
    ctx.beginPath(); // creating new path to draw circle
    // getting radius for circle according to the mouse pointer
    let radius = Math.sqrt(Math.pow((prevMouseX - e.offsetX), 2) + Math.pow((prevMouseY - e.offsetY), 2));
    ctx.arc(prevMouseX, prevMouseY, radius, 0, 2 * Math.PI); // creating circle according to the mouse pointer
    fillColor.checked ? ctx.fill() : ctx.stroke(); // if fillColor is checked fill circle else draw border circle
}
const drawTriangle = (e) => {
    ctx.beginPath(); // creating new path to draw circle
    ctx.moveTo(prevMouseX, prevMouseY); // moving triangle to the mouse pointer
    ctx.lineTo(e.offsetX, e.offsetY); // creating first line according to the mouse pointer
    ctx.lineTo(prevMouseX * 2 - e.offsetX, e.offsetY); // creating bottom line of triangle
    ctx.closePath(); // closing path of a triangle so the third line draw automatically
    fillColor.checked ? ctx.fill() : ctx.stroke(); // if fillColor is checked fill triangle else draw border
}
const startDraw = (e) => {
    if (selectedTool === "fill") {
        floodFill(e.offsetX, e.offsetY);
    } else {
        isDrawing = true;
        prevMouseX = e.offsetX; // passing current mouseX position as prevMouseX value
        prevMouseY = e.offsetY; // passing current mouseY position as prevMouseY value
        ctx.beginPath(); // creating new path to draw
        ctx.lineWidth = brushWidth; // passing brushSize as line width
        ctx.strokeStyle = selectedColor; // passing selectedColor as stroke style
        ctx.fillStyle = selectedColor; // passing selectedColor as fill style
        // copying canvas data & passing as snapshot value.. this avoids dragging the image
        snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);
    }
}
const drawing = (e) => {
    if (!isDrawing || !isIn) return; // if isDrawing is false return from here
    ctx.putImageData(snapshot, 0, 0); // adding copied canvas data on to this canvas
    if (selectedTool === "brush" || selectedTool === "eraser") {
        // if selected tool is eraser then set strokeStyle to white
        // to paint white color on to the existing canvas content else set the stroke color to selected color
        ctx.strokeStyle = selectedTool === "eraser" ? "#fff" : selectedColor;
        ctx.lineTo(e.offsetX, e.offsetY); // creating line according to the mouse pointer
        ctx.stroke(); // drawing/filling line with color
    } else if (selectedTool === "rectangle") {
        drawRect(e);
    } else if (selectedTool === "circle") {
        drawCircle(e);
    } else if (selectedTool === "triangle") {
        drawTriangle(e);
    }
}

function resize(width, height) {
    drawingBoard.style.width = Math.max(width, 512) + "px";
    drawingBoard.style.height = Math.max(height, 512) + "px";
    canvas.width = width;
    canvas.height = height;
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    widthInput.value = width;
    heightInput.value = height;
}

function colorMatch(data, pos, target) {
    return (
        data[pos] === target[0] &&
        data[pos + 1] === target[1] &&
        data[pos + 2] === target[2]
    );
}

function floodFill(startX, startY) {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const rgb = selectedColor.replace(/[^\d,]/g, '').split(',');
    const color = [parseInt(rgb[0]), parseInt(rgb[1]), parseInt(rgb[2])];

    const startPos = (startY * width + startX) * 4;
    const startColor = data.slice(startPos, startPos + 4);

    if (colorMatch(color, 0, startColor)) return; // 이미 같은 색이면 무시

    const queue = [[startX, startY]];

    while (queue.length > 0) {
        const [x, y] = queue.pop();
        const pos = (y * width + x) * 4;

        if (x < 0 || x >= width || y < 0 || y >= height) continue;
        if (!colorMatch(data, pos, startColor)) continue;

        // 색 변경
        data[pos] = color[0];
        data[pos + 1] = color[1];
        data[pos + 2] = color[2];

        queue.push([x + 1, y]);
        queue.push([x - 1, y]);
        queue.push([x, y + 1]);
        queue.push([x, y - 1]);
    }

    ctx.putImageData(imageData, 0, 0);
}

// tool-board

toolBtns.forEach(btn => {
    btn.addEventListener("click", () => { // adding click event to all tool option
        // removing active class from the previous option and adding on current clicked option
        document.querySelector(".tools .active").classList.remove("active");
        btn.classList.add("active");
        selectedTool = btn.id;
    });
});
sizeSlider.addEventListener("change", () => brushWidth = sizeSlider.value); // passing slider value as brushSize
colorBtns.forEach(btn => {
    btn.addEventListener("click", () => { // adding click event to all color button
        // removing selected class from the previous option and adding on current clicked option
        document.querySelector(".colors .selected").classList.remove("selected");
        btn.classList.add("selected");
        // passing selected btn background color as selectedColor value
        selectedColor = window.getComputedStyle(btn).getPropertyValue("background-color");
    });
});
colorPicker.addEventListener("change", () => {
    // passing picked color value from color picker to last color btn background
    colorPicker.parentElement.style.background = colorPicker.value;
    recentColors.pop()
    recentColors.unshift(colorPicker.value);
    for (let i = 0; i < 8; i++) {
        colorBtns[i + 16].style.backgroundColor = recentColors[i];
    }
    colorBtns[16].click();
});
resizeCanvas.addEventListener("click", () => {
    const width = parseInt(document.getElementById("widthInput").value);
    const height = parseInt(document.getElementById("heightInput").value);
    let tempCanvas = document.createElement("canvas");
    tempCanvas.width = width;
    tempCanvas.height = height;
    let tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(canvas, 0, 0, width, height);
    resize(width, height);
    ctx.drawImage(tempCanvas, 0, 0);
    tempCtx = null;
    tempCanvas = null;
})
clearCanvas.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // clearing whole canvas
    uploadImg.value = '';
    setCanvasBackground();
});

// canvas

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", drawing);
canvas.addEventListener("mouseenter", e => {
    if (isDrawing) startDraw(e);
})
canvas.addEventListener("mouseover", () => isIn = true);
canvas.addEventListener("mouseout", () => isIn = false);
document.addEventListener("mouseup", () => isDrawing = false);

// style-board

styleBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelector(".styles .active").classList.remove("active");
        btn.classList.add("active");
        selectedStyle = btn.id;
    });
})

uploadImg.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => {
        resize(img.width, img.height);
        ctx.drawImage(img, 0, 0)
    };
    img.src = URL.createObjectURL(file);
});

convertImg.addEventListener("click", async () => {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    worker.postMessage({
        type: 'transfer',
        styleName: selectedStyle,
        pixels: imageData.data,
        width: canvas.width,
        height: canvas.height,
        preserve: preserveColor.checked,
    });
});

saveImg.addEventListener("click", () => {
    const link = document.createElement("a"); // creating <a> element
    link.download = `${Date.now()}.jpg`; // passing current date as link download value
    link.href = canvas.toDataURL(); // passing canvasData as link href value
    link.click(); // clicking link to download image
});