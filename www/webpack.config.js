const CopyWebpackPlugin = require("copy-webpack-plugin");
const path = require('path');

module.exports = {
    entry: "./bootstrap.js",
    experiments: {
        asyncWebAssembly: true,
    },
    output: {
        path: path.resolve(__dirname, "dist"),
        filename: "bootstrap.js",
    },
    devtool: "source-map",
    mode: "production",
    plugins: [
        new CopyWebpackPlugin(
            [
                "index.html",
                "icons/*.svg",
                "background.png",
                "manifest.json",
                "background.js",
                "style.css",
            ]
        )
    ],
};
