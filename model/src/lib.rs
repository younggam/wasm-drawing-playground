#![cfg_attr(not(test), no_std)]
extern crate alloc;

use alloc::{vec, vec::Vec};
use burn::prelude::*;
use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        {InstanceNorm, InstanceNormConfig},
    },
    tensor::{activation::relu, module::conv_transpose2d, ops::ConvTransposeOptions},
};

#[derive(Module, Debug)]
pub struct TransformerNetwork<B: Backend> {
    conv_block: Vec<ConvLayer<B>>,
    residual_block: Vec<ResidualLayer<B>>,
    deconv0: DeconvLayer<B>,
    deconv1: DeconvLayer<B>,
    conv: ConvLayer<B>,
}

impl<B: Backend> TransformerNetwork<B> {
    pub fn init(device: &B::Device) -> Self {
        Self {
            conv_block: vec![
                ConvLayer::init(3, 32, 9, 1, true, device),
                ConvLayer::init(32, 64, 3, 2, true, device),
                ConvLayer::init(64, 128, 3, 2, true, device),
            ],
            residual_block: vec![
                ResidualLayer::init(128, 3, device),
                ResidualLayer::init(128, 3, device),
                ResidualLayer::init(128, 3, device),
                ResidualLayer::init(128, 3, device),
                ResidualLayer::init(128, 3, device),
            ],
            deconv0: DeconvLayer::init(128, 64, 3, 2, 1, true, device),
            deconv1: DeconvLayer::init(64, 32, 3, 2, 1, true, device),
            conv: ConvLayer::init(32, 3, 9, 1, false, device),
        }
    }

    pub async fn forward(&self, input: &[f32], width: usize, height: usize) -> Vec<f32> {
        let mut input = Tensor::<B, 1>::from_floats(input, &B::Device::default())
            .reshape([1, 3, height, width]);

        input = relu(self.conv_block[0].forward(input));
        let dims_0 = input.dims();
        input = relu(self.conv_block[1].forward(input));
        let dims_1 = input.dims();
        input = relu(self.conv_block[2].forward(input));

        for res in &self.residual_block {
            input = res.forward(input)
        }

        let input = relu(self.deconv0.forward(input, dims_1));
        let input = relu(self.deconv1.forward(input, dims_0));
        let output = self.conv.forward(input);

        output.into_data_async().await.to_vec().unwrap()
    }
}

#[derive(Module, Debug)]
pub struct ConvLayer<B: Backend> {
    reflection_pad: ReflectionPad2d,
    conv_layer: Conv2d<B>,
    norm_layer: Option<InstanceNorm<B>>,
}

impl<B: Backend> ConvLayer<B> {
    pub fn init(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        instance_norm: bool,
        device: &B::Device,
    ) -> Self {
        let padding_size = kernel_size / 2;
        Self {
            reflection_pad: ReflectionPad2d::init([
                padding_size,
                padding_size,
                padding_size,
                padding_size,
            ]),
            conv_layer: Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
                .with_stride([stride, stride])
                .init(device),
            norm_layer: if instance_norm {
                Some(
                    InstanceNormConfig::new(out_channels)
                        .with_affine(true)
                        .init(device),
                )
            } else {
                None
            },
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let input = self.reflection_pad.forward(input);
        let input = self.conv_layer.forward(input);
        if let Some(norm_layer) = &self.norm_layer {
            norm_layer.forward(input)
        } else {
            input
        }
    }
}

#[derive(Module, Clone, Debug)]
pub struct ReflectionPad2d {
    pub padding: [usize; 4],
}

impl ReflectionPad2d {
    pub fn init(padding: [usize; 4]) -> Self {
        Self { padding }
    }

    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [pl, pr, pt, pb] = self.padding;

        let shape = input.shape();
        let h = shape.dims[2];
        let w = shape.dims[3];

        let left = input.clone().slice(s![.., .., .., 0..pl]).flip([3]);
        let right = input.clone().slice(s![.., .., .., (w - pr)..w]).flip([3]);
        let padded_w = Tensor::cat(Vec::from([left, input, right]), 3);

        let top = padded_w.clone().slice(s![.., .., 0..pt, ..]).flip([2]);
        let bottom = padded_w
            .clone()
            .slice(s![.., .., (h - pb)..h, ..])
            .flip([2]);

        Tensor::cat(Vec::from([top, padded_w, bottom]), 2)
    }
}

#[derive(Module, Debug)]
pub struct ResidualLayer<B: Backend> {
    conv1: ConvLayer<B>,
    conv2: ConvLayer<B>,
}

impl<B: Backend> ResidualLayer<B> {
    pub fn init(channels: usize, kernel_size: usize, device: &B::Device) -> Self {
        Self {
            conv1: ConvLayer::init(channels, channels, kernel_size, 1, true, device),
            conv2: ConvLayer::init(channels, channels, kernel_size, 1, true, device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = input.clone();
        let out = relu(self.conv1.forward(input));
        self.conv2.forward(out) + identity
    }
}

#[derive(Module, Debug)]
pub struct DeconvLayer<B: Backend> {
    conv_transpose: ConvTranspose2d<B>,
    norm_layer: Option<InstanceNorm<B>>,
}

impl<B: Backend> DeconvLayer<B> {
    pub fn init(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        output_padding: usize,
        instance_norm: bool,
        device: &B::Device,
    ) -> Self {
        let padding_size = kernel_size / 2;
        Self {
            conv_transpose: ConvTranspose2dConfig::new(
                [in_channels, out_channels],
                [kernel_size, kernel_size],
            )
            .with_stride([stride, stride])
            .with_padding([padding_size, padding_size])
            .with_padding_out([output_padding, output_padding])
            .init(device),
            norm_layer: if instance_norm {
                Some(
                    InstanceNormConfig::new(out_channels)
                        .with_affine(true)
                        .init(device),
                )
            } else {
                None
            },
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>, original_dims: [usize; 4]) -> Tensor<B, 4> {
        let [_, _, h, w] = original_dims;
        let mut padding_out = self.conv_transpose.padding_out;
        padding_out[0] -= h % 2;
        padding_out[1] -= w % 2;
        let input = conv_transpose2d(
            input,
            self.conv_transpose.weight.val(),
            self.conv_transpose.bias.as_ref().map(|b| b.val()),
            ConvTransposeOptions::new(
                self.conv_transpose.stride,
                self.conv_transpose.padding,
                padding_out,
                self.conv_transpose.dilation,
                self.conv_transpose.groups,
            ),
        );
        if let Some(norm_layer) = &self.norm_layer {
            norm_layer.forward(input)
        } else {
            input
        }
    }
}
