#![cfg_attr(not(test), no_std)]
extern crate alloc;

use alloc::vec::Vec;
use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode},
    {InstanceNorm, InstanceNormConfig, Relu},
};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct TransformerNet<B: Backend> {
    conv1: ConvLayer<B>,
    in1: InstanceNorm<B>,
    conv2: ConvLayer<B>,
    in2: InstanceNorm<B>,
    conv3: ConvLayer<B>,
    in3: InstanceNorm<B>,
    res1: ResidualBlock<B>,
    res2: ResidualBlock<B>,
    res3: ResidualBlock<B>,
    res4: ResidualBlock<B>,
    res5: ResidualBlock<B>,
    deconv1: UpsampleConvLayer<B>,
    in4: InstanceNorm<B>,
    deconv2: UpsampleConvLayer<B>,
    in5: InstanceNorm<B>,
    deconv3: ConvLayer<B>,
    relu: Relu,
}

impl<B: Backend> TransformerNet<B> {
    pub fn init(device: &B::Device) -> Self {
        Self {
            conv1: ConvLayer::init(3, 32, 9, 1, device),
            in1: InstanceNormConfig::new(32).with_affine(true).init(device),
            conv2: ConvLayer::init(32, 64, 3, 2, device),
            in2: InstanceNormConfig::new(64).with_affine(true).init(device),
            conv3: ConvLayer::init(64, 128, 3, 2, device),
            in3: InstanceNormConfig::new(128).with_affine(true).init(device),
            res1: ResidualBlock::init(128, device),
            res2: ResidualBlock::init(128, device),
            res3: ResidualBlock::init(128, device),
            res4: ResidualBlock::init(128, device),
            res5: ResidualBlock::init(128, device),
            deconv1: UpsampleConvLayer::init(128, 64, 3, 1, Some(2.0), device),
            in4: InstanceNormConfig::new(64).with_affine(true).init(device),
            deconv2: UpsampleConvLayer::init(64, 32, 3, 1, Some(2.0), device),
            in5: InstanceNormConfig::new(32).with_affine(true).init(device),
            deconv3: ConvLayer::init(32, 3, 9, 1, device),
            relu: Relu::new(),
        }
    }

    pub async fn forward(&self, input: &[f32], width: usize, height: usize) -> Vec<f32> {
        let input = Tensor::<B, 1>::from_floats(input, &B::Device::default()).reshape([
            1,
            3,
            width,
            height,
        ]);

        let input = self
            .relu
            .forward(self.in1.forward(self.conv1.forward(input)));
        let input = self
            .relu
            .forward(self.in2.forward(self.conv2.forward(input)));
        let input = self
            .relu
            .forward(self.in3.forward(self.conv3.forward(input)));
        let input = self.res1.forward(input);
        let input = self.res2.forward(input);
        let input = self.res3.forward(input);
        let input = self.res4.forward(input);
        let input = self.res5.forward(input);
        let input = self
            .relu
            .forward(self.in4.forward(self.deconv1.forward(input)));
        let input = self
            .relu
            .forward(self.in5.forward(self.deconv2.forward(input)));
        let output = self.deconv3.forward(input);

        output.into_data_async().await.to_vec().unwrap()
    }
}

#[derive(Module, Debug)]
pub struct ConvLayer<B: Backend> {
    reflection_pad: ReflectionPad2d,
    conv2d: Conv2d<B>,
}

impl<B: Backend> ConvLayer<B> {
    pub fn init(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        device: &B::Device,
    ) -> Self {
        let kernel_size = kernel_size / 2;
        Self {
            reflection_pad: ReflectionPad2d::init([
                kernel_size,
                kernel_size,
                kernel_size,
                kernel_size,
            ]),
            conv2d: Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
                .with_stride([stride, stride])
                .init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let padded = self.reflection_pad.forward(input);
        self.conv2d.forward(padded)
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
        let right = input
            .clone()
            .slice(s![.., .., .., (w - pr)..w])
            .flip([3]);
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
pub struct ResidualBlock<B: Backend> {
    conv1: ConvLayer<B>,
    in1: InstanceNorm<B>,
    conv2: ConvLayer<B>,
    in2: InstanceNorm<B>,
    relu: Relu,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            conv1: ConvLayer::init(channels, channels, 3, 1, device),
            in1: InstanceNormConfig::new(channels)
                .with_affine(true)
                .init(device),
            conv2: ConvLayer::init(channels, channels, 3, 1, device),
            in2: InstanceNormConfig::new(channels)
                .with_affine(true)
                .init(device),
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = input.clone();
        let out = self
            .relu
            .forward(self.in1.forward(self.conv1.forward(input)));
        self.in2.forward(self.conv2.forward(out)) + residual
    }
}

#[derive(Module, Debug)]
pub struct UpsampleConvLayer<B: Backend> {
    interpolate2d: Option<Interpolate2d>,
    reflection_pad: ReflectionPad2d,
    conv2d: Conv2d<B>,
}

impl<B: Backend> UpsampleConvLayer<B> {
    pub fn init(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        upsample: Option<f32>,
        device: &B::Device,
    ) -> Self {
        let kernel_size = kernel_size / 2;
        Self {
            interpolate2d: upsample.map(|scale_factor| {
                Interpolate2dConfig::new()
                    .with_mode(InterpolateMode::Nearest)
                    .with_scale_factor(Some([scale_factor, scale_factor]))
                    .init()
            }),
            reflection_pad: ReflectionPad2d::init([
                kernel_size,
                kernel_size,
                kernel_size,
                kernel_size,
            ]),
            conv2d: Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
                .with_stride([stride, stride])
                .init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let input = if let Some(interpolate2d) = &self.interpolate2d {
            interpolate2d.forward(input)
        } else {
            input
        };
        self.conv2d.forward(self.reflection_pad.forward(input))
    }
}

#[derive(Module, Debug)]
pub struct What<B: Backend> {
    conv1: ConvLayer<B>,
    in1: InstanceNorm<B>,
    conv2: ConvLayer<B>,
    in2: InstanceNorm<B>,
    conv3: ConvLayer<B>,
    in3: InstanceNorm<B>,
    res1: ResidualBlock<B>,
    res2: ResidualBlock<B>,
    res3: ResidualBlock<B>,
    res4: ResidualBlock<B>,
    res5: ResidualBlock<B>,
    deconv1: UpsampleConvLayer<B>,
    in4: InstanceNorm<B>,
    deconv2: UpsampleConvLayer<B>,
    in5: InstanceNorm<B>,
    deconv3: ConvLayer<B>,
    relu: Relu,
}
