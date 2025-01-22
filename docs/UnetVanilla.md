# UnetVanilla: A Simple and Configurable U-Net Implementation

## Overview

UnetVanilla is a lightweight implementation of the U-Net architecture for image segmentation tasks. It provides flexibility in design through configurable parameters while maintaining the core principles of U-Net, such as encoder-decoder symmetry and skip connections.

---
## Features
- Customizable Parameters:
  - n_block: Number of encoder and decoder blocks.
  - channels: Number of initial feature channels, doubled in each subsequent block.
  - k_size: Kernel size for convolutional layers.
  - activation: Choice of activation function (relu, leakyrelu, sigmoid, or tanh).
  - num_classes: Number of output classes for segmentation.

- Flexible Architecture:
  - Encoder Path: Successive down-sampling with convolution and max-pooling layers.
  - Decoder Path: Up-sampling with transpose convolutions and skip connections.
  - Bridge: A bottleneck connecting the encoder and decoder paths.
  - Output Layer: Converts feature maps to class probabilities using a 1x1 convolution.
  
- Modular Design:
  - Encoders, decoders, and the bridge are built using reusable blocks for clarity and extensibility.

---
## How It Works
### Encoder Path
The encoder downsamples the input image, extracting hierarchical features using a series of convolutional blocks and max-pooling layers. Each block doubles the number of channels.

### Decoder Path
The decoder upsamples the feature maps using transpose convolutions, merges them with corresponding encoder outputs (via skip connections), and refines them through convolutional layers.

### Bridge
A bottleneck layer that serves as the connection between the deepest encoder block and the decoder.

### Output
A 1x1 convolution maps the final feature maps to `num_classes` channels, representing per-pixel class probabilities.

---
## Constructor Parameters

| Parameter      | Type   | Description                                                   | Default |
|----------------|--------|---------------------------------------------------------------|---------|
| `n_block`      | `int`  | Number of encoder/decoder blocks.                              | 4       |
| `channels`     | `int`  | Number of channels in the first layer, doubled in subsequent blocks. | 8       |
| `num_classes`  | `int`  | Number of output segmentation classes.                        | 3       |
| `k_size`       | `int`  | Kernel size for all convolutional layers.                     | 3       |
| `activation`   | `str`  | Activation function (`relu`, `leakyrelu`, `sigmoid`, `tanh`). | `relu`  |

---


