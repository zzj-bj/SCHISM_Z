# UnetSegmentor : after [Liang et al., 2022](https://www.sciencedirect.com/science/article/abs/pii/S0098300422001662)

## Overview

This network is an adaptation of [Liang et al., 2022](https://www.sciencedirect.com/science/article/abs/pii/S0098300422001662). UnetSegmentor is a CNN designed for efficient, automated segmentation of micro-CT images, critical for analyzing rock microstructures and physical properties. It replaces labor-intensive manual workflows with a fast, accurate, and flexible architecture for multi-class segmentation. Ideal for geoscience applications, it ensures precise mineral and pore identification with minimal effort.

---

## Features

- **Configurable Parameters**:
  - `n_block`: Number of encoder and decoder blocks.
  - `channels`: Number of initial feature channels, doubled in each subsequent block.
  - `k_size`: Kernel size for convolutional layers.
  - `activation`: Choice of activation function (`relu`, `leakyrelu`, `sigmoid`, or `tanh`).
  - `p`: Dropout probability for regularization.
  - `num_classes`: Number of output classes for segmentation.

- **Symmetrical Encoder-Decoder Design**:
  - **Encoder**: Downsamples the input using convolution and pooling layers.
  - **Bridge**: Bottleneck layer connecting the encoder and decoder.
  - **Decoder**: Upsamples and refines features with transposed convolutions and skip connections.
  - **Output Layer**: Generates a segmentation map with `num_classes` channels.

- **Modular Blocks**:
  - Encoder and decoder are composed of reusable convolutional blocks for flexibility and clarity.

---

## How It Works

### Encoder Path
The encoder downsamples the input image, extracting hierarchical features using convolutional blocks followed by max-pooling. Each block doubles the number of channels, enhancing feature richness.

### Decoder Path
The decoder upsamples the features using transposed convolutions, combines them with corresponding encoder outputs (via skip connections), and refines them through convolutional layers.

### Bridge
A bottleneck layer connects the deepest encoder block to the decoder. It captures global contextual features before upsampling begins.

### Output
A 3x3 convolution maps the final feature maps to `num_classes` channels, producing per-pixel class probabilities.

---

## Constructor Parameters

| Parameter      | Type    | Description                                                                 | Default |
|----------------|---------|-----------------------------------------------------------------------------|---------|
| `n_block`      | `int`   | Number of encoder and decoder blocks.                                       | 4       |
| `channels`     | `int`   | Number of channels in the first convolution layer, doubled in each block.   | 8       |
| `k_size`       | `int`   | Kernel size for all convolutional layers.                                  | 3       |
| `activation`   | `str`   | Activation function (`relu`, `leakyrelu`, `sigmoid`, `tanh`).               | `relu`  |
| `p`            | `float` | Dropout probability for regularization.                                    | 0.5     |
| `num_classes`  | `int`   | Number of output segmentation classes.                                     | 3       |

---
