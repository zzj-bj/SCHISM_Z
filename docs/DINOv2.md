# DinoV2Segmentor: Foundation Model-Based Segmentation, from [Brondolo et al., 2024](https://arxiv.org/pdf/2407.18100)

## Overview

DinoV2Segmentor leverages DINOv2, a state-of-the-art self-supervised learning (SSL) method for pre-training vision foundation models. Foundation models like DINOv2 are pre-trained on massive datasets to serve as adaptable bases for various downstream tasks, enabling efficient fine-tuning on smaller, task-specific datasets. This approach delivers superior performance compared to training models from scratch, particularly in data-scarce scenarios.

DINOv2 employs Vision Transformers (ViTs) and attention mechanisms to create robust image encoders capable of capturing complex patterns and features. Pre-trained on diverse "natural" image datasets, DINOv2 models excel in tasks such as image classification, object detection, and segmentation. DinoV2Segmentor builds on this foundation, integrating modular segmentation heads to adapt these powerful encoders for precise, multi-class image segmentation.

---
## Features

- **Configurable Parameters**:
  - `channels`: Number of initial feature channels in the segmentation head.
  - `k_size`: Kernel size for convolutional layers.
  - `linear_head`: Option to switch between a linear or CNN-based segmentation head.
  - `n_features`: Number of transformer layers used for feature aggregation.
  - `quantize`: Enables 4-bit quantization for reduced memory usage and faster inference. Depending on your GPU, this option might have to be disabled in order for the training to run nicely.
  - `peft`: Incorporates LoRA for efficient fine-tuning of the backbone model.
  - `size`: Backbone model size (`small`, `base`, or `large`).
  - `num_classes`: Number of output segmentation classes.
  - `activation`: Specifies the activation function for the CNN Head (`relu`, `leakyrelu`, `sigmoid`, or `tanh`).
  - `r`: LoRA rank parameter for fine-tuning.
  - `lora_alpha`: LoRA alpha parameter for scaling the update.
  - `lora_dropout`: Dropout probability for LoRA fine-tuning.

- **Foundation Model Backbone**:
  - Utilises DINOv2, a transformer-based foundation model pre-trained on large datasets.
  - Offers superior feature extraction and generalisation capabilities.

- **Flexible Segmentation Heads**:
  - **Linear Head**: Lightweight option for simpler segmentation tasks.
  - **CNN Head**: Multi-block architecture for more sophisticated segmentation.

- **Efficient Training**:
  - Parameter-Efficient Fine-Tuning (PEFT) via LoRA for training specific layers.
  - Quantization to optimise memory and computational requirements.

---
## How It Works

### Backbone
The DINOv2 foundation model extracts hierarchical features from input images. Depending on `n_features`, multiple transformer layers are aggregated for segmentation.

### Segmentation Heads
#### Linear Head
A simple and lightweight head designed for basic segmentation tasks. It processes transformer features and directly generates the segmentation map.

**Key Features**:
- Batch normalisation followed by a 1x1 convolution.
- Efficient upsampling with bilinear interpolation to match input image dimensions.

#### CNN Head
A more advanced head for high-quality segmentation, leveraging convolutional blocks for feature refinement and upsampling.

**Key Features**:
- Multi-block architecture with configurable `channels`.
- Upsampling via bicubic interpolation and pixel shuffling.
- Supports multiple activation functions (e.g., `relu`, `leakyrelu`).
- Dropout for regularisation and robust segmentation outputs.

### Output
The segmentation head generates a final segmentation map with `num_classes` channels, providing per-pixel class probabilities.

---
## Constructor Parameters

| Parameter       | Type    | Description                                                                | Default    |
|-----------------|---------|----------------------------------------------------------------------------|------------|
| `n_block`       | `int`   | Number of convolutional blocks in the CNN-based segmentation head.         | 4          |
| `channels`      | `int`   | Number of channels in the CNN-based head's first layer.                   | 512        |
| `k_size`        | `int`   | Kernel size for all convolutional layers.                                 | 3          |
| `linear_head`   | `bool`  | Whether to use a linear segmentation head (`True`) or CNN-based head.     | `True`     |
| `n_features`    | `int`   | Number of transformer layers used for feature extraction.                 | 1          |
| `quantize`      | `bool`  | Enables 4-bit quantization for memory efficiency.                         | `True`    |
| `peft`          | `bool`  | Enables parameter-efficient fine-tuning (LoRA).                          | `True`    |
| `size`          | `str`   | Backbone size (`small`, `base`, or `large`).                              | `base`     |
| `num_classes`   | `int`   | Number of output segmentation classes.                                    | 3          |
| `activation`    | `str`   | Activation function for CNN-based segmentation head.                      | `relu`     |
| `r`             | `int`   | LoRA rank parameter for fine-tuning.                                      | 32         |
| `lora_alpha`    | `int`   | LoRA alpha parameter for fine-tuning.                                     | 32         |
| `lora_dropout`  | `float` | Dropout probability for LoRA fine-tuning.                                 | 0.1        |

---
