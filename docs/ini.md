
## [Model]
Defines the model architecture and hyperparameters.

- `num_classes` is mandatory.
- Optional parameters like `k_size`, `activation`, `n_block`, and `channels` have predefined default values. These parameters vary depending on the specific network. Please refer to the network's documentation or file for detailed guidance on parameter configuration.
- Supported models:
  - `UnetVanilla` --> [more info here](https://github.com/FloFive/SCHISM/blob/main/docs/UnetVanilla.md)
  - `UnetSegmentor` (our adaptation of Liang et al. 2022) --> [more info here](https://github.com/FloFive/SCHISM/blob/main/docs/UnetSegmentor.md)
  - `DINOv2` (our own implementation; see [this link](https://github.com/FloFive) for more info) --> [more info here](https://github.com/FloFive/SCHISM/blob/main/docs/DINOv2.md)
  
More models will be added in the future.

## [Optimizer]
Specifies the optimizer. Choices include:

- `Adagrad`
- `Adam`
- `AdamW`
- `NAdam`
- `RMSprop`
- `RAdam`
- `SGD`

These options are derived from the `torch.optim` library, and parameters, if any, should match the library's documentation.

## [Scheduler]
Specifies the learning rate scheduler. Options include:

- `LRScheduler`
- `LambdaLR`
- `MultiplicativeLR`
- `StepLR`
- `MultiStepLR`
- `ConstantLR`
- `LinearLR`
- `ExponentialLR`
- `PolynomialLR`
- `CosineAnnealingLR`
- `SequentialLR`
- `ReduceLROnPlateau`
- `CyclicLR`
- `OneCycleLR`
- `CosineAnnealingWarmRestarts`

These options are derived from `torch.optim.lr_scheduler`. Parameters, if any, should match the library's documentation.

## [Training]
Configures training parameters.

- `batch_size`, `val_split`, `epochs` are required.
- Metrics: `DiceScore`, `GeneralizedDiceScore`, `MeanIoU` (preset).
- Note that the loss functions in use are `CategoricalCrossEntropy` for multiclass, and `BinaryCrossEntropy` for binary. Users don't need to specify it.


## [Data]
Configures data handling.

- `crop_size`: Size of image crops.
- `img_res`: Resolution to resize crops during training and inference.
- `num_samples`: Number of samples to use.