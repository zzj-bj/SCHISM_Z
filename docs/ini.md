
## [Model]
Defines the model architecture and hyperparameters.

- `num_classes` is mandatory.
- Optional parameters like `k_size`, `activation`, `n_block`, and `channels` have predefined default values. These parameters vary depending on the specific network. Please look at the network's documentation or file for detailed guidance on parameter configuration.
- Supported models:
  - `UnetVanilla` --> [more info here](https://github.com/FloFive/SCHISM/blob/main/docs/UnetVanilla.md) (preset by default)
  - `UnetSegmentor` (our adaptation of Liang et al. 2022) --> [more info here](https://github.com/FloFive/SCHISM/blob/main/docs/UnetSegmentor.md)
  - `DINOv2` (our own implementation; see [this link](https://github.com/FloFive) for the related paper) --> [more info here](https://github.com/FloFive/SCHISM/blob/main/docs/DINOv2.md)
  
More models will be added in the future.

## [Optimizer]
Specifies the optimizer. Choices include:

- `Adagrad`
- `Adam` (preset by default)
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
- `ConstantLR` (preset by default)
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

## [Loss]
Specifies the loss. Choices include:

- `PoissonNLLLoss`
- `CrossEntropyLoss` (preset by default)
- `BCEWithLogitsLoss`
- `GaussianNLLLoss`
- `NLLLoss`

These options are derived from `torch.nn`. Parameters, if any, should match the library's documentation.

## [Training]
Configures training parameters.

- `batch_size`, `val_split`, and `epochs` are required.
- `metrics`: `Jaccard` (preset by default), `F1`, `Accuracy`, `Precision` and `Recall`. To include a confusion matrix plot at the end of training, add `ConfusionMatrix` to the list of metrics.
- `weights`: The class weights for each present class will be calculated if this option is set to `True`. By default, it is set to `False`.


## [Data]
Configures data handling.

- `crop_size`: Size of image crops. The default value is 224px.
- `img_res`: Resolution to resize crops during training and inference. The default value is 560px.
- `num_samples`: Number of samples to use. The default value is 500 samples.
- `ignore_background`: Set whether the background class be ignored (`True`) during training and inference, or not (`False`). If set to `True`, `num_classes` should be decreased by one.

During training, images are split into crops of user-defined size (`crop_size`), then resized to `img_res` to ensure compatibility with various machines and GPUs while preserving detail. The same process is applied during inference, with patches reassembled into the original image.

