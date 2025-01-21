# CT Scanner Segmentation Framework
 
This framework provides tools for semantic segmentation of CT scanner images of rocks, but it is also applicable to any kind of images as long as semantic segmentation is required. The framework supports both training and inference workflows.
 
## Features
 
- **Training**: Allows users to train models on selected datasets.
- **Inference**: Enables users to predict segmentation masks on new data using trained models.
- **Data Handling**: Supports datasets with images and masks arranged in a specific directory structure.
- **Model Configuration**: Offers flexible model configuration through an INI file.
- **Normalization**: Supports dataset normalization using mean and standard deviation values provided via a JSON file or defaults to 0.5.
 
---
 
## Directory Structure
 
The data should be organized as follows:
```
datasets
|_dataset 1
   |_images
   |_masks
|_dataset 2
   |_images
   |_masks
|_dataset n
   |_images
   |_masks
|_data_stats.json (optional)
```

- **Images**: Directory containing the input images.
- **Masks**: Directory containing the corresponding segmentation masks.
- **data_stats.json**: (Optional) JSON file providing mean and standard deviation values for normalization.
 
---
 
## Configuration
 
Users must create an INI file to specify model parameters, optimizer settings, scheduler configurations, and training options.
 
### Example INI File
 
```
[Model]
n_block=5
channels=8
num_classes=3
model_type=UnetSegmentor
k_size=3
activation=leakyrelu
 
[Optimizer]
optimizer=Adam
 
[Scheduler]
scheduler=ConstantLR
 
[Training]
batch_size=4
val_split=0.8
epochs=3
metrics=DiceScore
 
[Data]
crop_size=128
img_res=560
num_samples=70
```

## Sections

### [Model]
Defines the model architecture and hyperparameters.

- `n_block`, `channels`, `num_classes` are mandatory.
- Optional parameters like `k_size`, `activation` have preset defaults.
- Supported models:
  - `UnetVanilla`,
  - `UnetSegmentor` (our adaptation of Liang et al. 2022),
  - `DINOv2` (our own implementation; see [this link](https://github.com/FloFive) for more info).
  
More models will be added in the future.

### [Optimizer]
Specifies the optimizer. Choices include:

- `Adagrad`
- `Adam`
- `AdamW`
- `NAdam`
- `RMSprop`
- `RAdam`
- `SGD`

These options are derived from the `torch.optim` library, and parameters, if any, should match the library's documentation.

### [Scheduler]
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

### [Training]
Configures training parameters.

- `batch_size`, `val_split`, `epochs` are required.
- Metrics: `DiceScore`, `GeneralizedDiceScore`, `MeanIoU` (preset).
- Note that the loss functions in use are `CategoricalCrossEntropy` for multiclass, and `BinaryCrossEntropy` for binary. Users don't need to specify it.


### [Data]
Configures data handling.

- `crop_size`: Size of image crops.
- `img_res`: Resolution to resize crops during training and inference.
- `num_samples`: Number of samples to use.

### Note
During training, images are split into crops of user-defined size (`crop_size`), then resized to `img_res` to ensure compatibility with various machines and GPUs while preserving detail. Upon completing a training session, several files will be generated in the weight folder:

- **data_stats.json**: Contains the standard deviation and mean values used to normalize the images.
- **hyperparameters.ini**: A copy of the INI file used for the training session.
- **learning_curves.png**: Displays the loss and metrics values as a function of the epochs.
- **model_best{metric(s)}.pth**: Contains the best model weights based on each metric specified in the INI file.
- **model_best_loss.pth**: Contains the best model weights based on the loss value.
- **test/train/val_indices.txt**: Records the indices of images and masks used for training, validation, and testing. These indices are formatted as `[dataset subfolder][image or mask number in the folder]`. For example, if you have 5,000 image/mask pairs, but `num_samples` is set to 3,000 and `val_split` is 0.8, then 2,400 indices will be recorded in `train_indices.txt`, 600 in `val_indices.txt`, and the remaining 2,000 in `test_indices.txt`.

During inference, the same process is applied, with patches reassembled into the original image.

## Training Workflow
1. Prepare the dataset following the directory structure.
2. Create an INI file with the necessary configurations.
3. Run the training script, specifying the dataset, output folder for weights, and the INI file.

## Inference Workflow
1. Use the trained weights saved during the training phase.
2. Provide the dataset for prediction.
3. Run the inference script specifying the weight folder and the data for prediction.

## Contributions
Contributions are welcome! Please fork the repository and submit a pull request.
