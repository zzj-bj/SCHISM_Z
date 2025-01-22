# SCHISM 

SCHISM stands for _Semantic Classification of High-resolution Imaging for Scanned Materials_. This framework provides tools for semantic segmentation of CT scanner images of rocks, but it is also applicable to any kind of image as long as semantic segmentation is required. The framework supports both training and inference workflows. As for the little trivia, this project got named after [this](https://www.youtube.com/watch?v=MM62wjLrgmA&ab_channel=TOOLVEVO) :) 

---
## Installation

1. Clone this repository to your local machine:
   ``` git clone <repository_url> ```

3. Navigate to the cloned directory:
   ``` cd <some path> SCHISM ```
   
---
## How to use

SCHISM offers two main functionalities: *Training* and *Inference*.

### General Steps
1. Organize your data in the required structure (see Data Preparation).
2. Set up an INI configuration file (see INI File Setup).
3. Run the main script:
   ``` python schism.py ```
4. Navigate through the command-line menu:
 - Option 1: Train a new model.
 - Option 2: Make predictions using a trained model.

---
### Training Workflow
1. Prepare the dataset: Ensure the dataset is organized according to the required directory structure.
2. Create an INI file: Define training parameters such as learning rate, batch size, and model architecture in the INI file.
3. Run the training command: Launch the training process, then select the training option and specify:
 - The dataset directory.
 - The output folder for model weights (and more).
 - The path to the INI file.

---
### Inference Workflow
To make predictions:
1. Use trained weights: Ensure the trained model weights are saved from the training phase.
2. Prepare the dataset for prediction: Organize the data in a compatible format.
3. Run the inference command: Launch the prediction process, then select the training option and specify:
 - The folder containing trained weights.
 - The dataset for prediction.

---
## INI File Setup
 
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
Please refer to the network's documentation for optional parameters specific to each model. Have a look a this page for more information about the INI setup.
 
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
 
## Sections














## Note
During training, images are split into crops of user-defined size (`crop_size`), then resized to `img_res` to ensure compatibility with various machines and GPUs while preserving detail. Upon completing a training session, several files will be generated in the weight folder:

- **data_stats.json**: Contains the standard deviation and mean values used to normalize the images.
- **hyperparameters.ini**: A copy of the INI file used for the training session.
- **learning_curves.png**: Displays the loss and metrics values as a function of the epochs.
- **model_best{metric(s)}.pth**: Contains the best model weights based on each metric specified in the INI file.
- **model_best_loss.pth**: Contains the best model weights based on the loss value.
- **test/train/val_indices.txt**: Records the indices of images and masks used for training, validation, and testing. These indices are formatted as `[dataset subfolder][image or mask number in the folder]`. For example, if you have 5,000 image/mask pairs, but `num_samples` is set to 3,000 and `val_split` is 0.8, then 2,400 indices will be recorded in `train_indices.txt`, 600 in `val_indices.txt`, and the remaining 2,000 in `test_indices.txt`.

The same process is applied during inference, with patches reassembled into the original image.


## Contributions
Contributions are welcome! Please fork the repository and submit a pull request.
