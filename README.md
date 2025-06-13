![gradient_image](https://github.com/user-attachments/assets/5e76c773-82b8-4790-b5b9-5bfff6eed0a1)

SCHISM stands for _Semantic Classification of High-resolution Imaging for Scanned Materials_. This framework provides tools for semantic segmentation of CT scanner images of rocks, but it is also applicable to any kind of image as long as semantic segmentation is required. The framework supports both training and inference workflows. As for the little trivia, this project got named after [this](https://www.youtube.com/watch?v=MM62wjLrgmA&ab_channel=TOOLVEVO) :) 

---
## :gear: Installation

1. Clone this repository to your local machine:
   ``` git clone git@github.com:FloFive/SCHISM.git ```

3. Navigate to the cloned directory:
   ``` cd <some path> SCHISM ```
3. Install the library (python 3.9 mini is required)
   ``` pip install -e .```
   
---
## :question: How to use

SCHISM offers three main functionalities: **Preprocesing** ,  **Training** and **Inference**.
the module 'Preprocesing' has two menus 'Json generation' and 'Normalization' 

### General Steps
1. Organize your data in the required structure (see Data Preparation).
2. Set up an INI configuration file (see INI File Setup).
3. Run the main script:
   ``` python schism.py ```
4. Navigate through the command-line menu:
    - Option 1: Preprocessing 
    - Option 2: Train a new model.
    - Option 3: Make predictions using a trained model.
5. The option '1: Preprocessing' opens a new menu. This new menu allows you to generate a file in 'Json' format or to normalize masks.
---
### Preprocesing
In this menu, we have access to two options that allow us to:
    - Json generation: in development
    - Normalization: creation of a grayscale image (0-255) in 8-bit format from a mask

### - Json generation
    This module is in development    

###  - Normalization
1. Prepare the dataset: Ensure the dataset is organized according to the required directory structure (presented below).    
2. run the mormalization command: Launch the preprocessing process, then lauch the normalization and specify:
    - The dataset directory (presented below).
3. During the normalization process, a directory named 'normalized' is created in each subdirectory. This is where the normalized images will be stored.If the directory does not exist, it will be created. Otherwise, the existing directory will be used. When creating the images, any images with the same name will be replaced.
4. At the end of the normalization process, a new organization for the subdirectories is obtained (see below).

---
### Training Workflow
1. Prepare the dataset: Ensure the dataset is organized according to the required directory structure (presented below).
2. Create an INI file: Define training parameters such as learning rate, batch size, and model architecture in the INI file (presented below).
3. Run the training command: Launch the training process, then select the training option and specify:
    - The dataset directory: contains one or more datasets. The ordering and sorting of the data are explained later in this readme.
    - The output folder: the space where, amongst others, a folder containing the model weights will be created after training. The files saved in the folder are later described in this readme.
    - The path to the INI file.

---
### Inference Workflow
To make predictions:
1. Use trained weights: Ensure the trained model weights are saved from the training phase.
2. Prepare the dataset for prediction: Organize the data in a compatible format.
3. Run the inference command: Launch the prediction process, then select the training option and specify:
    - The folder containing trained weights.
    - The dataset for prediction.
 4. A directory will be created. The name of this directory will follow the logic: 'preds_X', where 'X' represents the name of the metric used for this generation. For example, if the metric is 'Jaccard', the directory will be named 'preds_Jaccard'. Additionally, '_X' will also be appended to the names of the generated images (for instance, image000.tif will become image000_Jaccard.tif). If the directory does not exist, it will be created. Otherwise, the existing directory will be used. When creating the images, any images with the same name will be replaced.
---
## :scroll: INI File Setup

Below is an example of an INI file:

```
[Model]
n_block=4
channels=8
num_classes=3
model_type=UnetSegmentor
k_size=3
activation=leakyrelu
 
[Optimizer]
optimizer=Adam
lr=0.01

[Scheduler]
scheduler = ConstantLR

[Loss]
loss= CrossEntropyLoss
ignore_background=True
weights=True

[Training]
batch_size=4
val_split=0.8
epochs=50
metrics=Jaccard, ConfusionMatrix
 
[Data]
crop_size=128
img_res=560
num_samples=7000
```

For information on both the network configurations and the INI file setup, please refer to [this page](https://github.com/FloFive/SCHISM/blob/main/docs/ini.md).

---
## 👾 Data Preparation

The data should be organized as follows:

``` 
- Before normalization
data <--- Select this folder for data input during normalization, training or inference.
|_dataset 1/ 
|   |_images/ <--- Contains grayscale TIFF images, sequentially named for logical ordering (e.g., image0000.tif, image0001.tif, etc.).
|   |_masks/ <--- Contains corresponding TIFF masks, named to match their respective images (e.g., mask0000.tif for image0000.tif).
|_dataset 2/
|   |_images/
|   |_masks/
|_dataset n/
|   |_images/
|   |_masks/
|_data_stats.json <--- This file is optional.
```
``` 
 -After normalization
data <--- Select this folder for data input during normalization, training or inference.
|_dataset 1/ 
|   |_images/ <--- Contains grayscale TIFF images, sequentially named for logical ordering (e.g., image0000.tif, image0001.tif, etc.).
|   |_masks/ <--- Contains corresponding TIFF masks, named to match their respective images (e.g., mask0000.tif for image0000.tif).
|   |_Normalized/ <--- Contains images mormalized
|_dataset 2/
|   |_images/
|   |_masks/
|   |_Normalized/
|_dataset n/
|   |_images/
|   |_masks/
|   |_Normalized/
|_data_stats.json <--- This file is optional.


- **Images**: Directory containing the input images.
- **Masks**: Directory containing the corresponding segmentation masks.
- **Normalized**: Directory containing the corresponding images mormalized from the masks.
- **data_stats.json**: (Optional) A JSON file containing mean and standard deviation values for normalization. Currently, this file must be set manually and should follow this format:

```
{
    "dataset1": [
        [0.52, 0.52, 0.52],
        [0.31, 0.31, 0.31]
    ],
    "dataset2": [
        [0.46, 0.46, 0.46],
        [0.5, 0.5, 0.5]
    ],

   [...]

    "datasetn": [
        [0.11, 0.11, 0.11],
        [0.42, 0.42, 0.42]
    ]
}
```
---
## 💾 Training Output Files

 Upon completing a training session, several files will be generated in the weight folder:

- **data_stats.json**: The standard deviation and mean values used to normalize the images.
- **hyperparameters.ini**: A copy of the INI file used for the training session.
- **learning_curves.png**: Displays the loss and metrics values as a function of the epochs.
- **model_best_{metric(s)}.pth**: Contains the best model weights based on each metric specified in the INI file.
- **model_best_loss.pth**: Contains the best model weights based on the loss value.
- **test/train/val_indices.txt**: Records the indices of images and masks used for training, validation, and testing. These indices are formatted as `[dataset subfolder][image or mask number in the folder]`. For example, if you have 5,000 image/mask pairs, but `num_samples` is set to 3,000 and `val_split` is 0.8, then 2,400 indices will be recorded in `train_indices.txt`, 600 in `val_indices.txt`, and the remaining 2,000 in `test_indices.txt`.

---
## :heart_on_fire: Contributions
Contributions are welcome! Please fork the repository and submit a pull request.

---
## 📚 Citation / Bibtex

If you use our solution or find our work helpful, please consider citing it as follows:

```
@misc{schism2025,
  title       = {SCHISM: Semantic Classification of High-resolution Imaging for Scanned Materials},
  author      = {Florent Brondolo and Samuel Beaussant and Soufiane Elbouazaoui and Saïd Ezzedine and Pierre Fancelli},
  year        = {2025},
  howpublished= {\url{https://github.com/FloFive/SCHISM}},
  note        = {GitHub repository}
}
```
