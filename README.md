<div align="center">
  <img width="450" height="450" alt="Hi you :)" src="https://github.com/user-attachments/assets/b85455bc-6234-48cf-8cc7-97ce432eae71" />
</div>

[SCHISM](https://www.youtube.com/watch?v=MM62wjLrgmA&ab_channel=TOOLVEVO) stands for _Semantic Classification of High-resolution Imaging for Scanned Materials_. This framework provides tools for the semantic segmentation of CT scanner images of rocks, but it is also applicable to any kind of image as long as semantic segmentation is required. The framework supports both training and inference workflows.

---
## :gear: Installation

1. Clone this repository to your local machine:
   ``` git clone git@github.com:FloFive/SCHISM.git ```

3. Navigate to the cloned directory:
   ``` cd <some path> SCHISM ```
3. Install the library (Python 3.9 mini is required)
   ``` pip install -e .```
   
---
## :question: How to use

SCHISM offers three main functionalities: **Preprocessing**,  **Training** and **Inference**.

### General steps
1. Organise your data in the required structure (see Data Preparation).
2. Set up an INI configuration file (see INI File Setup).
3. Run the main script:
   ``` python schism.py ```
4. Navigate through the command-line menu:
    - Option 1- Preprocessing: Customise your data by computing dataset-specific mean and standard deviation for improved normalisation during training and/or reformat your segmentation masks to match the input format required by SCHISM.
    - Option 2- Training: Train a new model.
    - Option 3- Inference: Make predictions using a trained model.

---
### Preprocessing workflow
Three available options :
   - Auto brightness/contrast adjustment: Automatically adjust the brightness and contrast of your images. This process rescales pixel values based on histogram minimum/maximum (hmin/hmax). The original `images` folder will be renamed to `raw_images`, and the new adjusted images will be saved in a newly created `images` folder. The function has been inspired by the work of [Schindelin et al. 2012](https://www.nature.com/articles/nmeth.2019) / [Fiji](https://github.com/fiji). Two options are made available to the user:
       - ref image: Use one chosen image to set hmin/hmax for all images (consistent contrast).
       - per image: Compute hmin/hmax separately for each image (max local contrast, less consistency).
   - JSON generation: Compute the mean and standard deviation from part or all of your dataset. The results will be saved as a JSON file in your dataset folder.
   - Normalisation: Process your data to produce SCHISM-compatible segmentation masks. The original `masks` folder will be renamed to `raw_masks`, and the new, normalised masks will be saved in a newly created `masks` folder.

:warning: Input data must follow the format described in the [Data preparation](https://github.com/FloFive/SCHISM/tree/main?tab=readme-ov-file#-data-preparation) section of the documentation.
    
---
### Training workflow
1. Prepare the dataset: Ensure the dataset is organised according to the required directory structure (presented below).
2. Create an INI file: Define training parameters such as learning rate, batch size, and model architecture in the INI file (presented below).
3. Run the training command: Launch the training process, then select the training option and specify:
    - The dataset directory: Contains one or more datasets. The ordering and sorting of the data are explained [later in this README](https://github.com/FloFive/SCHISM/tree/main?tab=readme-ov-file#-data-preparation).
    - The output folder: This is the workspace where all generated results are stored. After training, it will include the model weights, along with other relevant outputs. Each file within output/ is described in detail [later in this README](https://github.com/FloFive/SCHISM/tree/Pierre_dev?tab=readme-ov-file#-training-output-files). 
    - The path to the INI file ([described here](https://github.com/FloFive/SCHISM/tree/Pierre_dev?tab=readme-ov-file#scroll-ini-file-setup)).

---
### Inference workflow
To make predictions:
1. Use trained weights: Ensure the trained model weights are saved from the training phase.
2. Prepare the dataset for prediction: Ensure your data is structured in the format required by SCHISM for inference. See the [Data preparation](https://github.com/FloFive/SCHISM/tree/main?tab=readme-ov-file#-data-preparation) section for details.
3. Run the inference command: Launch the prediction process, then select the training option and specify:
    - The folder containing trained weights.
    - The dataset for prediction.

Predictions on the user's data will be saved in a directory named after the metric used during inference (e.g., `preds_X`, where `X` is the name of the selected evaluation metric).

---
## :scroll: .ini configuration file

Below is an example of an `.ini` configuration file. For detailed explanations of the network settings and the full INI specification, see the [INI file documentation](https://github.com/FloFive/SCHISM/blob/main/docs/ini.md).

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

---
## 👾 Data preparation

The data should be organised as follows:

``` 
data/  <--- Select this folder for normalisation, training, or inference
├── dataset_1/
│   ├── images/       # Grayscale TIFF images (e.g., image0000.tif, image0001.tif, ...)
│   ├── masks/        # Corresponding TIFF masks (e.g., mask0000.tif for image0000.tif)
│   └── raw_masks/    # Optional: original, unnormalized masks (renamed during normalisation)
├── dataset_2/
│   ├── images/
│   ├── masks/
│   └── raw_masks/
├── ...
├── dataset_n/
│   ├── images/
│   ├── masks/
│   └── raw_masks/
└── data_stats.json   # Optional, generated during JSON creation
```

### Directory descriptions

   - images/: Contains the grayscale TIFF input images, sequentially named for logical ordering.
   - masks/: Contains segmentation masks in SCHISM-compatible format (after normalisation, or provided by the user).
   - raw_masks/: Backup of original masks before normalisation.
   - data_stats.json: (Optional) Automatically generated during JSON creation. Stores mean and standard deviation values per dataset.
  
---
## 💾 Training Output Files

 Upon completing a training session, several files will be generated in the weight folder:

- **data_stats.json**: The standard deviation and mean values used to normalise the images.
- **hyperparameters.ini**: A copy of the INI file used for the training session.
- **learning_curves.png**: Displays the loss and metrics values as a function of the epochs.
- **model_best_{metric(s)}.pth**: Contains the best model weights based on each metric specified in the INI file.
- **model_best_loss.pth**: Contains the best model weights based on the loss value.
- **test/train/val_indices.txt**: Saves the indices of images and masks used for training, validation, and testing. These indices are formatted as `[dataset subfolder][image or mask number in the folder]`. For example, if you have 5,000 image/mask pairs, but `num_samples` is set to 3,000 and `val_split` is 0.8, then 2,400 indices will be recorded in `train_indices.txt`, 600 in `val_indices.txt`, and the remaining 2,000 in `test_indices.txt`.

---
## 🐞 Debugging

A constant named `DEBUG_MODE` is defined in `tools/constants.py`.  
- If `DEBUG_MODE = True`, SCHISM will display the full Python trace when an error occurs.  
- If `DEBUG_MODE = False`, only a concise error message is shown.  

This allows switching between developer-friendly debugging and cleaner end-user output.

---
## :heart_on_fire: Contributions
Contributions are welcome! Please fork the repository and submit a pull request.

---
## :disappointed: Found a bug? 
If you spot a bug or have a problem running the code, please open an issue.
If you have any questions or need further assistance, don't hesitate to contact Florent Brondolo ([florent.brondolo@akkodis.com](mailto:florent.brondolo@akkodis.com))
or Samuel Beaussant ([samuel.beaussant@akkodis.com](mailto:samuel.beaussant@akkodis.com)).

---
## 📚 Citation / Bibtex

If you use our solution or find our work helpful, please consider citing it as follows:

```
@misc{schism2025,
  title       = {SCHISM: Semantic Classification of High-resolution Imaging for Scanned Materials},
  author      = {Florent Brondolo and Samuel Beaussant and Mehdi Mankaï and Saïd Ezzedine and Ozan Yazar and Pierre Fancelli},
  year        = {2025},
  howpublished= {\url{https://github.com/FloFive/SCHISM}},
  note        = {GitHub repository}
}
```
