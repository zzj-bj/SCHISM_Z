"""
            ███████╗   ██████╗  ██╗  ██╗  ██╗  ███████╗  ███╗   ███╗
            ██╔════╝  ██╔════╝  ██║  ██║  ██║  ██╔════╝  ████╗ ████║
            ███████╗  ██║       ███████║  ██║  ███████╗  ██╔████╔██║
            ╚════██║  ██║       ██╔══██║  ██║  ╚════██║  ██║╚██╔╝██║
            ███████║  ╚██████╗  ██║  ██║  ██║  ███████║  ██║ ╚═╝ ██║
            ╚══════╝   ╚═════╝  ╚═╝  ╚═╝  ╚═╝  ╚══════╝  ╚═╝     ╚═╝
            Semantic Classification of High-resolution Imaging for Scanned Materials    
"""

# @title ## **Hyperparameters & parameters (manual setting)**
# @markdown &lArr; _**Press the triangle to run this cell after setting the parameters**_

import sys
import os
import colorsys
# import random
# from classes.stack import Stack
# from classes.resUNet import ResUNet
# from classes.util import Util
# from classes.clr_callback  import CyclicLR
# import datetime
# import cv2
# import pandas as pd
# from matplotlib.colors import ListedColormap
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors
import tensorflow as tf
# import keras
# from tensorflow.keras.metrics import IoU, F1Score, BinaryIoU
# from tensorflow.keras.metrics import BinaryAccuracy, Accuracy, Hinge, MeanIoU
# from keras import backend as K

# os.environ["SM_FRAMEWORK"] = "tf.keras"
# import segmentation_models as sm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# if 'google.colab' in sys.modules:
#     from google.colab import drive
#     drive.mount('/content/gdrive')
#     project_dir = '/content/gdrive/MyDrive/Schism/'
#     sys.path.append(os.path.join(project_dir, "code/10-11-23-AC/"))
# else:

# Setting project directories
project_dir = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(project_dir, "code"))
runs_dir = os.path.join(project_dir, 'runs')
data_dir = os.path.join(project_dir, 'data')


# from tensorflow.keras.optimizers import Adam
# from tensorboard import program
# from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def generate_colors(n_class):
    # Ensure there's at least one class
    n_class = max(n_class, 1)

    # Generate colors evenly distributed in the color space
    hsv_colors = [(x / n_class, 1, 1) for x in range(n_class)]

    # Convert HSV colors to RGB
    rgb_colors = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*hsv)) for hsv in hsv_colors]

    return rgb_colors


def remove_spaces_from_list(word_list):
    return [word.replace(" ", "") for word in word_list]


# Example usage:
IMAGE_TYPE = {'scanner': 0, 'image' :1} # not to be changed

# @markdown `img_side_length` : choose a pixel side length from the dropdown menu.
# Has to be divisible by 2 * the number of **MaxPooling()** (4 here)
img_side_length = 512  # @param [128, 224, 256, 384, 512, 768, 1024, 1280, 1664, 2048] {type:"raw"}
img_width = int(img_side_length)
img_height = int(img_side_length)

# img_width = 256 # @param {type:"integer"}
# img_height = 256 # @param {type: "integer"}
if (img_height % 16 != 0) and (img_width % 16 != 0):
    raise ValueError('Image size must be a multiple of 16')


# @markdown `num_sample` : set the number of slices. If set to _None_, all images will be selected.
num_sample = 1000  #@param {type:"raw"}
if num_sample is not None:
    num_sample = int(num_sample)

#@markdown `imgtype` : specify the image type.
imgtype = "scanner" # @param ["scanner", "image"]
imageType = IMAGE_TYPE[imgtype]

#@markdown  `pretrained` : tick to use a pretrained model. The architecture can be selected fom the `backbone` dropdown list.
pretrained = True #@param {type:"boolean"}

# supprimer les backbones
#@markdown `backbone` : specify the model backbone type.
backbone = 'resnet50' # @param ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101']

#@markdown `image_preprocessing_functions` : specify function names as a list, like _\["example1", "example2"\]_, to define the image preprocessing functions. Valid options include:
#@markdown * CLAHE
#@markdown * adjustlog
#@markdown * adjustgamma
#@markdown * normalize_histograms
#@markdown * contrast_stretching
image_preprocessing_functions = [CLAHE]# @param {type:"raw"}
image_preprocessing_functions = remove_spaces_from_list(image_preprocessing_functions)

#@markdown `metrics` : define metric name(s) by providing a list, such as _\[example1, example2\]_. If no metrics are set, the default metric used will be accuracy. Valid options include:
#@markdown * Recommended for binary classification
#@markdown    * BinaryIoU
#@markdown    * BinaryAccuracy
#@markdown    * BinaryCrossentropy
#@markdown * Recommended for multiclass classification
#@markdown    * CategoricalAccuracy
#@markdown    * CategoricalCrossentropy
#@markdown    * OneHotMeanIoU
#@markdown    * OneHotIoU
metrics = ["CategoricalCrossentropy, OneHotMeanIoU"] # @param {type:"raw"}
metrics = remove_spaces_from_list(metrics)

#@markdown `featuremaps` : specify the number of filters used in the deep learning architecture from the dropdown list. The number of filters at its deepest layer will be calculated as filters * 16.
featuremaps = 16  # @param [4, 8, 16, 32, 64] {type:"raw"}

#@markdown `epochs` : set the number of training iterations for the model.
epochs = 20 #@param {type:"integer"}

#@markdown `val_split` : set the percentage split between training and testing data using the slider.
val_split = 80  #@param {type:"slider", min:0, max:100, step:5}
val_split = val_split / 100

#@markdown  `displaySummary` : tick to display a summary of the Convolutional Neural Network, including information on layers, parameters, and more.
displaySummary = True #@param {type:"boolean"}

#@markdown `maxNorm` : a regularization technique, typically set between 1 to 5, implementing "max-norm regularization." This method helps prevent neural network weights from growing excessively during training, mitigating the risk of overfitting.
maxNorm = 3 #@param {type:"integer"}

#@markdown `learningRate` : adjust the learning rate within the range of 0.1 to 0.00001 to control the step size during optimization.
learningRate = 1e-4 #@param {type:"raw"}

#@markdown `batchNorm` : if True: reduces overfitting/increases generalization.
batchNorm = True # @param {type:"boolean"}

#@markdown `batch_size` : define the batch size for the tf tensors. Set accordingly to GPU availability, ram space, and dataset size.
batch_size = 3  # @param {type:"integer"}

#@markdown `save_model` : saves the model/ weights / and metric curves.
save_model = True # @param {type:"boolean"}

#@markdown `dropOut` : tick the box to enable dropout regularization to reduce overfitting and enhance generalization.
dropOut = True # @param {type:"boolean"}

#@markdown `dropOutRate` : if `dropOut` is enabled, adjust the dropout rate starting from 0.1 until reaching the optimal dropout rate.
dropOutRate = 0.4 #@param {type:"raw"}

#@markdown `L2` : enter a value between 0.00001 and 0.1 to apply regularization, which adds a penalty to the model for having large weights and helps prevent overfitting. The default value is 0.0001.
L2 = 1e-4 #@param {type:"raw"}

#@markdown `early_stopping` : specify metrics to monitor (metrics must match those set in metrics). Training halts as soon as performance on the validation set diminishes. Use the format _\["example1", "example2"\]_ Leave the brackets empty for no early stopping.
early_stopping = [] # @param {type:"raw"}
early_stopping = remove_spaces_from_list(early_stopping)
common_metrics = [value for value in metrics if value in early_stopping]
if len(early_stopping) > 0:
  if not common_metrics:
    raise Exception('Early stopping metrics need to be specified in metrics')

#@markdown `loss_early_stopping` : tick the box to halt model training as soon as performance on a validation loss diminishes.
loss_early_stopping = False # @param {type:"boolean"}

#@markdown `patience` : if either early_stopping or loss_early_stopping is set, specify the number of epochs before stopping.
patience = 25 # @param {type:"integer"}
