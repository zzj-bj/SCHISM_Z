# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:33:49 2022
@author: florent.brondolo
"""
import sys
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from .stack import Stack
from .ProgressBar import ProgressBar
from math import sqrt
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.exposure import (match_histograms, equalize_adapthist, rescale_intensity,
                             equalize_hist, adjust_gamma, adjust_log, histogram)
from sklearn.preprocessing import OneHotEncoder

class Util:
    def __repr__(self):
        return 'Util'
    def __init__(self, **kwargs):
# Initialize properties with default values
        self.Xtrain = self.Ytrain = self.Xtest = self.Ytest = np.zeros([0, 0, 0, 0])
        self.name = kwargs.get('name', "")
        self.numSlice = kwargs.get('numSlice', None)
        self.image_preprocessing_functions = kwargs.get('image_preprocessing_functions', None)
        self.scaler = kwargs.get('scaler', None)
        self.validation_split = kwargs.get('validation_split', 0.3)
        self.inference = kwargs.get('isInference', False)
        if "stackImage" in kwargs and "stackLabel" not in kwargs:
            self._handle_single_stack(kwargs["stackImage"])
        elif "stackImage" in kwargs and "stackLabel" in kwargs:
            self._handle_stack_pairs(kwargs["stackImage"], kwargs["stackLabel"])
        else:
            raise ValueError(f"{repr(self)} - Only lists of stacks are accepted as inputs")
    def _handle_single_stack(self, stackImageList):
        """Handles cases where only image stack is provided (inference)."""
        self.imageType = stackImageList[0].getImageType()
        self.numClass = stackImageList[0].getNumClass()
        numSliceSampledPerStack = (self.numSlice // len(stackImageList)) if self.numSlice else None
        selectedSliceImage = []
        pb = ProgressBar(len(stackImageList), txt=f"{repr(self)} - Loading {self.name}")
        for stackImage in stackImageList:
            selectedSliceImage += self._select_slices(stackImage, numSliceSampledPerStack)
            pb += 1
        self.stackImage = Stack(
            imageType=self.imageType,
            width=stackImageList[0].getWidth(),
            height=stackImageList[0].getHeight(),
            isSegmented=stackImageList[0].getIsSegmented(),
            isSliceListSupplied=True,
            selectedFiles=selectedSliceImage,
            channel=stackImageList[0].getChannel(),
            numClass=self.numClass
        )
        self.numSlice = self.stackImage.getStackSize()
    def _handle_stack_pairs(self, stackImageList, stackLabelList):
        """Handles cases where image-label stack pairs are provided."""
        self._validate_image_types(stackImageList)
        self.numClass = stackImageList[0].getNumClass()
        numSliceSampledPerStack = (self.numSlice // len(stackImageList)) if self.numSlice else None
        selectedSliceImage = []
        selectedSliceLabel = []
        pb = ProgressBar(len(stackImageList), txt=f"{repr(self)} - Loading {self.name}")
        for img_stack, label_stack in zip(stackImageList, stackLabelList):
            selectedSliceImage += self._select_slices(img_stack, numSliceSampledPerStack)
            selectedSliceLabel += self._select_slices(label_stack, numSliceSampledPerStack)
            pb += 1
        self.stackImage = Stack(
            width=stackImageList[0].getWidth(),
            height=stackImageList[0].getHeight(),
            isSegmented=stackImageList[0].getIsSegmented(),
            isSliceListSupplied=True,
            selectedFiles=selectedSliceImage,
            channel=stackImageList[0].getChannel()
        )
# Normalize class labels
        stacked_labels = np.stack(selectedSliceLabel, axis=-1)
        class_mapping = self._create_class_mapping(stacked_labels)
        stacked_labels = self._normalize_class_labels(stacked_labels, class_mapping)
        self.uniqueClass = sorted(class_mapping.values())
        self.stackLabel = Stack(
            width=stackLabelList[0].getWidth(),
            height=stackLabelList[0].getHeight(),
            isSegmented=stackLabelList[0].getIsSegmented(),
            isSliceListSupplied=True,
            selectedFiles=[stacked_labels[..., i] for i in range(stacked_labels.shape[-1])],
            channel=stackLabelList[0].getChannel()
        )
        self.numClass = len(class_mapping)
        self.stackImage.setNumClass(self.numClass)
        self.classFrequency = self._calculate_class_weights()
        for img_stack, label_stack in zip(stackImageList, stackLabelList):
            img_stack.setNumClass(len(class_mapping))
            label_stack.setNumClass(len(class_mapping))
    def _select_slices(self, stack, numSliceSampledPerStack):
        if numSliceSampledPerStack and numSliceSampledPerStack <= len(stack.getListSlice()):
            indices = random.sample(range(stack.getStackSize()), numSliceSampledPerStack)
            return stack.getSliceFromPosition(indices)
        return stack.getListSlice()
    def _validate_image_types(self, stackImageList):
        """Ensures all stacks have the same image type."""
        first_image_type = stackImageList[0].getImageType()
        for stack in stackImageList:
            if stack.getImageType() != first_image_type:
                raise ValueError(f"{repr(self)} - Image types must match across stacks")
        self.imageType = first_image_type
    def _create_class_mapping(self, stacked_sequence):
        unique_classes = np.unique(stacked_sequence)
        return {cls: idx for idx, cls in enumerate(unique_classes)}
    def _normalize_class_labels(self, stacked_sequence, class_mapping):
        return np.vectorize(class_mapping.get)(stacked_sequence)
    def _calculate_class_weights(self):
        unique_classes, class_counts = np.unique(self.stackLabel.getListSlice(), return_counts=True)
        class_weights = 1.0 / class_counts
        total_weight = np.sum(class_weights)
        return class_weights / total_weight
# Getters for attributes
    def getNumSlice(self):
        return self.numSlice
    def getXtrain(self):
        return self.Xtrain
    def getYtrain(self):
        return self.Ytrain
    def getXtest(self):
        return self.Xtest
    def getYtest(self):
        return self.Ytest
    def getNumClass(self):
        return self.numClass
    def getStackImage(self):
        return self.stackImage
    def getImageType(self):
        return self.imageType
    def getStackLabel(self):
        return self.stackLabel
    def getValidationSplit(self):
        return self.validation_split
    def getImagePreprocessingFunctions(self):
        return self.image_preprocessing_functions
    def getClassFrequency(self):
        return self.classFrequency
    def getUniqueClass(self):
        return self.uniqueClass
# Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to dataset
def apply_CLAHE(self, dataset):
   """
   Apply CLAHE to a dataset of images.
   Args:
       dataset (list of ndarray): List of input images.
   Returns:
       ndarray: CLAHE-enhanced images.
   """
   preview = True  # Display images for testing purposes
   output = []  # Store processed images
   for img in dataset:
       img_normalized = (img.astype(np.float32) - np.min(img)) / (np.max(img) - np.min(img)) * 2.0 - 1.0
       matched = equalize_adapthist(img_normalized)
       output.append(matched)
       if preview:  # Display original and processed image
           fig, axes = plt.subplots(1, 2)
           axes[0].imshow(img, cmap='gray')
           axes[0].set_title('Original', fontsize=10)
           axes[1].imshow(matched, cmap='gray')
           axes[1].set_title('CLAHE adjusted', fontsize=10)
           plt.show()
           preview = False
   return np.array(output)

# Apply logarithmic correction to a dataset
def apply_log_correction(self, dataset):
   """
   Apply logarithmic adjustment to a dataset of images.
   Args:
       dataset (list of ndarray): List of input images.
   Returns:
       ndarray: Log-adjusted images.
   """
   preview = True
   output = []
   for img in dataset:
       matched = adjust_log(img, 3)
       output.append(matched)
       if preview:
           fig, axes = plt.subplots(1, 2)
           axes[0].imshow(img, cmap='gray')
           axes[0].set_title('Original', fontsize=10)
           axes[1].imshow(matched, cmap='gray')
           axes[1].set_title('Log adjusted', fontsize=10)
           plt.show()
           preview = False
   return np.array(output)

# Apply gamma correction to a dataset
def apply_gamma_correction(self, dataset):
   """
   Apply gamma adjustment to a dataset of images.
   Args:
       dataset (list of ndarray): List of input images.
   Returns:
       ndarray: Gamma-adjusted images.
   """
   preview = True
   output = []
   for img in dataset:
       matched = adjust_gamma(img, 1)
       output.append(matched)
       if preview:
           fig, axes = plt.subplots(1, 2)
           axes[0].imshow(img, cmap='gray')
           axes[0].set_title('Original', fontsize=10)
           axes[1].imshow(matched, cmap='gray')
           axes[1].set_title('Gamma adjusted', fontsize=10)
           plt.show()
           preview = False
   return np.array(output)

# Normalize histograms based on a reference image
def normalize_histograms(self, dataset):
   """
   Normalize the histograms of images in a dataset using the best reference image.
   Args:
       dataset (list of ndarray): List of input images.
   Returns:
       ndarray: Histogram-normalized images.
   """
   def select_best_reference(dataset):
       """
       Select the best representative image from the dataset based on histogram similarity.
       Args:
           dataset (list of ndarray): List of input images.
       Returns:
           ndarray: The best representative image.
       """
       overall_histogram = np.zeros(256, dtype=np.float64)
       for img in dataset:
           overall_histogram += np.histogram(img, bins=256)[0]
       overall_histogram /= overall_histogram.sum()
       representativeness_scores = []
       for img in dataset:
           img_histogram = np.histogram(img, bins=256)[0]
           img_histogram /= img_histogram.sum()
           similarity_score = np.sum(np.minimum(overall_histogram, img_histogram))
           representativeness_scores.append(similarity_score)
       best_index = np.argmax(representativeness_scores)
       return dataset[best_index]
   best_reference = select_best_reference(dataset)
   preview = True
   normalized_images = []
   for img in dataset:
       if img.ndim == 3:  # Multi-channel image (color)
           matched_channels = [match_histograms(img[:, :, ch], best_reference[:, :, ch]) for ch in range(img.shape[2])]
           matched = np.stack(matched_channels, axis=2)
       else:
           matched = match_histograms(img, best_reference)
       normalized_images.append(matched)
       if preview:
           fig, axes = plt.subplots(1, 2)
           axes[0].imshow(img, cmap='gray')
           axes[0].set_title('Original', fontsize=10)
           axes[1].imshow(matched, cmap='gray')
           axes[1].set_title('Normalized', fontsize=10)
           plt.show()
           preview = False
   return np.array(normalized_images)

# Contrast stretching for dataset
def contrast_stretching(self, dataset):
   """
   Apply contrast stretching to a dataset of images.
   Args:
       dataset (list of ndarray): List of input images.
   Returns:
       ndarray: Contrast-stretched images.
   """
   preview = True
   for i, img in enumerate(dataset):
       p2, p98 = np.percentile(img, (5, 95))
       matched = rescale_intensity(img, in_range=(p2, p98))
       dataset[i] = matched
       if preview:
           fig, axes = plt.subplots(1, 2)
           axes[0].imshow(img, cmap='gray')
           axes[0].set_title('Original', fontsize=10)
           axes[1].imshow(matched, cmap='gray')
           axes[1].set_title('Contrast stretched', fontsize=10)
           plt.show()
           preview = False
   return dataset
