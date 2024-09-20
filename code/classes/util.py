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
#from tensorflow.keras.utils import to_categorical
from .ProgressBar import ProgressBar
from math import sqrt
import cv2
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.exposure import match_histograms, equalize_adapthist, rescale_intensity, equalize_hist, adjust_gamma, adjust_log, histogram
from sklearn.preprocessing import OneHotEncoder

class Util:
      
    def __repr__(self):
        return 'Util'

    def __init__(self, **kwargs):
        self.Xtrain = self.Ytrain = self.Xtest = self.Ytest = np.zeros([0,0,0,0])
        
        if ("name" in kwargs):
            self.name = kwargs['name']
        else:
            self.name = ""

        if ("numSlice" in kwargs):
            self.numSlice = kwargs['numSlice']
        else:
            self.numSlice = None   
        
        if ("image_preprocessing_functions" in kwargs):
            self.image_preprocessing_functions = kwargs['image_preprocessing_functions']
        else:
            self.image_preprocessing_functions = None   
               
        if ("scaler" in kwargs):
            self.scaler = kwargs['scaler']
        
        if ("validation_split" in kwargs):
            self.validation_split = kwargs['validation_split']
        else:
            self.validation_split = 0.3
        
        if ("isInference" in kwargs):
            self.inference = kwargs['isInference']
        else:
            self.inference = False
        
        #If only one image stack (inference case) is passed 
        if ("stackImage" in kwargs) and ("stackLabel" not in kwargs): 
              stackImageList = kwargs['stackImage']
              self.imageType = stackImageList[0].getImageType()

              if self.numSlice is not None:
                  #An equal number of slice per stack will be selected
                  numSliceSampledPerStack = int(self.numSlice/len(stackImageList))
                  numSliceProvided = True                   
              else:
                  numSliceProvided = False
              
              self.numClass = stackImageList[0].getNumClass()

              #numSlice = 0
                  
              selectedSliceImage = []
              #For each stack in the list of stack provided
              pb = ProgressBar(len(stackImageList), txt=repr(self) +'- Loading '+ self.name)

              for stackImageTmp in stackImageList:
                  if numSliceProvided: 
                      #Check if the number of slices to be randomly selected is less or equal to 
                      #the amount of slices present in the stack
                      #If the amount of requested image is inferior to 
                      #the total amount of image available in the folder
                      #then a sampling can be done
                      if numSliceSampledPerStack <= len(stackImageTmp.getListSlice()):
                          indexSlice = random.sample(range(0, stackImageTmp.getStackSize()), numSliceSampledPerStack)
                          selectedSliceImage = selectedSliceImage + (stackImageTmp.getSliceFromPosition(indexSlice))
                          #numSlice += numSliceSampledPerStack
                      else:
                      #If there are less images in the folder than requested samples
                      #then we take all of the available images in the folder
                          selectedSliceImage = selectedSliceImage + (stackImageTmp.getListSlice())
                          #numSlice += len(selectedSliceImage)
                  else:
                      selectedSliceImage = selectedSliceImage + (stackImageTmp.getListSlice())
                      #numSlice += len(selectedSliceImage)

                  pb += 1

                  self.stackImage = Stack(imageType = self.imageType,
                                  width = stackImageList[0].getWidth(),
                                  height = stackImageList[0].getHeight(),
                                  isSegmented = stackImageList[0].getIsSegmented(), 
                                  #maskPreprocessing = stackImageList[0].getMaskPreprocessing(),
                                  #isMulticlass = stackImageList[0].getIsMulticlass(),
                                  isSliceListSupplied = True,
                                  selectedFiles = selectedSliceImage,
                                  channel=stackImageList[0].getChannel(),
                                  numClass = self.numClass)

              self.numSlice = self.stackImage.getStackSize()
        #If one or more pair of images/labels are provided
        elif ("stackImage" in kwargs) and ("stackLabel" in kwargs):
            
            def uniqueClass(unique_classes):
                class_mapping ={}
                # Map unique classes to a sequential range
                for idx, cls in enumerate(unique_classes):
                    class_mapping[cls] = idx
                return class_mapping
            
            def normalizeClass(stacked_sequence, class_mapping):
                # Normalize the class labels in the entire sequence based on the mapping
                for i in range(stacked_sequence.shape[-1]):
                    stacked_sequence[..., i] = np.vectorize(class_mapping.get)(stacked_sequence[..., i])
                return stacked_sequence

            stackImageList = kwargs['stackImage']
            stackLabelList = kwargs['stackLabel']

            # Check if all objects belong to the same class
            first_image_type = stackImageList[0].getImageType()
            for stack in stackImageList:
                if stack.getImageType() != first_image_type:
                    raise Exception(repr(self) + ' class - Image types must be the same between stacks')
                else:
                    self.imageType = first_image_type

            if self.numSlice is not None:
                #An equal number of slice per stack will be selected
                numSliceSampledPerStack = int(self.numSlice/len(stackImageList))
                numSliceProvided = True                   
            else:
                numSliceProvided = False
            
            if stackImageList[0].getNumClass() == stackLabelList[0].getNumClass():
                self.numClass = stackImageList[0].getNumClass()
                
            selectedSliceImage = []
            selectedSliceLabel = []
            #For each stack in the list of stack provided
            pb = ProgressBar(len(stackImageList)+4, txt=repr(self) +'- Loading '+ self.name)
            for stackImageTmp, stackLabelTmp in zip(stackImageList, stackLabelList):
                if numSliceProvided: 
                    #Check if the number of slices to be randomly selected is less or equal to 
                    #the amount of slices present in the stack
                    #If the amount of requested image is inferior to 
                    #the total amount of image available in the folder
                    #then a sampling can be done
                    if numSliceSampledPerStack <= len(stackImageTmp.getListSlice()):
                        indexSlice = random.sample(range(0, stackImageTmp.getStackSize()), numSliceSampledPerStack)
                        selectedSliceImage = selectedSliceImage + (stackImageTmp.getSliceFromPosition(indexSlice))
                        selectedSliceLabel = selectedSliceLabel + (stackLabelTmp.getSliceFromPosition(indexSlice))
                    else:
                    #If there are less images in the folder than requested samples
                    #then we take all of the available images in the folder
                        selectedSliceImage = selectedSliceImage + (stackImageTmp.getListSlice())
                        selectedSliceLabel = selectedSliceLabel + (stackLabelTmp.getListSlice())
                else:
                    selectedSliceImage = selectedSliceImage + (stackImageTmp.getListSlice())
                    selectedSliceLabel = selectedSliceLabel + (stackLabelTmp.getListSlice())
                pb += 1

            self.stackImage = Stack(width = stackImageList[0].getWidth(),
                            height = stackImageList[0].getHeight(),
                            isSegmented = stackImageList[0].getIsSegmented(), 
                            isSliceListSupplied = True,
                            selectedFiles = selectedSliceImage,
                            channel=stackImageList[0].getChannel())

            
            #Class count
            stacked_sequence = np.stack(selectedSliceLabel, axis=-1)
            pb += 1
            unique_classes = np.unique(stacked_sequence)
            pb += 1
            class_mapping = uniqueClass(unique_classes)
            pb += 1
            stacked_sequence = normalizeClass(stacked_sequence, class_mapping)
            
            #retreive unique normalized class from the class_mapping dict
            self.uniqueClass = []
            keys = list(class_mapping.keys())
            keys.sort()
            for key in keys:
                self.uniqueClass.append(class_mapping[key])
            
            pb += 1
            listMask = []
            for i in range(stacked_sequence.shape[-1]):
                # Extract a single channel image
                image = stacked_sequence[..., i]  
                listMask.append(image)
            
            self.stackLabel = Stack(width = stackLabelList[0].getWidth(),
                            height = stackLabelList[0].getHeight(),
                            isSegmented = stackLabelList[0].getIsSegmented(), 
                            isSliceListSupplied = True,
                            selectedFiles = listMask,
                            channel = stackLabelList[0].getChannel())

            self.numClass = len(class_mapping)
            print(self.numClass)
            self.stackImage.setNumClass(self.numClass)
            self.numSlice = self.stackImage.getStackSize()
            self.classFrequency = self.weights_calc()
            listSize = len(kwargs['stackImage'])
            for i in range(0,listSize):
                kwargs["stackImage"][i].setNumClass(len(class_mapping))
                kwargs['stackLabel'][i].setNumClass(len(class_mapping))
        else:
            raise Exception(repr(self) + ' class - Only list of stacks are accepted as inputs')
    
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
 
    def weights_calc(self):
        """
        Calculate class weights based on label frequencies.
        
        Args:
        labels (list or numpy array): List of class labels.
        
        Returns:
        class_weights (numpy array): Array of class weights normalized to sum to one.
        """
        # Calculate class frequencies
        unique_classes, class_counts = np.unique(self.stackLabel.getListSlice(), return_counts=True)
        
        # Sort classes by frequency in ascending order
        sorted_classes = unique_classes[np.argsort(class_counts)]
        
        # Calculate weights inversely proportional to frequencies
        class_weights = 1.0 / class_counts
        
        # Create a dictionary to store class weights
        class_weight_dict = {class_label: weight for class_label, weight in zip(sorted_classes, class_weights)}
        
        # Normalize the weights so that they sum to one
        total_weight = sum(class_weight_dict.values())
        class_weights = [weight / total_weight for weight in class_weights]
        
        return class_weights


    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to dataset
    def CLAHE(self, dataset):
        print("CLAHE") # Should be removed after deployment
        preview=True  # A boolean variable to control display of test images
        output = [] # Create an empty list to store the processed cubes
        for img in dataset:  # Loop through cubes in dataset
            img_normalized = (img.astype(np.float32) - np.min(img)) / (np.max(img) - np.min(img)) * 2.0 - 1.0
            matched = equalize_adapthist(img_normalized)  # Apply CLAHE to the normalized image
            output.append(matched)  # Append CLAHE-enhanced cube to output
            if preview:  # Display original and CLAHE-enhanced images for the first slice in the first cube
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Original', fontsize=10)
                axes[1].imshow(matched, cmap='gray')
                axes[1].set_title('CLAHE adjusted', fontsize=10)
                plt.show()
                preview = False
        return np.array(output)

    # Function to perform histogram equalization on a dataset
    def adjustlog(self, dataset):
        preview = True  # A boolean variable to control display of test images
        print("adjust log") # Should be removed after deployment
        output = []  # Create an empty list to store the equalized cubes
        for img in dataset:  # Iterate over each cube in the input dataset
            matched = adjust_log(img, 3)  # Apply histogram equalization using equalize_hist() function
            output.append(matched)  # Append the equalized cube to the output list
            if preview:  # If test is True, display a comparison of original and equalized cube images
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Original', fontsize=10)
                axes[1].imshow(matched, cmap='gray')
                axes[1].set_title('Log adjusted', fontsize=10)
                plt.show()
                preview = False  # Set test to False to only display the comparison for the first cube
        return np.array(output)  # Convert the output list to a NumPy array and return it

    # Function to perform histogram equalization on a dataset of 3D cubes
    def adjustgamma(self, dataset):
        preview = True  # A boolean variable to control display of test images
        print("adjust gamma") # Should be removed after deployment
        output = []  # Create an empty list to store the equalized cubes
        for img in dataset:  # Iterate over each cube in the input dataset
            matched = adjust_gamma(img, 1)  # Apply histogram equalization using equalize_hist() function
            output.append(matched)  # Append the equalized cube to the output list
            if preview:  # If test is True, display a comparison of original and equalized cube images
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Original', fontsize=10)
                axes[1].imshow(matched, cmap='gray')
                axes[1].set_title('Gamma adjusted', fontsize=10)
                plt.show()
                preview = False  # Set test to False to only display the comparison for the first cube
        return np.array(output)  # Convert the output list to a NumPy array and return it
    '''
    # Function to perform histogram equalization on a dataset of 3D cubes
    def equalize_histogram(self, dataset):
        preview = True  # A boolean variable to control display of test images
        print("equalize_histogram") # Should be removed after deployment
        output = []  # Create an empty list to store the equalized cubes
        for img in dataset:  # Iterate over each cube in the input dataset
            matched = equalize_hist(img)  # Apply histogram equalization using equalize_hist() function
            output.append(matched)  # Append the equalized cube to the output list
            if preview:  # If test is True, display a comparison of original and equalized cube images
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(img)
                axes[1].imshow(matched)
                plt.show()
                preview = False  # Set test to False to only display the comparison for the first cube
        return np.array(output)  # Convert the output list to a NumPy array and return it
    '''
    def normalize_histograms(self, dataset):
        """
        Normalize the histograms of a dataset using the best representative image.
    
        Args:
            dataset (list of ndarray): A list of input images.
    
        Returns:
            list of ndarray: The normalized images.
        """
        print("normalize_histograms") # Should be removed after deployment

        def select_best_reference(dataset):
            """
            Select the best representative image from a dataset.
        
            Args:
                dataset (list of ndarray): A list of input images.
        
            Returns:
                ndarray: The best representative image.
            """
            # Calculate the overall histogram for the entire dataset
            overall_histogram = np.zeros(256, dtype=np.float64)  # Change dtype to float64
            for img in dataset:
                overall_histogram += histogram(img, nbins=256)[0]
        
            # Normalize the overall histogram
            overall_histogram = overall_histogram / overall_histogram.sum()
        
            # Compute representativeness scores for each image
            representativeness_scores = []
            for img in dataset:
                img_histogram = histogram(img, nbins=256)[0]
                img_histogram = img_histogram / img_histogram.sum()
                similarity_score = np.sum(np.minimum(overall_histogram, img_histogram))
                representativeness_scores.append(similarity_score)
        
            # Find the index of the image with the highest representativeness score
            best_index = np.argmax(representativeness_scores)
        
            # Return the best representative image
            return dataset[best_index]

        # Select the best representative image
        best_reference = select_best_reference(dataset)
        
        preview = True  # A boolean variable to control display of test images
        normalized_images = []
        for img in dataset:
            # Perform histogram matching separately for each channel (if multichannel)
            if img.ndim == 3:  # Check if the image is multichannel (color)
                matched_channels = []
                for channel in range(img.shape[2]):
                    matched = match_histograms(img[:, :, channel], best_reference[:, :, channel])
                    matched_channels.append(matched)
                matched = np.stack(matched_channels, axis=2)
            else:
                # For single-channel images, perform histogram matching directly
                matched = match_histograms(img, best_reference)
    
            normalized_images.append(matched)
    
            if preview:
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Original', fontsize=10)
                axes[0].axis('off')
                axes[1].imshow(matched, cmap='gray')
                axes[1].set_title('Normalized', fontsize=10)
                axes[1].axis('off')
                plt.show()
                preview = False
        return np.array(normalized_images)

    def contrast_stretching(self, dataset):
            preview=True # A boolean variable to control display of test images
            print("contrast_stretching") # Should be removed after deployment
            for i, img in enumerate(dataset):
                p2, p98 = np.percentile(img, (5, 95))
                matched = rescale_intensity(img, in_range=(p2, p98))
                dataset[i] = matched
                if preview:  # Display original and contrast-stretched images for the first slice in the first batch
                      fig, axes = plt.subplots(1, 2)
                      axes[0].imshow(img, cmap='gray')
                      axes[0].set_title('Original', fontsize=10)
                      axes[1].imshow(matched, cmap='gray')
                      axes[1].set_title('Contrast stretched', fontsize=10)
                      plt.show()
                      preview = False
            return dataset
            
    def loadData(self):
        
        if hasattr(self, 'stackLabel'): #Training case
            X = np.zeros([self.stackImage.getStackSize(),self.stackImage.getHeight(),self.stackImage.getWidth(),self.stackImage.getChannel()])
            y = np.zeros([self.stackLabel.getStackSize(),self.stackLabel.getHeight(),self.stackLabel.getWidth(),self.stackLabel.getChannel()])

            for i in range(self.stackImage.getStackSize()):
                X[i,:,:,:] = self.stackImage.getListSlice()[i]
                y[i,:,:,:] = self.stackLabel.getListSlice()[i]
                
            #image treatment
            # Call the functions using their names from the array
            if self.image_preprocessing_functions:
                for func_name in self.image_preprocessing_functions:
                    func = getattr(self, func_name, None)
                    if func is not None and callable(func):
                        X = func(X)

            if self.numClass > 2: #one hot encode multiclass dataset
                encoder = OneHotEncoder(sparse_output=False)
                Y_encoded = encoder.fit_transform(y.reshape(-1, 1))
                y = Y_encoded.reshape((self.numSlice, self.stackLabel.getHeight(), self.stackLabel.getWidth(), self.numClass))
                del encoder, Y_encoded
            else: #clipping binary data between 0 and 1
                y = np.stack([np.minimum(np.maximum(arr, 0), 1) for arr in y], axis=0).astype(np.float32)

            if self.inference:
                Xtrain_transformed = (self.scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape))  # Apply the scaler to the training image data
                self.Xtest = None
                self.Xtrain = Xtrain_transformed
                self.Ytest = None
                self.Ytrain = y
                del X, y
            else:
                Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=self.validation_split)
                del X, y

                Xtrain_transformed = np.zeros_like(Xtrain)
                Xtest_transformed = np.zeros_like(Xtest)

                scaler = MinMaxScaler(feature_range = (0, 1), clip=True)  # Initialize a MinMaxScaler to scale the image data
                Xtrain_transformed = (scaler.fit_transform(Xtrain.reshape(-1, 1)).reshape(Xtrain.shape))  # Apply the scaler to the training image data
                Xtest_transformed = (scaler.transform(Xtest.reshape(-1,1)).reshape(Xtest.shape))  # Apply the scaler to the testing image data
                self.scaler = scaler

                self.Xtest = Xtest_transformed
                self.Xtrain = Xtrain_transformed
                self.Ytest = Ytest
                self.Ytrain = Ytrain

            return self.Xtrain, self.Ytrain, self.Xtest, self.Ytest

        else: #Inference case

            X = np.zeros([self.stackImage.getStackSize(),self.stackImage.getHeight(),self.stackImage.getWidth(),self.stackImage.getChannel()])

            for i in range(self.stackImage.getStackSize()):
                X[i,:,:,:] = self.stackImage.getListSlice()[i]
                
            #image treatment
            # Call the functions using their names from the array
            if self.image_preprocessing_functions:
                for func_name in self.image_preprocessing_functions:
                    func = getattr(self, func_name, None)
                    if func is not None and callable(func):
                        X = func(X)

            X_transformed = np.zeros_like(X)
            
            #scaler = MinMaxScaler(feature_range = (0, 1), clip=True)  # Initialize a MinMaxScaler to scale the image data
            X_transformed = (self.scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape))  # Apply the scaler to the training image data
          
            self.Xtrain = X_transformed
            self.Xtest = None
            self.Ytest = None
            self.Ytrain = None

            return self.Xtrain, self.Ytrain, self.Xtest, self.Ytest

'''
        # Create a tf.data.Options object to specify options for the datasets
        options = tf.data.Options()
        # Set auto_shard_policy option to OFF, indicating to turn off auto sharding for distributed training
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
     
        # Apply the options specified in the options object to the training dataset
        training_dataset = training_dataset.with_options(options)
        # Apply the options specified in the options object to the test dataset
        test_dataset = test_dataset.with_options(options)
     
        return training_dataset, test_dataset
    
        def normalize_image_sequence(input_folder, output_folder):
            def count_eligible_images(input_folder):
                count = 0

                # Iterate through all image files in the input folder
                for filename in os.listdir(input_folder):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')):
                        count += 1

                return int(count)
            
            tot_file = count_eligible_images(input_folder)
            print(tot_file)
            pb = ProgressBar(tot_file,txt='Loading')

            # Create output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            class_mapping = {}  # Mapping dictionary for class normalization

            # List to store all images in the sequence
            image_sequence = []

            # Iterate through all image files in the input folder
            for filename in os.listdir(input_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')):
                    image_path = os.path.join(input_folder, filename)
                    
                    # Use tifffile to read TIFF files
                    if image_path.lower().endswith(('.tif', '.tiff')):
                        image = tiff.imread(image_path)
                        # Transpose TIFF images to (width, height) format if necessary
                        if image.shape[0] < image.shape[1]:
                            image = np.transpose(image, (1, 0))
                    else:
                        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

                    # Check if the image is grayscale (1 channel)
                    if image.ndim == 2:
                        # Image is already grayscale
                        pass
                    else:
                        # Convert multi-channel image to grayscale
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    image_sequence.append(image)
                    pb+=1

            # Stack all images in the sequence along a new axis
            stacked_sequence = np.stack(image_sequence, axis=-1)

            # Calculate unique class values across the entire sequence
            unique_classes = np.unique(stacked_sequence)
            tot_file = count_eligible_images(input_folder)
            

            pb2 = ProgressBar(tot_file,txt='- Loading')
            # Map unique classes to a sequential range
            for idx, cls in enumerate(unique_classes):
                class_mapping[cls] = idx
                pb2+=1
                
            pb3 = ProgressBar((stacked_sequence.shape[-1]), txt='Loading')
            
            # Normalize the class labels in the entire sequence based on the mapping
            for i in range(stacked_sequence.shape[-1]):
                stacked_sequence[..., i] = np.vectorize(class_mapping.get)(stacked_sequence[..., i])
                pb3+=1

            pb4 = ProgressBar((stacked_sequence.shape[-1]),txt='- Loading')
            
            # Save the modified sequence as individual images in the output folder
            for i in range(stacked_sequence.shape[-1]):
                output_path = os.path.join(output_folder, f"image_{i}.png")  # Output as PNG
                image = stacked_sequence[..., i]  # Extract a single channel image
                cv2.imwrite(output_path, image)
                #print(f"Saved image {i} to {output_path}")
                pb4+=1

            return class_mapping

        # Example usage:
        input_folder = "C:\\Users\\florent.brondolo\\Documents\\SCHISM\\Data\\bentheimer\\testMasks"
        output_folder = "C:\\Users\\florent.brondolo\\Documents\\SCHISM\\Data\\bentheimer\\test"
        class_mapping = normalize_image_sequence(input_folder, output_folder)

        # Display class mapping
        print("Class Mapping:")
        for original_cls, mapped_cls in class_mapping.items():
            print(f"Original Class {original_cls} -> Normalized Class {mapped_cls}")
            
    def cleanMask(self, img):    
        height=self.stackLabel.getHeight()
        width=self.stackLabel.getWidth()
        channel=self.stackLabel.getChannel()
        
        COLORS = (
            #R
            (1, 0, 0),
            #G
            (0, 1, 0),
            #B
            (0, 0, 1),
            #k
            (0, 0, 0),
        )
        
        def closest_color(rgb):
            r, g, b = rgb
            color_diffs = []
            for color in COLORS:
                cr, cg, cb = color
                color_diff = sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
                color_diffs.append((color_diff, color))
            return min(color_diffs)[1]
        
        result = [np.dot(closest_color(rgb),[1,2,3]) for rgb in img.reshape(height*width, channel)]       
        nColours = len(np.unique(result))  
        
        img2=cv2.cvtColor(np.float32(img),cv2.COLOR_BGR2RGB).reshape((-1,3))  
       
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1)
        K=nColours
        attempts=20
        ret,label,center=cv2.kmeans(img2,K,None,criteria,attempts,flags=cv2.KMEANS_RANDOM_CENTERS)
        label = label.flatten()
        res = center[label]
        result_image = res.reshape((img.shape))
        
        result = np.zeros([height, width, channel],dtype=np.float32)
        
        for c in center:
            # select color and create mask
            layer = result_image.copy()
            mask = cv2.inRange(layer, c, c)
            # apply mask to layer 
            layer[mask == 0] = [0,0,0]
            result+=layer     
        #print("-- cleaning -- ")
        return result
'''  