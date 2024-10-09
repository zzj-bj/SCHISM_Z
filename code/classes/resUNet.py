# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:45:14 2022

@author: florent.brondolo
"""
import sys
from tensorflow.python.client import device_lib
import tensorflow as tf
import keras
from tensorflow.python.keras.layers import add, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Activation
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import Model, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from datetime import datetime
from .util import Util
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import CategoricalFocalCrossentropy
import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
import segmentation_models as sm


class ResUNet:
    def __repr__(self):
        return 'ResUNet'

    def __init__(self, **kwargs):
        """_summary_

        Raises:
            Exception: _description_
            Exception: _description_
        """
        if kwargs['pathLogDir'] is not None:
            self.pathLogDir = kwargs['pathLogDir']
        else:
            raise Exception(
                repr(self) + ' class - pathLogDir variable must me provided')

        self.fileName = "run-" + datetime.now().strftime("%d-%m-%Y--%Hh-%Mm-%Ss")
        self.logdir = os.path.join(self.pathLogDir, self.fileName)

        self.FILE_TXT = '-----------------------------------'+'\r'
        self.FILE_TXT = self.FILE_TXT + 'runName = ' + self.fileName + '\r'
        self.FILE_TXT = self.FILE_TXT + '-----------------------------------'+'\r'

        if "featuremaps" in kwargs:
            self.featuremaps = int(kwargs['featuremaps'])
        else:
            self.featuremaps = 8
        self.FILE_TXT = self.FILE_TXT + \
            "\nfeaturemaps = " + str(self.featuremaps)

        if "data" in kwargs:
            if isinstance(kwargs['data'], (Util)):
                self.util = kwargs['data']
                self.Xtrain = self.util.getXtrain()
                self.Ytrain = self.util.getYtrain()
                self.Xtest = self.util.getXtest()
                self.Ytest = self.util.getYtest()
                self.numClass = self.util.getNumClass()
                self.FILE_TXT = self.FILE_TXT + \
                    "\nnumClass = " + str(self.numClass)
                self.FILE_TXT = self.FILE_TXT + \
                    "\nnumSample = " + str(self.util.getNumSlice())
                self.FILE_TXT = self.FILE_TXT + "\nval_split = " + \
                    str(self.util.getValidationSplit())
                self.FILE_TXT = self.FILE_TXT + \
                    "\nimage_preprocessing_functions = ["
                if len(self.util.getImagePreprocessingFunctions()) > 0:
                    marker = len(
                        self.util.getImagePreprocessingFunctions()) - 1
                    i = 0
                    for function in self.util.getImagePreprocessingFunctions():
                        if len(self.util.getImagePreprocessingFunctions()) == 1:
                            self.FILE_TXT = self.FILE_TXT + function
                        elif len(self.util.getImagePreprocessingFunctions()) > 1:
                            if marker == i:
                                self.FILE_TXT = self.FILE_TXT + function
                            else:
                                self.FILE_TXT = self.FILE_TXT + function + ","
                        i += 1
                self.FILE_TXT = self.FILE_TXT + "]"
                self.FILE_TXT = self.FILE_TXT + "\nimageType = " + \
                    str(self.util.getImageType())
            else:
                raise Exception(
                    repr(self) + ' class - training and testing dataset must be provided')

        if "epochs" in kwargs:
            self.epochs = int(kwargs['epochs'])
        else:
            self.epochs = 50
        self.FILE_TXT = self.FILE_TXT + "\nepochs = " + str(self.epochs)

        if "batch_size" in kwargs:
            self.batch_size = int(kwargs['batch_size'])
        else:
            self.batch_size = 8
        self.FILE_TXT = self.FILE_TXT + \
            "\nbatch_size = " + str(self.batch_size)

        if "learningRate" in kwargs:
            self.learningRate = float(kwargs['learningRate'])
        else:
            self.learningRate = 1e-5
        self.FILE_TXT = self.FILE_TXT + \
            "\nlearningRate = " + str(self.learningRate)

        if "L2" in kwargs:
            self.L2 = float(kwargs['L2'])
        else:
            self.L2 = 1e-5
        self.FILE_TXT = self.FILE_TXT + "\nL2 = " + str(self.L2)

        if "batchNorm" in kwargs:
            self.batchNorm = bool(kwargs['batchNorm'])
            self.FILE_TXT = self.FILE_TXT + \
                "\nbatchNorm = " + str(self.batchNorm)
        else:
            self.batchNorm = False
            self.FILE_TXT = self.FILE_TXT + "\nbatchNorm = none"

        if "maxNorm" in kwargs:
            self.maxNorm = int(kwargs['maxNorm'])
        else:
            self.maxNorm = 1
        self.FILE_TXT = self.FILE_TXT + "\nMaxNorm = " + str(self.maxNorm)

        if "dropOut" in kwargs:
            self.dropOut = bool(kwargs['dropOut'])
            self.FILE_TXT = self.FILE_TXT + "\ndropOut = " + str(self.dropOut)
        else:
            self.dropOut = False
            self.FILE_TXT = self.FILE_TXT + "\ndropOut = none"

        if self.dropOut:
            if "dropoutRate" in kwargs:
                self.dropoutRate = float(kwargs['dropoutRate'])
            else:
                self.dropoutRate = 0.1
            self.FILE_TXT = self.FILE_TXT + \
                "\ndropoutRate = " + str(self.dropoutRate)
        else:
            self.dropoutRate = None
            self.FILE_TXT = self.FILE_TXT + "\ndropoutRate = none"

        if "dataGeneration" in kwargs:
            self.dataGeneration = bool(kwargs['dataGeneration'])
        else:
            self.dataGeneration = False
        self.FILE_TXT = self.FILE_TXT + \
            "\ndataGeneration = " + str(self.dataGeneration)

        if "additional_augmentation_factor" in kwargs:
            self.additional_augmentation_factor = kwargs['additional_augmentation_factor']
        else:
            self.additional_augmentation_factor = 2
        self.FILE_TXT = self.FILE_TXT + "\nadditional_augmentation_factor = " + \
            str(self.additional_augmentation_factor)

        if "patience" in kwargs:
            self.patience = int(kwargs['patience'])
        else:
            self.patience = 5
        self.FILE_TXT = self.FILE_TXT + "\npatience = " + str(self.patience)

        if "padding" in kwargs:
            if kwargs['padding'] == "same" or kwargs['padding'] == "valid":
                self.padding = kwargs['padding']
            else:
                self.padding = "same"
        else:
            self.padding = "same"
        self.FILE_TXT = self.FILE_TXT + "\npadding = " + str(self.padding)

        if "pretrained" in kwargs:
            self.pretrained = bool(kwargs['pretrained'])
            if "backbone" in kwargs:
                self.backbone = str(kwargs['backbone'])
                self.FILE_TXT = self.FILE_TXT + \
                    "\nbackbone= " + str(self.backbone)
        else:
            self.pretrained = False
        self.FILE_TXT = self.FILE_TXT + \
            "\npretrained = " + str(self.pretrained)

        if "img_height" in kwargs:
            self.img_height = int(kwargs['img_height'])
        else:
            self.img_height = self.Xtrain.shape[1]
        self.FILE_TXT = self.FILE_TXT + \
            "\nimg_height = " + str(self.img_height)

        if "img_width" in kwargs:
            self.img_width = int(kwargs['img_width'])
        else:
            self.img_width = self.Xtrain.shape[2]
        self.FILE_TXT = self.FILE_TXT + "\nimg_width = " + str(self.img_width)

        if "metrics" in kwargs:
            if "early_stopping" in kwargs:
                metrics_early_stopping = kwargs['early_stopping']
                metrics_early_stopping_tmp = []

            self.FILE_TXT = self.FILE_TXT + "\nmetrics = ["
            self.metrics = kwargs['metrics']
            if len(self.metrics) > 0:
                marker = len(kwargs['metrics']) - 1
                i = 0
                metrics = []
                for metric_name in self.metrics:
                    if len(kwargs['metrics']) == 1:
                        self.FILE_TXT = self.FILE_TXT + str(metric_name)
                    elif len(kwargs['metrics']) > 1:
                        if marker == i:
                            self.FILE_TXT = self.FILE_TXT + str(metric_name)
                        else:
                            self.FILE_TXT = self.FILE_TXT + \
                                str(metric_name) + ","
                    if metric_name == 'OneHotIoU' or metric_name == 'OneHotMeanIoU':
                        # Pass num_classes for OneHotIoU and OneHotMeanIoU metrics
                        if metric_name == 'OneHotIoU':
                            target_class_ids = np.array(
                                self.util.getUniqueClass(), dtype=np.int32)  # Convert to int32
                            metric_instance = globals()[metric_name](
                                num_classes=self.numClass, target_class_ids=target_class_ids)
                        else:
                            metric_instance = globals()[metric_name](
                                num_classes=self.numClass)
                    else:
                        # For other metrics, create instances without num_classes
                        metric_instance = globals()[metric_name]()

                    metrics.append(metric_instance)

                    if "early_stopping" in kwargs:
                        if len(metrics_early_stopping) > 0:
                            if metric_name in metrics_early_stopping:
                                metrics_early_stopping_tmp.append(
                                    'val_'+metric_instance.name)
                    i += 1
                self.metrics = metrics
                if "early_stopping" in kwargs:
                    self.early_stopping = metrics_early_stopping_tmp
            else:
                self.metrics.append('accuracy')
                self.FILE_TXT = self.FILE_TXT + "accuracy"
        else:
            self.metrics.append('accuracy')
            self.FILE_TXT = self.FILE_TXT + "accuracy"
        self.FILE_TXT = self.FILE_TXT + "]"

        if "early_stopping" in kwargs:
            self.FILE_TXT = self.FILE_TXT + "\nearly_stopping = ["
            if len(self.early_stopping) > 0:
                marker = len(self.early_stopping) - 1
                i = 0
                for early_stopping in self.early_stopping:
                    if len(self.early_stopping) == 1:
                        self.FILE_TXT = self.FILE_TXT + str(early_stopping)
                    elif len(self.early_stopping) > 1:
                        if marker == i:
                            self.FILE_TXT = self.FILE_TXT + str(early_stopping)
                        else:
                            self.FILE_TXT = self.FILE_TXT + \
                                str(early_stopping) + ","
                    i += 1
            self.FILE_TXT = self.FILE_TXT + "]"

        if "loss_early_stopping" in kwargs:
            self.loss_early_stopping = kwargs['loss_early_stopping']
            self.FILE_TXT = self.FILE_TXT + "\nloss_early_stopping = " + \
                str(self.loss_early_stopping)

        if "save_model" in kwargs:
            self.save_model = kwargs['save_model']
        else:
            self.save_model = False
        self.FILE_TXT = self.FILE_TXT + \
            "\nsave_model = " + str(self.save_model)

        self.FILE_TXT = self.FILE_TXT + "\n------- Informational inputs -------"

        self.num_gpus = self.getAvailableGPU()
        self.FILE_TXT = self.FILE_TXT + "\nGPU(s) = " + str(self.num_gpus)

        if "loss" in kwargs:
            self.loss = kwargs['loss']
        else:
            if self.numClass == 2:  # Binary
                self.loss = self.weighted_binary_crossentropy()
                # self.loss = BinaryCrossentropy(from_logits=True)
            else:  # Multiclass
                self.loss = CategoricalFocalCrossentropy()
                # self.loss = self.weighted_categorical_crossentropy()
                # self.loss = CategoricalCrossentropy(from_logits=True)
        self.FILE_TXT = self.FILE_TXT + "\nLoss function = " + str(self.loss)

        if "displaySummary" in kwargs:
            self.displaySummary = kwargs['displaySummary']
        else:
            self.displaySummary = True

    def getTrainData(self):
        return self.Xtrain, self.Ytrain

    def getTestData(self):
        return self.Xtest, self.Ytest

    def getAvailableGPU(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def getValidationSplit(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.validation_split

    def dataGenerator(self, seed, batch_size, isTrainSet, forVisualisation=False):
        if isTrainSet:
            '''
            image_datagen = ImageDataGenerator(#featurewise_center=True,
                                               #featurewise_std_normalization=True,
                                               #zca_whitening=True,
                                               rotation_range=360,
                                               width_shift_range=0.5,
                                               height_shift_range=0.5,
                                               horizontal_flip=True,
                                               vertical_flip=True,
                                               shear_range=0.5,
                                               #zoom_range=0.5,
                                               fill_mode='reflect')
            '''
            # TO TEST
            image_datagen = ImageDataGenerator(rotation_range=15,  # Rotate images up to 15 degrees
                                               width_shift_range=0.1,  # Shift width by up to 10%
                                               height_shift_range=0.1,  # Shift height by up to 10%
                                               horizontal_flip=True,  # Horizontal flip
                                               vertical_flip=True,  # Vertical flip
                                               shear_range=0.2,  # Shear up to 20%
                                               fill_mode='reflect'  # Fill mode
                                               )
            image_datagen.fit(self.Xtrain)
            genX1 = image_datagen.flow(
                self.Xtrain, self.Ytrain, batch_size=batch_size, seed=seed, shuffle=isTrainSet)
            image_datagen.fit(self.Ytrain)
            genX2 = image_datagen.flow(
                self.Ytrain, self.Xtrain, batch_size=batch_size, seed=seed, shuffle=isTrainSet)
        else:
            image_datagen = ImageDataGenerator()
            image_datagen.fit(self.Xtrain)
            genX1 = image_datagen.flow(
                self.Xtrain, self.Ytrain, batch_size=batch_size, seed=seed, shuffle=isTrainSet)
            image_datagen.fit(self.Ytrain)
            genX2 = image_datagen.flow(
                self.Ytrain, self.Xtrain, batch_size=batch_size, seed=seed, shuffle=isTrainSet)

        while True:
            if forVisualisation:
                X1i = genX1.next()
                X2i = genX2.next()
                yield [X1i[0], X2i[0]], [X2i[1], X1i[1]]
            else:
                X1i = genX1.next()
                X2i = genX2.next()
                yield X1i[0], X2i[0]

    def getModel(self):
        return self.model

    def weighted_categorical_crossentropy(self):
        weights = K.variable(self.util.getClassFrequency(), dtype='float32')

        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss
        return loss

    # taken from https://github.com/huanglau/Keras-Weighted-Binary-Cross-Entropy/blob/master/DynCrossEntropy.py
    def weighted_binary_crossentropy(self):
        def loss(y_true, y_pred):
            # calculate the binary cross entropy
            bin_crossentropy = K.binary_crossentropy(y_true, y_pred)
            # apply the weights
            weights = y_true * \
                self.util.getClassFrequency(
                )[1] + (1. - y_true) * self.util.getClassFrequency()[0]
            weighted_bin_crossentropy = weights * bin_crossentropy
            return K.mean(weighted_bin_crossentropy)
        return loss

    def setModel(self):
        # To off Batch Normalization, set BN to False
        def batch_Norm_Activation(x):
            if self.batchNorm and self.dropOut:
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
                x = Dropout(self.dropoutRate)(x)
            elif self.batchNorm and not self.dropOut:
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
            elif not self.batchNorm and self.dropOut:
                x = Activation("relu")(x)
                x = Dropout(self.dropoutRate)(x)
            else:
                x = Activation("relu")(x)
            return x
        '''
        def decoder(inputLayer, i, lvl):
            deconv = Conv2DTranspose(self.featuremaps * i, (2, 2), strides=(2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(inputLayer)
            uconv = concatenate([deconv, layerList[lvl]])
            uconv = Conv2D(self.featuremaps * i, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv)
            uconv = batch_Norm_Activation(uconv)
            shortcut = Conv2D(self.featuremaps * i, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv)
            shortcut = batch_Norm_Activation(shortcut)
            uconv = Conv2D(self.featuremaps * i, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv)
            uconv = batch_Norm_Activation(uconv)
            uconv = Conv2D(self.featuremaps * i, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv)
            uconv = batch_Norm_Activation(uconv)
            outputLayer = add([shortcut, uconv])
            return outputLayer
        '''
        # channelTmp =  self.Xtrain.shape[-1]

        if self.pretrained:

            preprocess_input1 = sm.get_preprocessing(self.backbone)

            # preprocess input
            if self.util.getImageType() == 0:
                self.Xtrain = np.concatenate([self.Xtrain] * 3, axis=-1)
                self.Xtest = np.concatenate([self.Xtest] * 3, axis=-1)
            self.Xtrain = preprocess_input1(self.Xtrain)
            self.Xtest = preprocess_input1(self.Xtest)

            if self.numClass > 2:
                self.model = sm.Unet(
                    self.backbone, encoder_weights='imagenet', classes=self.numClass, activation='softmax')
            elif self.numClass <= 2:
                self.model = sm.Unet(
                    self.backbone, encoder_weights='imagenet', classes=1, activation='sigmoid')

            if self.displaySummary:
                self.model.summary()

        # if not using pretrained models
        else:
            inputs = keras.layers.Input(
                shape=(self.img_height, self.img_width, self.Xtrain.shape[-1]))
            # Architecture borrowed from https://github.com/AhadMomin/semantic-segmentation-digital-rock-physics
            conv1 = Conv2D(self.featuremaps * 1, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(inputs)
            conv1 = batch_Norm_Activation(conv1)
            conv1 = Conv2D(self.featuremaps * 1, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(conv1)
            conv1 = batch_Norm_Activation(conv1)
            pool1 = MaxPooling2D((2, 2))(conv1)

            conv2 = Conv2D(self.featuremaps * 2, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(pool1)
            conv22 = batch_Norm_Activation(conv2)
            shortcut = Conv2D(self.featuremaps * 2, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(conv22)
            shortcut = batch_Norm_Activation(shortcut)
            conv2 = Conv2D(self.featuremaps * 2, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(conv22)
            conv2 = batch_Norm_Activation(conv2)
            conv2 = Conv2D(self.featuremaps * 2, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(conv2)
            conv2 = batch_Norm_Activation(conv2)
            conv2 = add([shortcut, conv2])
            pool2 = MaxPooling2D((2, 2))(conv2)

            conv3 = Conv2D(self.featuremaps * 4, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(pool2)
            conv33 = batch_Norm_Activation(conv3)
            shortcut1 = Conv2D(self.featuremaps * 4, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(conv33)
            shortcut1 = batch_Norm_Activation(shortcut1)
            conv3 = Conv2D(self.featuremaps * 4, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(conv33)
            conv3 = batch_Norm_Activation(conv3)
            conv3 = Conv2D(self.featuremaps * 4, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(conv3)
            conv3 = batch_Norm_Activation(conv3)
            conv3 = add([shortcut1, conv3])
            pool3 = MaxPooling2D((2, 2))(conv3)

            conv4 = Conv2D(self.featuremaps * 8, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(pool3)
            conv44 = batch_Norm_Activation(conv4)
            shortcut2 = Conv2D(self.featuremaps * 8, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(conv44)
            shortcut2 = batch_Norm_Activation(shortcut2)
            conv4 = Conv2D(self.featuremaps * 8, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(conv44)
            conv4 = batch_Norm_Activation(conv4)
            conv4 = Conv2D(self.featuremaps * 8, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(conv4)
            conv4 = batch_Norm_Activation(conv4)
            conv4 = add([shortcut2, conv4])
            pool4 = MaxPooling2D((2, 2))(conv4)

            convm = Conv2D(self.featuremaps * 16, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(pool4)
            convm = batch_Norm_Activation(convm)
            shortcut3 = Conv2D(self.featuremaps * 16, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(convm)
            shortcut3 = batch_Norm_Activation(shortcut3)
            convm = Conv2D(self.featuremaps * 16, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(convm)
            convm = batch_Norm_Activation(convm)
            convm = Conv2D(self.featuremaps * 16, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(convm)
            convm = batch_Norm_Activation(convm)
            convm = add([shortcut3, convm])

            deconv4 = Conv2DTranspose(self.featuremaps * 8, (2, 2), strides=(2, 2), padding=self.padding,
                                      kernel_regularizer=regularizers.L2(l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(convm)
            uconv4 = concatenate([deconv4, conv4])
            uconv4 = Conv2D(self.featuremaps * 8, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv4)
            uconv4 = batch_Norm_Activation(uconv4)
            shortcut4 = Conv2D(self.featuremaps * 8, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv4)
            shortcut4 = batch_Norm_Activation(shortcut4)
            uconv4 = Conv2D(self.featuremaps * 8, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv4)
            uconv4 = batch_Norm_Activation(uconv4)
            uconv4 = Conv2D(self.featuremaps * 8, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv4)
            uconv4 = batch_Norm_Activation(uconv4)
            uconv4 = add([shortcut4, uconv4])

            deconv3 = Conv2DTranspose(self.featuremaps * 4, (2, 2), strides=(2, 2), padding=self.padding,
                                      kernel_regularizer=regularizers.L2(l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv4)
            uconv3 = concatenate([deconv3, conv3])
            uconv3 = Conv2D(self.featuremaps * 4, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv3)
            uconv3 = batch_Norm_Activation(uconv3)
            shortcut5 = Conv2D(self.featuremaps * 4, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv3)
            shortcut5 = batch_Norm_Activation(shortcut5)
            uconv3 = Conv2D(self.featuremaps * 4, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv3)
            uconv3 = batch_Norm_Activation(uconv3)
            uconv3 = Conv2D(self.featuremaps * 4, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv3)
            uconv3 = batch_Norm_Activation(uconv3)
            uconv3 = add([shortcut5, uconv3])

            deconv2 = Conv2DTranspose(self.featuremaps * 2, (2, 2), strides=(2, 2), padding=self.padding,
                                      kernel_regularizer=regularizers.L2(l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv3)
            uconv2 = concatenate([deconv2, conv2])
            uconv2 = Conv2D(self.featuremaps * 2, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv2)
            uconv2 = batch_Norm_Activation(uconv2)
            shortcut6 = Conv2D(self.featuremaps * 2, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv2)
            shortcut6 = batch_Norm_Activation(shortcut6)
            uconv2 = Conv2D(self.featuremaps * 2, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv2)
            uconv2 = batch_Norm_Activation(uconv2)
            uconv2 = Conv2D(self.featuremaps * 2, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv2)
            uconv2 = batch_Norm_Activation(uconv2)
            uconv2 = add([shortcut6, uconv2])

            deconv1 = Conv2DTranspose(self.featuremaps * 1, (2, 2), strides=(2, 2), padding=self.padding,
                                      kernel_regularizer=regularizers.L2(l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv2)
            uconv1 = concatenate([deconv1, conv1])
            uconv1 = Conv2D(self.featuremaps * 1, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv1)
            uconv1 = batch_Norm_Activation(uconv1)
            shortcut7 = Conv2D(self.featuremaps * 1, (3, 3), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv1)
            shortcut7 = batch_Norm_Activation(shortcut7)
            uconv1 = Conv2D(self.featuremaps * 1, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv1)
            uconv1 = batch_Norm_Activation(uconv1)
            uconv1 = Conv2D(self.featuremaps * 1, (2, 2), padding=self.padding, kernel_regularizer=regularizers.L2(
                l2=self.L2), kernel_constraint=keras.constraints.max_norm(self.maxNorm))(uconv1)
            uconv1 = batch_Norm_Activation(uconv1)
            uconv1 = add([shortcut7, uconv1])

            if self.numClass > 2:
                final = Conv2D(self.numClass, (1, 1),
                               padding=self.padding, activation='softmax')(uconv1)
            elif self.numClass <= 2:
                final = Conv2D(1, (1, 1), padding=self.padding,
                               activation='sigmoid')(uconv1)

            self.model = Model(inputs, final)

            if self.displaySummary:
                self.model.summary()

    def run(self, forVisualisation=False):
        callbacks = []
        # tensorboard = keras.callbacks.TensorBoard(log_dir= self.logdir)
        # clr_triangular = CyclicLR(mode='triangular')
        if len(self.early_stopping) > 0:
            for early_stopping in self.early_stopping:
                early_stopping_tmp = keras.callbacks.EarlyStopping(
                    monitor=early_stopping, mode='max', verbose=1, patience=self.patience, restore_best_weights=True)
                callbacks.append(early_stopping_tmp)
        if self.loss_early_stopping:
            early_stopping_loss = keras.callbacks.EarlyStopping(
                monitor='val_loss', mode='min', verbose=1, patience=self.patience, restore_best_weights=True)
            callbacks.append(early_stopping_loss)
        """
        if len(self.early_stopping) > 0:
            print(self.early_stopping)
            i=0
            for early_stopping in self.early_stopping:
                name_metric = 'metric_' + str(i)
                early_stopping = keras.callbacks.EarlyStopping(monitor='binary_iou', patience=self.patience, restore_best_weights=True)
                i+=1
                callbacks.append(early_stopping)
        if self.loss_early_stopping:
            early_stopping_loss = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
            callbacks.append(early_stopping_loss)
        """
        # earlyStoppingValAcc=keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max',verbose=1, patience=self.patience)
        # earlyStoppingValLoss=keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',verbose=1, patience=self.patience)
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
        # lr_finder = LRFinder(min_lr=1e-5, max_lr=1)

        print('gpu : ' + str(len(self.num_gpus)))
        if len(self.num_gpus) > 0:
            steps_per_epoch = self.Xtrain.shape[0]//(
                self.batch_size * len(self.num_gpus))
            validation_steps = self.Xtest.shape[0]//(
                self.batch_size * len(self.num_gpus))
        elif len(self.num_gpus) == 0:
            steps_per_epoch = self.Xtrain.shape[0]//self.batch_size
            validation_steps = self.Xtest.shape[0]//self.batch_size
        # TODO
        if self.dataGeneration:
            batch_size_val = self.batch_size//2

            train_generator = self.dataGenerator(
                seed=105, batch_size=self.batch_size, isTrainSet=True, forVisualisation=False)
            test_generator = self.dataGenerator(
                seed=105, batch_size=batch_size_val, isTrainSet=False, forVisualisation=False)

            if self.additional_augmentation_factor != 0:
                steps_per_epoch = steps_per_epoch * self.additional_augmentation_factor

            self.model.compile(optimizer=Adam(self.learningRate),
                               loss=self.loss,
                               metrics=self.metrics)

            hist_model = self.model.fit_generator(generator=train_generator,
                                                  validation_data=test_generator,
                                                  steps_per_epoch=steps_per_epoch,
                                                  validation_steps=validation_steps,
                                                  # shuffle=True,
                                                  epochs=self.epochs,
                                                  callbacks=[callbacks])
        else:
            self.model.compile(optimizer=Adam(self.learningRate),
                               loss=self.loss,
                               metrics=self.metrics)

            hist_model = self.model.fit(x=self.Xtrain,
                                        y=self.Ytrain,
                                        validation_data=(
                                            self.Xtest, self.Ytest),
                                        shuffle=True,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_steps=validation_steps,
                                        callbacks=[callbacks])

            # Saves the history under the form of a plot
            training_history = hist_model.history
            metric_names = list(hist_model.history.keys())
            for metric_name in metric_names:
                plt.plot(training_history[metric_name], label=metric_name)
            plt.title('Metrics - ' + self.fileName)
            plt.xlabel('Epoch')
            plt.ylabel('Metric value')
            plt.legend(metric_names, loc='best')

            # Saves the model and the weights for future loading and usage
            if self.save_model:
                if not os.path.exists(self.logdir):
                    os.makedirs(self.logdir)
                # saves the plot
                plt.savefig(os.path.join(self.logdir, 'metrics.png'))
                # Save the scaler to a binary file
                if self.util is not None:
                    dump(self.util.scaler, os.path.join(
                        self.logdir, 'std_scaler_image.bin'), compress=True)
                # Saves the log file containing the variables used to compile the model
                with open(self.logdir+'/logs.txt', 'w') as f:
                    f.write(self.FILE_TXT)
                model_json = self.model.to_json()
                newpath = os.path.join(self.logdir, 'weights')
                print(newpath)
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                with open(os.path.join(newpath, 'weights.json'), "w") as json_file:
                    json_file.write(model_json)
                    self.model.save_weights(
                        os.path.join(newpath, 'weights.h5'))
                    self.model.save(newpath)

        # return self.model
