# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:42:39 2022

@author: florent.brondolo
"""
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from skimage import io
from skimage.color import rgb2gray  # Import rgb2gray for converting to grayscale
import random
from .ProgressBar import ProgressBar
from sklearn.preprocessing import LabelEncoder
import tifffile as tiff
import cv2

class Stack:
            
    def __repr__(self):
         return 'Stack'
     
    #def __init__(self, name, isSegmented, maskPreprocessing, isMulticlass, width=None, height=None, channel=1, path=None):
    def __init__(self, **kwargs):
        self.listSlice = []
        self.listSlicePath = []
        
        if "name" in kwargs:
            self.name = kwargs['name']
        
        if "isSegmented" in kwargs:
            self.isSegmented = kwargs['isSegmented']
        else:
            self.isSegmented = False
        
        if "imageType" in kwargs:
            self.imageType = kwargs['imageType']
        else:
            self.imageType = 0 #We assume that it's a grayscale scanner img if not specified
        '''
        if "isMulticlass" in kwargs:
            self.isMulticlass = kwargs['isMulticlass']
        else:
            self.isMulticlass = False
        '''

        if "stackImage" in kwargs:
            self.stackImage = kwargs['stackImage']
        else:
            self.stackImage = None
        
        if "stackLabel" in kwargs:
            self.stackLabel = kwargs['stackLabel']
             
        if "path" in kwargs:
            self.path = kwargs['path']
        else:
            self.path = None

        if "channel" in kwargs:
            self.channel = kwargs['channel']
        '''
        else:
            self.channel = 1
        '''
        
        if ("width" in kwargs) and ("height" in kwargs):
            self.width = kwargs['width']
            self.height = kwargs['height']
        else:
            #Check which length is the shortest, then applies it to both height and width
            #This is done in order to keep a square volume
            widthTmp = cv2.imread(self.path +'//'+ next(os.walk(self.path))[2][0]).shape[1]
            heightTmp = cv2.imread(self.path +'//'+ next(os.walk(self.path))[2][0]).shape[0]
            if widthTmp < heightTmp:
                self.width = widthTmp
                self.height = widthTmp
            elif widthTmp > heightTmp:
                self.width = heightTmp
                self.height = heightTmp
            else:
                self.width = widthTmp
                self.height = heightTmp 

        #path setting
        # if a path is provided: next
        # if no path is provided: either a list of slice is provided (used in the Util class), or it'll raise an error
        # if a path and a list of slice are provided : an error will be raised
        if (self.path is None):
          if ("isSliceListSupplied" not in kwargs):
              raise Exception(repr(self) + 'class - a folder path or a list of slices must be provided')   
          else: #isSliceListSupplied is in kwargs
              self.isSliceListSupplied = kwargs['isSliceListSupplied']
        else: #path is in kwargs (path is provided)
           if ("isSliceListSupplied" not in kwargs):
              self.isSliceListSupplied = False   
              #Get the total amount of slices present in the CT scan
              numTotalSlice = len([name for name in os.listdir(self.path)])
           else: #isSliceListSupplied is in kwargs
              raise Exception(repr(self) + 'class - path and isSliceListSupplied cannot be set at the same time')
        
        '''
        if ("numSlice" in kwargs) and ("selectedFiles" not in kwargs): #If a number of slices is specified, but no list of preselected files is
            try:
              #Check if a number of slice to pick has been specified
              #If yes : n slices are randomly selected (no duplicate)
              #If not : all slices are selected
              if kwargs['numSlice'] == None:
                self.numSlice = numTotalSlice
                self.selectedFiles = range(numTotalSlice) 
              else:
                if self.numSlice <= numTotalSlice: # Check that numSlice is < to the total number of available slices  
                    self.numSlice = kwargs['numSlice']
                    self.selectedFiles = random.sample(range(0,numTotalSlice-1), self.numSlice) 
                else:
                    raise Exception(repr(self) + ' class - numSlice should be less or equal to the number of slice in the folder provided')    
            except:
                raise Exception(repr(self) + ' class - if numSlice is provided and selectedFiles is not : the path must me provided')       
        elif ("numSlice" not in kwargs) and ("selectedFiles" not in kwargs): #If all slices are selected, and not preselected files are passed on
            if self.isSegmented: #if the stack processed is a mask, then selectedFiles should have been provided
                raise Exception(repr(self) + ' class - if the stack is a mask stack then the associated selectedFiles from the corresponding image stack must be provided')    
            else:   
              try:
                  self.numSlice = numTotalSlice
                  self.selectedFiles = range(numTotalSlice)
              except:
                  raise Exception(repr(self) + ' class - if numSlice is not provided and selectedFiles is not: the path must me provided')       
        elif ("numSlice" in kwargs) and ("selectedFiles" in kwargs): #If a number of slice and the list of preselected files are passed on
            #Check if a number of slice to pick has been specified
            #If yes : n slices are randomly selected (no duplicate)
            #If not : all slices are selected
            if (type(kwargs['numSlice']) == int) or (type(kwargs['numSlice']) == float):
              if int(kwargs['numSlice']) != (len(kwargs['selectedFiles'])):
                raise Exception(repr(self) + ' class - numSlice and selectedFiles must display the same length')
              else:
                self.numSlice = int(kwargs['numSlice'])
                self.selectedFiles = kwargs['selectedFiles']
            elif kwargs['numSlice'] == None:
              self.numSlice = numTotalSlice
              self.selectedFiles = kwargs['selectedFiles']
            else:
              raise Exception(repr(self) + ' class - numSlice must either be a number or None')
        elif ("numSlice" not in kwargs) and ("selectedFiles" in kwargs): #If only the list of preselected files is passed on
            self.selectedFiles = kwargs['selectedFiles']
            self.numSlice = (len(self.selectedFiles))
        '''
        '''
        #Cleaning is needed if masks are pixelated
        #By default set at False if the stack is not the set of segmented slices
        if self.isSegmented:
          self.maskPreprocessing = kwargs['maskPreprocessing']
        else:
          self.maskPreprocessing = False
        '''
        
        if "numSlice" in kwargs and "selectedFiles" not in kwargs:
            # If a number of slices is specified, but no list of preselected files is:
            # - If numSlice is None, all slices are selected.
            # - If numSlice is a valid number, n slices are randomly selected (no duplicate).
            # - If numSlice is greater than the total number of slices, an exception is raised.
            if kwargs["numSlice"] is None:
                  self.numSlice = numTotalSlice
                  self.selectedFiles = range(numTotalSlice)
            elif kwargs["numSlice"] <= numTotalSlice:
                  self.numSlice = kwargs["numSlice"]
                  self.selectedFiles = random.sample(range(0, numTotalSlice - 1), self.numSlice)
            else:
                  raise Exception(
                      repr(self)
                      + " class - numSlice should be less or equal to the number of slice in the folder provided"
                  )
        elif "numSlice" not in kwargs and "selectedFiles" not in kwargs:
            # If all slices are selected, and not preselected files are passed on:
            # - If the stack is a mask stack, an exception is raised.
            # - If the stack is an image stack, all slices are selected.
            if self.isSegmented:
                raise Exception(
                    repr(self)
                    + " class - if the stack is a mask stack then the associated selectedFiles from the corresponding image stack must be provided"
                )
            else:
                  self.numSlice = numTotalSlice
                  self.selectedFiles = range(numTotalSlice)
        elif "numSlice" in kwargs and "selectedFiles" in kwargs:
            # If a number of slices and the list of preselected files are passed on:
            # - If numSlice is a valid number and the length of the list of preselected files is different, an exception is raised.
            # - If numSlice is None, the length of the list of preselected files is used.
            # - If numSlice is not a valid number, an exception is raised.
            if (
                type(kwargs["numSlice"]) == int
                or type(kwargs["numSlice"]) == float
            ) and int(kwargs["numSlice"]) != len(kwargs["selectedFiles"]):
                raise Exception(
                    repr(self)
                    + " class - numSlice and selectedFiles must display the same length"
                )
            elif kwargs["numSlice"] is None:
                self.numSlice = len(kwargs["selectedFiles"])
                self.selectedFiles = kwargs["selectedFiles"]
            else:
                raise Exception(repr(self) + " class - numSlice must either be a number or None")
        elif "numSlice" not in kwargs and "selectedFiles" in kwargs:
            # If only the list of preselected files is passed on:
            # - The length of the list of preselected files is used.
            self.selectedFiles = kwargs["selectedFiles"]
            self.numSlice = len(self.selectedFiles)


        self.setListSlice()
        
    ### GETs
    def getPath(self):
        return self.path
    
    def getIsSegmented(self):
        return self.isSegmented
    
    def getImageType(self):
        return self.imageType
    '''
    def getIsMulticlass(self):
        return self.isMulticlass
    '''

    def getName(self):
        return self.name
    
    '''
    def getMaskPreprocessing(self):
        return self.maskPreprocessing
    '''
    
    def getSliceFromPosition(self, position):
        #If the user wants a single slice to be returned
        if isinstance(position, int):
            return self.listSlice[position]
        #If multiple slices are selected
        elif isinstance(position, list):
            return [self.listSlice[i] for i in position]
        else:
            raise Exception(repr(self) + ' class - getSliceFromPosition can only take integer or list of intergers as input')
            
    def getSliceFromName(self, name):
        sliceTmp = next((x for x in self.listSlice if x.name == name), None)
        return sliceTmp
    
    def getChannel(self):
        return self.channel
    
    def getStackSize(self):
          return self.numSlice

    def getListSlice(self):
        return self.listSlice    
    
    def getWidth(self):
        return self.width
    
    def getHeight(self):
        return self.height
    
    """
    def getListSlicePath(self):
        return self.listSlicePath
    """
    
    def getStackImage(self):
        return self.stackImage
    
    def getStackLabel(self):
        return self.stackLabel
        
    def getSelectedFiles(self):
        return self.selectedFiles
    
    def getNumClass(self):
        return self.numClass
    """ 
    def getstackTransformed(self):
        return self.stackTransformed
    """
    def getIsSliceListSupplied(self):
        return self.isSliceListSupplied
    
    def cropAroundCenter(self, img):
        """
          Given a NumPy / OpenCV 2 image, crops it to the given width and height,
          around it's centre point
          """
        image_size = (img.shape[1], img.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))
        
        '''
        if (width > image_size[0]):
          width = image_size[0]
        
        if (height > image_size[1]):
          height = image_size[1]
        '''
        x1 = int(image_center[0] - self.width * 0.5)
        x2 = int(image_center[0] + self.width * 0.5)
        y1 = int(image_center[1] - self.height * 0.5)
        y2 = int(image_center[1] + self.height * 0.5)

        return img[y1:y2, x1:x2]   

    def setPath(self, path):
        self.path = path

    def setListSlice(self):
        #If a list of slices is supplied already
        if self.isSliceListSupplied:
            self.listSlice = self.selectedFiles    
        else:
            #If no list of slices is supplied, the slices must be generated
            image_files = sorted(os.listdir(self.path))
       
            # Iterate through all image files in the input folder
            def imageReading(image_path):
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')):
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
                        image = np.expand_dims(image, axis=-1)
                        self.channel = 1
                    else:
                        #if RGB as input and the user want grayscale scanner img as output
                        if self.imageType == 0 :
                            # Convert multi-channel image to grayscale
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            image = np.expand_dims(image, axis=-1)
                            self.channel = 1
                        #if RGB as input and the user want RGB img type as output
                        elif self.imageType == 1:
                            self.channel = 3
                            
                    image = self.cropAroundCenter(image)
                    image_sequence.append(image)
                else:
                    raise Exception(repr(self) + ' class - Files can be png, jpg, jpeg, bmp, gif, tif or tiff')

            #If it's a scanner/image
            if self.isSegmented == False:
                pb = ProgressBar(self.numSlice, txt=repr(self)+'- Loading '+ self.name)
                image_sequence = []
                for index in self.selectedFiles:    
                    image_path = os.path.join(self.path, image_files[index])
                    imageReading(image_path)
                    self.listSlicePath.append(image_path)
                    self.numClass = 0
                    pb += 1
                self.listSlice = image_sequence
            #If it's a label/mask
            else:
                pb = ProgressBar(self.numSlice, txt=repr(self)+'- Loading '+ self.name)
                image_sequence = []
                for index in self.selectedFiles:
                    image_path = os.path.join(self.path, image_files[index])
                    imageReading(image_path)
                    self.listSlicePath.append(image_path)
                    self.numClass = 0
                    pb += 1
                self.listSlice = image_sequence
    
    def setNumClass(self, numClass):
        self.numClass = numClass
        
    def setStackImage(self, stackImage):
        self.stackImage = stackImage
    
    def setStackLabel(self, stackLabel):
        self.stackLabel = stackLabel