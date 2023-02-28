'''
Requirements:
Segmentation is the process of dividing an image into different regions based on the characteristics of pixels to
identify objects or boundaries to simplify an image and more efficiently analyze it. The goal of segmentation is to
simplify and change the representation of an image into something that is more meaningful and easier to analyze.
Image segmentation is the process of assigning a label to every pixel in an image such that pixels with the same label
share certain characteristics. In this assignment, students are asked to write a program that implements algorithms for
image segmentation. Details of the functions are described below:

Function 1: Snakes algorithm we try to move snake in a direction where energy is minimum. Snake model is designed to
vary its shape and position while tending to search through the minimal energy state.  Snake propagates through the
domain of the image to reduce the energy function, and intends to dynamically move to the local minimum. You are
required to implement a Snakes algorithm for active contours.

Function2: A watershed is a transformation defined on a grayscale image. The name refers metaphorically to a geological
watershed, or drainage divide, which separates adjacent drainage basins.  The watershed transformation treats the image
it operates upon like a topographic map, with the brightness of each point representing its height, and finds the lines
that run along the tops of ridges. You are required to implement the Watershed algorithm in python to perform the image
segmentation.

Function 3: K Means is a clustering algorithm. It is used to identify different classes or clusters in the given data
based on how similar the data is. Data points in the same group are more similar to other data points in that same group
than those in other groups. The main idea of performing the following process is to find those areas of pixels that
share the same color hue parameter value. You are required to implement the K-means for Segmentation.

Function 4: Mean shift implicitly models this distribution using a smooth continuous non-parametric model. The key to
mean shift is a technique for efficiently finding peaks in this high-dimensional data distribution without ever
computing the complete function explicitly. The idea is to replace each pixel with the mean of the pixels in a range
neighborhood and whose value is within a distance d. You are required to implement the Mean shift algorithm for segmentation.
'''

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import active_contour

cam = cv.VideoCapture(0)


# cam = cv2.VideoCapture('video.mp4')

# Preprocessing
def preprocessing(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
    return img

#function 1: Snakes algorithm Opencv
def snakes(img):



