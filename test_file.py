# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:26:12 2017

@author: Danney
"""
from skimage import filters, io
from skimage.transform import rescale
from skimage.color import rgb2grey
import matplotlib.pyplot as plt

import numpy as np
import cv2

def neutre(im):  
    kernel = np.ones((6,6),np.uint8)
    resd = cv2.erode(cv2.dilate(np.int16(im*255),kernel,1),kernel,1)
    im = im/resd
    
    return im

    
plt.close('all')
image = io.imread("./img/brut.png")

image = rgb2grey(image)
image = rescale(image, 1, multichannel = False)


sauv = filters.threshold_sauvola(image, window_size = 101)
res = image>sauv
#
#plt.figure(), plt.imshow(image,cmap=plt.cm.gray)
#plt.figure(), plt.imshow(sauv,cmap=plt.cm.gray)
#plt.figure(), plt.imshow(res,cmap=plt.cm.gray)


netejar(image)