# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:33:44 2017

@author: richa
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from skimage.morphology import label
from skimage.measure import regionprops

import cv2
import cv


im = color.rgb2gray(io.imread("./img/Practica5-1.png"))
imb = im>128
#kernel = np.ones((6,1),np.uint8)

plt.figure(),plt.imshow(im,cmap='gray')
imb_neg = 1-imb
plt.figure(), plt.imshow(imb_neg,cmap='gray')

label_image = label(imb_neg)
regions = regionprops(label_image)

hight = []
for prop in regions:      
    minr, minc, maxr, maxc = prop['BoundingBox']
    hight.append(int(maxc-minc))
plt.hold(False)

kern_siz = np.median(hight)

kernel = np.ones((int(kern_siz/2),1),np.uint8)

imb_neg = cv2.erode(cv2.dilate(np.int16(imb_neg),kernel,1),kernel,1)

label_image = label(imb_neg)
regions = regionprops(label_image)

plt.figure()
for prop in regions:
    plt.plot(prop['centroid'][1],prop['centroid'][0],marker='o')
        
    minr, minc, maxr, maxc = prop['BoundingBox']
    hight.append(int(maxc-minc))
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    plt.plot(bx, by, '-b', linewidth=2.5)
    plt.hold(True)
plt.hold(False)

for i in range(1,20):
    
    minr, minc, maxr, maxc = regions[i]['BoundingBox']
    line = im[minr:(maxr+1),minc:(maxc+1)]
    plt.figure(), plt.imshow(line,cmap='gray')

#plt.close('all')




















