# -*- coding: utf-8 -*-
"""

@author: Dani, richard
"""

import numpy as np
import cv2
from skimage import color, filters, io
import matplotlib.pyplot as plt
import get_chars as gc

def evenNum(number):
    return number + (number%2)
    
def oddNum(number):
    return number + np.uint(not(number%2))


def rescale(im,final_size):
    sy,sx = final_size
    
    if(np.shape(im) != final_size):
        modified_im = np.ones(final_size)
        current_size = np.shape(im)
        
        if(current_size[1]>current_size[0]):
            im = cv2.resize(im,(sx,evenNum(int(sx*current_size[0]/sy))))
        else:
            im = cv2.resize(im,(evenNum(int(sy*current_size[1]/sx)),sy))
            
        xmid = int(round(np.shape(modified_im)[0]/2))
        ymid = int(round(np.shape(modified_im)[1]/2))            
        modified_im[(xmid-int(np.floor(np.shape(im)[0]/2))):(xmid+int(np.floor(np.shape(im)[0]/2))),
                    (ymid-int(np.floor(np.shape(im)[1]/2))):(ymid+int(np.floor(np.shape(im)[1]/2)))] = im 
            
    return modified_im
    
