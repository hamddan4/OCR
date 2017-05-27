# -*- coding: utf-8 -*-
"""

@author: Dani, richard
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def evenNum(number):
    return number + (number%2)
    
def oddNum(number):
    return number + np.uint(not(number%2))


def rescale(im,final_size,padding):
    sy,sx = final_size
    sy,sx = sy-padding,sx-padding
    
    if(np.shape(im) != (sy,sx)):
        modified_im = np.ones(final_size)
        current_size = np.shape(im)
        
        if(current_size[1]>current_size[0]):
            pass
            im = cv2.resize(im,(sx,int(sx*current_size[0]/current_size[1])))
        else:
            pass
            im = cv2.resize(im,(int(sy*current_size[1]/current_size[0]),sy))
            
        xmid = int((sx+padding)/2)
        ymid = int((sy+padding)/2)            
        modified_im[(xmid-int(np.shape(im)[0]/2)):(xmid+int(np.ceil(np.shape(im)[0]/2.0))),
                    (ymid-int(np.shape(im)[1]/2)):(ymid+int(np.ceil(np.shape(im)[1]/2.0)))] = im 
            
    return modified_im
 
max_plots = 0
def plt_i(im,title=None, maxp=20):
    global max_plots
    max_plots += 1
    if(max_plots < maxp):
        plt.figure(), plt.imshow(im,cmap=plt.cm.gray)
    else:
        raise Exception("Too many plots are open! I might just saved your ass here...")
#        pass
    if not(title == None):
        plt.title(title)   
