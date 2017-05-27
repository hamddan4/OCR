# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Automatically detect rotation and line spacing of an image of text using
Radon transform
If image is rotated by the inverse of the output, the lines will be
horizontal (though they may be upside-down depending on the original image)
It doesn't work with black borders
"""

from __future__ import division, print_function
from skimage.transform import radon, rotate
from numpy import asarray, mean, array
import numpy
import matplotlib.pyplot as plt
from matplotlib.mlab import rms_flat

from get_chars import plt_i

try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic
    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax

filename = 'img/skew.png'

def get_rotation(im,params):

    I = asarray(im)
    
    I = I - mean(I)  # Demean; make the brightness extend above and below zero
    
    plt.figure()
    if(params["TEST_MODE"]["rotation"]):
        plt.subplot(1, 2, 1)
        plt.imshow(I)
    # Do the radon transform and display the result
    sinogram = radon(I)
    
    if(params["TEST_MODE"]["rotation"]):
        plt.subplot(1, 2, 2)
        plt.imshow(sinogram.T, aspect='auto')
        plt.gray()
    
    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
    r = array([rms_flat(line) for line in sinogram.transpose()])
    rotation = argmax(r)
    plt.axhline(rotation, color='r')
    
    return rotation

def fix_rotation(im, params):
    
    rotation = get_rotation(im, params)
    
    imr = rotate(im, 90-rotation, preserve_range=True)
    
    return imr