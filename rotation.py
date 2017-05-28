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

from im_treatement import resizing

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
    """
    Funcio que descobreix la rotacio de la imatge amb la transformada de redo
    * Inputs:
    - im = skimage.io image
    - params = diccionari de parametres
    *Outputs:
    - rotation = angle de rotacio
    """

    I = asarray(im)
    
    I = I - mean(I)  # Demean; make the brightness extend above and below zero
    
    # Do the radon transform and display the result
    sinogram = radon(I)
    
    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
    r = array([rms_flat(line) for line in sinogram.transpose()])
    rotation = argmax(r)
    
    if(params["TEST_MODE"]["rotation"]):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(I)
        plt.subplot(1, 3, 2)
        plt.imshow(sinogram.T, aspect='auto')
        plt.gray()
        plt.axhline(rotation, color='r')
        
    return rotation

def fix_rotation(im, params):
    """
    Rota la imatge perque les linies estiguin horitzontalment
    * Inputs:
    - im = skimage.io image
    - params = diccionari de paràmetres
    *Outputs:
    - im = imatge amb el text amb orientacio horitzontal 
    """
    
    if(params["status_msg"]): print("Rotation Module Activated: Get Skew Angle")
    
    x, y = im.shape
    sz = 400
    
    if(x>sz):
        im_small = resizing(im, sz)
    else:
        im_small = im
    rotation = get_rotation(im_small, params)
    
    if(params["status_msg"]): print("Rotating: ", 90-rotation)
    
    imr = rotate(im, 90-rotation, preserve_range=True, cval=numpy.max(im))
    if(params["TEST_MODE"]["rotation"]):
        plt.subplot(1, 3, 3)
        plt.imshow(imr)
        
    if(params["status_msg"]): print("Rotation OK")
    
    
    return imr