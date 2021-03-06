
"""
Created on Wed May 10 18:33:44 2017

@author: richa
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import label
from skimage.measure import regionprops

from utils import plt_i, notZero

import cv2

import scipy.ndimage as nd

def plt_s(regions):
    """
    A partir d'unes regions, les printa al plot
    * Inputs:
    - regions = list of props of regionprops
    """
    for prop in regions:
        plt.plot(prop['centroid'][1],prop['centroid'][0],marker='o')
        minr, minc, maxr, maxc = prop['BoundingBox']
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        plt.plot(bx, by, '-b', linewidth=2.5)
        plt.hold(True)
    plt.hold(False)
    


def get_h_w(region):
    """
    A partir d'una regio, calcula l'alcada i l'amplada
    * Inputs:
    - regions = list of props of regionprops
    *Outputs:
    - height = alcada de regio
    - width = amplada de regio
    """
    minr, minc, maxr, maxc = region['BoundingBox']
    height = int(maxc-minc)
    width = int(maxr-minr)
    return height, width
    
def get_lists_regions(regions):
    """
    A partir d'unes regions, retorna una llista amb alcades i amplades
    * Inputs:
    - regions = list of props of regionprops
    *Outputs:
    - m_height = mitjana d'alcada
    - m_width = mitjana d'amplada
    """
    height = []
    width = []
    for prop in regions:      
        minr, minc, maxr, maxc = prop['BoundingBox']
        height.append(int(maxc-minc))
        width.append(int(maxr-minr))
    
    return height, width
def get_medians(regions):
    """
    A partir d'unes regions, retorna la mediana d'alcada i amplada
    * Inputs:
    - regions = list of props of regionprops
    *Outputs:
    - m_height = mediana d'alcada
    - m_width = mediana d'amplada
    """
    
    height, width = get_lists_regions(regions)
    
    m_height = np.median(height);
    m_width = np.median(width)
    return m_height, m_width

def get_means(regions):
    """
    A partir d'unes regions, retorna la mitjana d'alcada i amplada
    * Inputs:
    - regions = list of props of regionprops
    *Outputs:
    - m_height = mitjana d'alcada
    - m_width = mitjana d'amplada
    """
    
    height, width = get_lists_regions(regions)
    
    m_height = np.mean(height);
    m_width = np.mean(width)
    return m_height, m_width
    
def split_im_regions(im, regions, mean_height=1):
    """
    A partir d'unes regions i una imatge, retorna les imatges retallades de les 
    regions. Mean height ha quedat deprecated
    * Inputs:
    - im = skimage.io image
    - params = diccionari de parametres
    - mean_height = deprecated
    *Outputs:
    - im_list = llista de imatges
    """
    im_list = []
    for i in range(len(regions)):
        minr, minc, maxr, maxc = regions[i]['BoundingBox']
        charac = im[(minr):(maxr),(minc):(maxc)]
        im_list.append(charac)
        
    return im_list
    
def close_vert_median(im):
    """
    Closing vertical de la imatge a partir de la mediana de les regions de la
    mateixa.
    * Inputs:
    - im = skimage.io image
    *Outputs:
    - im = imatge amb closing
    - kern_siz = mediana
    """
    if not(im.diagonal().shape == (1L,)):
        #Labeling the image in order to get mean of height of the regions
    
        label_image = label(im)
    
        regions = regionprops(label_image)
        if(len(regions)):
            height, width = get_medians(regions)
    
            #With the mean of height, we can do a vertical close in order to join
            # dots of "i" and other marks

            kern_siz = np.median(height)
            kernel = np.ones((notZero(kern_siz/2),1),np.uint8)
            im = cv2.erode(cv2.dilate(np.int16(im),kernel,1),kernel,1)
        else:
            kern_siz = 1.0
    else:
        kern_siz = 1.0
    return im, kern_siz

def get_lines(params, im, mean_height):
    """
    A partir d'una imatge, n'extreu les frases
    * Inputs:
    - im = skimage.io image
    - params = diccionari de parametres
    *Outputs:
    - line_im_list = llista de caracters
    """
    #We will apply y-derivative of gaussian filter
    
    imy = np.zeros(im.shape,dtype=np.float64)
    nd.filters.sobel(im,0,imy)
    
#    x, y = im.shape
#    ws = x / 220
    
    label_image = label(im, connectivity = 2)
    regions = regionprops(label_image)
    m_height, m_width = get_medians(regions)
    
    
    g10 = nd.filters.gaussian_filter(imy, notZero(m_width/2), order=(1,0))
    gbin = g10<0

    kernel = np.ones((notZero(m_height/3),1),np.uint8)
    gbin = cv2.dilate(np.int16(gbin),kernel,1)
#    plt_i(gbin)
#    kernel = np.ones((int(notZero(m_height/7)),1),np.uint8)
#    gbin = cv2.dilate(np.int16(gbin),kernel,1)
#    plt_i(gbin)
    
    label_image = label(gbin, connectivity = 2)
    regions = regionprops(label_image)
    
     
    if(params["TEST_MODE"]["line_detect"]):
        plt_i(im)
        plt_i(imy)
        plt_i(g10)
        plt_i(gbin)

        plt.hold(True)
        
        plt_s(regions)
        
        for i in range(0,len(regions)):
            minr, minc, maxr, maxc = regions[i]['BoundingBox']
            charac = im[(minr-mean_height):(maxr+mean_height),(minc-mean_height):(maxc+mean_height)]
            plt_i(charac)
        plt.hold(False)
        
    line_im_list = split_im_regions(im, regions, mean_height)
        
    return line_im_list

def get_words_from_line(params, im):
    """
    A partir d'una frase, n'extreu les paraules
    * Inputs:
    - im = skimage.io image
    - params = diccionari de parmetres
    *Outputs:
    - word_im_list = llista de paraules
    """
    
    kernel = np.ones((2,2),np.uint8)
#    imd = cv2.dilate(np.int16(imb),kernel,1)
    
    imd1, mean_height = close_vert_median(im)
    
    kernel = np.ones((1,int(mean_height/1.35)),np.uint8)
    imd2 = cv2.dilate(np.int16(imd1),kernel,1)

    
    label_image = label(imd2, connectivity = 2)
    regions = regionprops(label_image)
    
    if len(regions):
        regions = sorted(regions, key=lambda x: x.bbox[1])
        
    if(params["TEST_MODE"]["word_detect"]):
        plt_i(im,"word_detect im")
        plt_i(imd1,"word_detect im1")
        plt_i(imd2,"word_detect im2")
        
        plt.hold(True)
        
        plt_s(regions)

        plt.hold(False)

    word_im_list = split_im_regions(im, regions, mean_height)
    
    return word_im_list

def get_letters(params, im):
    """
    A partir d'una paraula, n'extreu les lletres
    * Inputs:
    - im = skimage.io image
    - params = diccionari de parametres
    *Outputs:
    - ch_im_list = llista de caracters
    """
    
    #detect if they are itallics
    x, y = im.shape
    
    imd1, mean_height = close_vert_median(im)
    labels1 = label(imd1, connectivity = 2)
    regions1 = regionprops(labels1)
    m_height1, m_width1 = get_medians(regions1)
    
    
    
    label_image = label(imd1)
    regions = regionprops(label_image)

    if len(regions):
        regions = sorted(regions, key=lambda x: x.bbox[1])

    ch_im_list = []
    for i in range(len(regions)):
        minr, minc, maxr, maxc = regions[i]['BoundingBox']
        charac = im[(0):(x),(minc):(maxc)]
        ch_im_list.append(charac)

#    ch_im_list = split_im_regions(im, regions, mean_height)
    
    if(params["TEST_MODE"]["char_detect"]): 
        for img in ch_im_list:
            plt_i(img, maxp=30)
        
    return ch_im_list
    
def get_all(im, params):
    '''
    Analyzes a binary image and returns the letters
    input:
    im : numpy.uint8 image values where low values are the letters
    params : data struct used for debugging
    '''    
    if(params["status_msg"]): print "Getting Chars Activated."
    
    imn = 1-im
    imb, mean_height = close_vert_median(imn)
    label_image = label(imb)
    regions = regionprops(label_image)
    height, width = get_medians(regions)
    
    im_lines = get_lines(params, imn, height)
    
    structure = []

    for im_line in im_lines:
        if not(im_line.diagonal().shape == (1L,)): 
            if(np.max(im_line) == 1):
                
                im_words = get_words_from_line(params, im_line)
                list_letters = []
                for im_word in im_words:
                    if not(im_word.diagonal().shape == (1L,)): 
                        if(np.max(im_word) == 1):
                            
                            im_letters = get_letters(params, im_word)
                            for i in range(len(im_letters)):
                                im_letters[i] = 1-im_letters[i]
                            list_letters.append(im_letters)
                structure.append(list_letters) 
    
    if(params["status_msg"]): print "Getting Chars OK."
    
    return structure




















