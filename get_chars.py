# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:33:44 2017

@author: richa
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, filters, io
from skimage.morphology import label
from skimage.measure import regionprops

import cv2

import scipy.ndimage as nd

max_plots = 0
def plt_i(im):
    global max_plots
    max_plots += 1
    if(max_plots < 20):
        plt.figure(), plt.imshow(im,cmap=plt.cm.gray)
    else:
        raise Exception("Too many plots are open!")
#        pass

def plt_s(regions):
    for prop in regions:
        plt.plot(prop['centroid'][1],prop['centroid'][0],marker='o')
        minr, minc, maxr, maxc = prop['BoundingBox']
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        plt.plot(bx, by, '-b', linewidth=2.5)
        plt.hold(True)
    plt.hold(False)
    
def resizing(im, contour):
    x, y = im.shape
    dismx = x / contour
    dimx = x / dismx
    dimy = y / dismx
    im2 = cv2.resize(im, (dimy, dimx)) 
    return im2    
      
def neutre(im):  
    
    [x, y] = im.shape
    kernel = np.ones((x/100,x/100),np.uint8)
    resd = cv2.erode(cv2.dilate(np.int16(im*255),kernel,1),kernel,1)
    plt_i(resd)
    plt.title('closed for neutralization');
    im = im/resd
    
    return im
    
def get_means(regions):
    hight = []
    width = []
    for prop in regions:      
        minr, minc, maxr, maxc = prop['BoundingBox']
        hight.append(int(maxc-minc))
        width.append(int(maxr-minr))
    
    m_height = np.median(hight);
    m_width = np.median(width)
    return m_height, m_width

    
def split_im_regions(im, regions, mean_height):
    im_list = []
    for i in range(len(regions)):
        minr, minc, maxr, maxc = regions[i]['BoundingBox']
        charac = im[(minr):(maxr+1),(minc):(maxc+1)]
        im_list.append(charac)
        
    return im_list
    
def apply_threshold(im):
    x, y = im.shape
    
    ws = x / 10
    ws = ws + np.uint(not(ws%2))
    
    sauv = filters.threshold_sauvola(im, window_size = ws)
    plt_i(sauv)
    plt.title('sauvola');
    res = im>sauv
    #kernel = np.ones((6,1),np.uint8)
    return res

def noise_removal(im):
    im = np.uint8(im*255)
    
    blur = cv2.GaussianBlur(im,(9,9),0)
    ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    imb = (im + th2)
    plt_i(imb)
    plt.title('Otsu thresholding + Gaussian filtering');

    #Fem un close per detectar regions o taques massa grans
    x, y = im.shape
    sz = x / 250
    kernel = np.ones((sz,sz),np.uint8)
    imc = cv2.erode(imb,kernel,1)
    
    plt_i(imc)
    plt.title('noise closed')
    
    neg = (255-imc)/255
    label_image = label(neg)
    plt_i(label_image)
    
    regions = regionprops(label_image)
    plt_i(neg)
    plt.hold(True)
    plt_s(regions)
    plt.hold(False)
    
    height, width = get_means(regions)
    
    
    return im

def close_vert_median(im):
    #Labeling the image in order to get mean of height of the regions
    label_image = label(im)
    regions = regionprops(label_image)
    
    height, width = get_means(regions)

    #With the mean of height, we can do a vertical close in order to join
    # dots of "i" and other marks
    kern_siz = np.median(height)
    kernel = np.ones((int(kern_siz/2),1),np.uint8)
    im = cv2.erode(cv2.dilate(np.int16(im),kernel,1),kernel,1)
    
    return im, kern_siz

    

#plt.figure(),plt.imshow(im,cmap='gray')
def get_lines(params, im, mean_height):
    #We will apply y-derivative of gaussian filter
    
    imy = np.zeros(im.shape,dtype=np.float64)
    nd.filters.sobel(im,0,imy)
    
    x, y = im.shape
    ws = x / 220
    
    g10 = nd.filters.gaussian_filter(imy, ws, order=(1,0))
    gbin = g10>0
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
    
    imb = 255-im
    
    kernel = np.ones((2,2),np.uint8)
#    imd = cv2.dilate(np.int16(imb),kernel,1)
    
    imd1, mean_height = close_vert_median(imb)
    
    kernel = np.ones((1,int(mean_height/1.35)),np.uint8)
    imd2 = cv2.dilate(np.int16(imd1),kernel,1)

    if(params["TEST_MODE"]["word_detect"]):
        plt_i(im)
        plt_i(imb)
        plt_i(imd1)
        plt_i(imd2)
    
    label_image = label(imd2, connectivity = 2)
    regions = regionprops(label_image)

    word_im_list = split_im_regions(im, regions, mean_height)
    
    return word_im_list

def get_letters(params, im):
    
    
    imb = 255-im
    
    #detect if they are itallics
    imd1, mean_height = close_vert_median(imb)
    labels1 = label(imd1, connectivity = 2)
    regions1 = regionprops(labels1)
    m_height1, m_width1 = get_means(regions1)
    
    
    label_image = label(imd1)
    regions = regionprops(label_image)

    if len(regions):
        regions = sorted(regions, key=lambda x: x.bbox[1])

#    if(params["TEST_MODE"]["char_detect"]):
#        plt.figure()
#        for prop in regions:
#            plt.plot(prop['centroid'][1],prop['centroid'][0],marker='o')
#            minr, minc, maxr, maxc = prop['BoundingBox']
#            bx = (minc, maxc, maxc, minc, minc)
#            by = (minr, minr, maxr, maxr, minr)
#            plt.plot(bx, by, '-b', linewidth=2.5)
#            plt.hold(True)
#        plt.hold(False)
    
    ch_im_list = split_im_regions(im, regions, mean_height)
    
    if(params["TEST_MODE"]["char_detect"]): 
        for img in ch_im_list:
            plt_i(img)
        
    return ch_im_list
#plt.figure(), plt.imshow(imb_neg,cmap='gray')
def get_all(params):
    '''
    Analyzes an image and returns a list where every element is a
    word conaining in a form of list of images of letters.
    *BETA 1. BY THE MOMENT IT ONLY RETURNS THE IMAGES OF LETTERS*
    '''
    im = color.rgb2gray(io.imread("./img/"+params["image_name"]))
    plt_i(im)
    plt.title('original');
    
#    im = neutre(im)
#    plt_i(im)
#    plt.title('neutre');
    
    im = resizing(im, params['contour'])
    plt_i(im)
    plt.title('resized')
    
    im = apply_threshold(im)
    plt_i(im)
    plt.title('thresholded');
#    plt.close('all')    
    im = noise_removal(im)
    plt_i(im)
    plt.title('noise romved');
    
    
#    imb, mean_height = close_vert_median(imb)
    label_image = label(im)
    regions = regionprops(label_image)
    height, width = get_means(regions)
    
    lines = get_lines(params, im, height)
    
    for i in range(len(lines)):
        try:
            words = get_words_from_line(params, lines[i])
            for j in range(len(words)):
                try: 
                    letters = get_letters(params, words[j])
                    words[j] = letters
                except ValueError:
                    del words[j]
                    pass
        except ValueError:
            pass
        lines[i] = words        

    return lines




















