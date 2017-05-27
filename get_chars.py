
"""
Created on Wed May 10 18:33:44 2017

@author: richa
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, filters, io
from skimage.morphology import label, remove_small_holes, remove_small_objects
from skimage.measure import regionprops

from utils import oddNum

import cv2

import scipy.ndimage as nd

max_plots = 0
def plt_i(im,title=None):
    global max_plots
    max_plots += 1
    if(max_plots < 20):
        plt.figure(), plt.imshow(im,cmap=plt.cm.gray)
    else:
        raise Exception("Too many plots are open! I might just saved your ass here...")
#        pass
    if not(title == None):
        plt.title(title)

def plt_s(regions):
    for prop in regions:
        plt.plot(prop['centroid'][1],prop['centroid'][0],marker='o')
        minr, minc, maxr, maxc = prop['BoundingBox']
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        plt.plot(bx, by, '-b', linewidth=2.5)
        plt.hold(True)
    plt.hold(False)
    
def resizing(im, params):
    contour = params["contour"]
    im2 = im
    if(contour != None):
        x, y = im.shape
        dismx = x / float(contour)
        dimx = int(x / dismx)
        dimy = int(y / dismx)
        im2 = cv2.resize(im, (dimy, dimx)) 
    return im2    
      
def neutre(im, params):  
    [x, y] = im.shape
    kernel = np.ones((x/85,x/85),np.uint8)
    resd = cv2.erode(cv2.dilate(np.int16(im),kernel,1),kernel,1)
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(resd)
        plt.title('closed for neutralization');
    resd = np.array(resd, dtype=np.float)
    im = np.divide(im,resd)
    return im

def get_h_w(region):
    minr, minc, maxr, maxc = region['BoundingBox']
    height = int(maxc-minc)
    width = int(maxr-minr)
    return height, width
    
def get_lists_regions(regions):
    height = []
    width = []
    for prop in regions:      
        minr, minc, maxr, maxc = prop['BoundingBox']
        height.append(int(maxc-minc))
        width.append(int(maxr-minr))
    
    return height, width
def get_means(regions):
    
    height, width = get_lists_regions(regions)
    
    m_height = np.median(height);
    m_width = np.median(width)
    return m_height, m_width

    
def split_im_regions(im, regions, mean_height):
    im_list = []
    for i in range(len(regions)):
        minr, minc, maxr, maxc = regions[i]['BoundingBox']
        charac = im[(minr):(maxr+1),(minc):(maxc+1)]
        im_list.append(charac)
        
    return im_list
    
def apply_threshold(im,params):
    x, y = im.shape
    print im
    ws = oddNum(x / 10)
    
    sauv = filters.threshold_sauvola(im, window_size = ws)
    sauv = np.nan_to_num(sauv)
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(sauv) 
        plt.title('sauvola')
    
    res = np.greater(im,sauv)
    #kernel = np.ones((6,1),np.uint8)
    
    return res

def noise_removal(im, params):
    
    blur = cv2.GaussianBlur(im,(9,9),0)
    ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    imb = (im + th2)
    return imb
    
def del_big_spots(im, params):
    #Fem un close per detectar regions o taques massa grans
    im = np.array(im, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
    imb = cv2.erode(im, kernel,iterations=1)
    
    label_image = label(1-imb)
    regions = regionprops(label_image)
    
    areas = np.array([x.area for x in regions])
#    m_area = np.mean(areas)
    thr = np.mean(areas)*6 + np.std(areas)
    
    im2 = np.uint8(remove_small_holes(imb,  thr))
    res= np.array(np.array((1-im2 + im), dtype=bool), dtype=np.uint8)
    
    plt_i(im,"im")
    plt_i(imb,"imb")
    plt_i(im2,"imb2")
    plt_i(res, "res")
#    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
#    im3 = cv2.erode(im3,kernel,1)
    return res

def del_small_spots(im, params):
    #Fem un close per detectar regions o taques massa grans

    
#    im = 1-im
    im = np.array(im, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1))
    imb = cv2.erode(im, kernel)
    
    label_image = label(1-imb)
    regions = regionprops(label_image)
    
    areas = np.array([x.area for x in regions])
    m_area = np.mean(areas)

    im2 = np.uint8(remove_small_holes(imb,  m_area/6))
#    im2 = np.uint8(remove_small_objects(imb,  moda_area * 3))

    return im2
    
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
    im = io.imread("./img/"+params["image_name"]);

    if(len(im.shape) != 2):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im)
        plt.title('original')
    
    im = resizing(im, params)
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im)
        plt.title('rezised')
    
    im = noise_removal(im, params)
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im)
        plt.title('Otsu thresholding + Gaussian filtering \n (noise removed)')    

    
    im = neutre(im, params)
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im)
        plt.title('neutre');    
    
    im = apply_threshold(im, params)
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im)
        plt.title('thresholded')
#    plt.close('all')    
    
#    im = np.uint8(im)*255
    
    im = del_big_spots(im, params)
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im)
        plt.title('big spots deleted')
            
    im = del_small_spots(im, params)
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im)
        plt.title('small spots deleted')
    
    
        
    imb, mean_height = close_vert_median(im)
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




















