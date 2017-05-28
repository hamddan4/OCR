# -*- coding: utf-8 -*-
from utils import plt_i, oddNum
import cv2
import numpy as np
from skimage import filters
from skimage.morphology import remove_small_holes, label
from skimage.measure import regionprops

def apply_threshold(im,params):
    x, y = im.shape
    ws = oddNum(x / 10)
    
    sauv = filters.threshold_sauvola(im, window_size = ws)
    sauv = np.nan_to_num(sauv)
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(sauv, "sauvola") 
    
    res = np.greater(im,sauv)
    #kernel = np.ones((6,1),np.uint8)
    
    return res
    
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
    
#    plt_i(im,"im")
#    plt_i(imb,"imb")
#    plt_i(im2,"imb2")
#    plt_i(res, "res")
    
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
    
def apply_morf(im, params):
    if(params["status_msg"]): print "Starting Morphology: Sauvola Thresholding"
    
    im = apply_threshold(im, params)
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im, "thresholded")
    if(params["status_msg"]): print "Threshold Done, next step: Big Spots Deletion"
    
    im = del_big_spots(im, params)
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im, "big spots deleted")
        
    if(params["status_msg"]): print "Big Spots Deleted, next step: Small Spot Deletion"
            
    im = del_small_spots(im, params)
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im, "small spots deleted")
    
    
    if(params["status_msg"]): print "Small Spots Deleted. Morphology Done"
    
    return im