from utils import plt_i
import numpy as np
import cv2

def resizing(im, contour):
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
        plt_i(resd, "closed for neutralization")
        
    resd = np.array(resd, dtype=np.float)
    im = np.divide(im,resd)
    return im

def noise_removal(im, params):
    
    blur = cv2.GaussianBlur(im,(9,9),0)
    ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    imb = (im + th2)
    return imb
    
def treatement(im, params):
    
    if(params["status_msg"]): print "Starting im_treatement: Resizing"
    
    im = np.array(im, dtype=np.uint8)
    
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im, "original")
    
    im = resizing(im, params["contour"])
    
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im, "resized")
    if(params["status_msg"]): print "Resizing OK. Next step: noise removal"
    
    im = noise_removal(im, params)
    
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im,'Otsu thresholding + Gaussian filtering \n (noise removed)') 
    if(params["status_msg"]): print "Noise Removal OK. Next step: Neutralizing img"
    
    im = neutre(im, params)
    im = np.nan_to_num(im)
    
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im, "neutre")
    
    if(params["status_msg"]): print "Neutre OK. im_treatement Done"
    
    return im