from utils import plt_i
import numpy as np
import cv2

from utils import oddNum

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
    im_denoised = cv2.fastNlMeansDenoising(im)

    return im_denoised
    
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
        plt_i(im,'Noise removed') 
    if(params["status_msg"]): print "Noise Removal OK. Next step: Neutralizing img"
    
    im = neutre(im, params)
    im = np.nan_to_num(im)
    
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(im, "neutre")
    
    if(params["status_msg"]): print "Neutre OK. im_treatement Done"
    
    return im