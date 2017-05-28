from utils import plt_i
import numpy as np
import cv2

from utils import oddNum, notZero

def resizing(im, contour):
    """
    Funcio que escala la imatge. Util per reduir el temps total del proces.
    * Inputs:
    - im = skimage.io image
    - params = diccionari de parametres
    *Outputs:
    - im = imatge escalada
    """
    
    im2 = im
    if(contour != None):
        x, y = im.shape
        dismx = x / float(contour)
        dimx = int(x / dismx)
        dimy = int(y / dismx)
        im2 = cv2.resize(im, (dimy, dimx)) 
    return im2    
      
def neutre(im, params):  
    """
    Funcio que neutralitza el color de la imatge. Util quan es esta tacat o hi
    ha variancies.
    * Inputs:
    - im = skimage.io image
    - params = diccionari de parametres
    *Outputs:
    - im = imatge amb color neutralitzat
    """
    
    [x, y] = im.shape
    sz = notZero(x/85)
    if (sz < 5):
        sz = 5
    kernel = np.ones((sz,sz),np.uint8)
    resd = cv2.erode(cv2.dilate(np.int16(im),kernel,1),kernel,1)
    if(params["TEST_MODE"]["im_treatement"]): 
        plt_i(resd, "closed for neutralization")
        
    resd = np.array(resd, dtype=np.float)
    im = np.divide(im,resd)
    return im

def noise_removal(im, params):
    """
    Utilitzant la tecnica NMeans, elimina el soroll de la imatge
    * Inputs:
    - im = skimage.io image
    - params = diccionari de parametres
    *Outputs:
    - im = imatge sense soroll
    """
    
    im_denoised = cv2.fastNlMeansDenoising(im)

    return im_denoised
    
def treatement(im, params):
    """
    Funcio que aplica tots el procedimens de millora de la cualitat
    de la imatge
    * Inputs:
    - im = skimage.io image
    - params = diccionari de parametres
    *Outputs:
    - im =  resultat de la imatge un cop ha pasat per tots els processos de
            millora d'imatge
    """
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