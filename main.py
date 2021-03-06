# -*- coding: utf-8 -*-
"""

@author: Dani, richard
"""

#import net_training as nt

import get_chars as gc
import string 
import net_training as net
import numpy as np
import utils 
reload(gc)

import matplotlib.pyplot as plt
from skimage import io
import cv2

import rotation
reload(rotation)
import im_treatement
reload(im_treatement)
import morf
reload(morf)
import sys

lines = []
trans_table = list(string.digits)+list(string.ascii_uppercase)+list(string.ascii_lowercase)

global_params = {
        "TEST_MODE": {
            "line_detect":False,
            "word_detect":False,
            "char_detect":False,
            
            "im_treatement":False,
            
            "rotation":False,
        },
        
        "new_net":False,
        "image_name": "sign.jpg",
        
        "contour" : 1200,
        
        "status_msg" : True
    }


def neural_predict(im,model):    
    """
    Funcio que passa la imatge a la xarxa neuronal i retorna la lletra predita.
    * Inputs:
    - im = skimage.io image
    - model = model neuronal
    *Outputs:
    - letter = caracter predit
    """
    im = im / np.max(im)        
    try:
        im = utils.rescale(im.astype(float),(64,64),4)
        im = np.reshape(im,(1,64,64,1))
    
        letter = trans_table[net.net_predict(im,model)]
        return letter
    except: 
        return "?"
    
def process_img(im, params=global_params):
    """
    Funció que fa tot el procediment per obtenir el text de la imatge.
    * Inputs:
    - im = skimage.io image
    - params = diccionari de parametres
    *Outputs:
    - text = String amb el text trobat a la imatge
    """
    plt.close('all') 
    
    if(len(im.shape) != 2):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        
    if(params["status_msg"]): print "Starting new request"
    
    im = im_treatement.treatement(im, params) 

    im = morf.apply_morf(im, params)

    im = rotation.fix_rotation(im, params)
    
    lines = gc.get_all(im, params)
    
    if(params["status_msg"]): print "Predicting results..."
    if(params["status_msg"]): print "====================="
    
    model = net.loadmodel('model')
    text = ""
    for line in lines:
        for word in line:
            for letter in word:
                result = neural_predict(letter,model)
                if(params["status_msg"]): sys.stdout.write(result)
                text += result
            if(params["status_msg"]): sys.stdout.write(" ")
            text += " "
        if(params["status_msg"]): sys.stdout.write("\n")
        text += "\n"
        
    if(params["status_msg"]): print "========================"
    if(params["status_msg"]): print "Finished all operations."
    return text

def main():
    """
    Benvingut a l'OCR!
    Canvia la imatge dessitgada al parametre "image_name" de la variable 
    global_param i assegurat que aquesta imatge esta a la carpeta ./img
    
    Aquesta funcio et guardara a text.txt el resultat de la compilacio. Vagi
    de gust!
    
    """
    global lines
    im = io.imread("./img/"+global_params["image_name"])

    text = process_img(im, global_params)
    
    f = open("test.txt","w") 
    f.write(text)
    f.close()
    
if __name__ == "__main__":
    main()
    