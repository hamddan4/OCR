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
        "image_name": "rot14.png",
        
        "contour" : None,
        
        "status_msg" : True
    }


def neural_predict(im,model):    
    im = im / np.max(im)        
    try:
        im = utils.rescale(im.astype(float),(64,64),4)
        im = np.reshape(im,(1,64,64,1))
    
        letter = trans_table[net.net_predict(im,model)]
        return letter
    except: 
        return "?"
    
def process_img(im, params=global_params):
    #    if(params["new_net"]):
#        nt.net_train()
    plt.close('all') 
    
    if(len(im.shape) != 2):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        
    if(params["status_msg"]): print "Starting new request"
    
    im = im_treatement.treatement(im, params) 

    im = morf.apply_morf(im, params)

    im = rotation.fix_rotation(im, params)
    
    lines = gc.get_all(im, params)
    
    if(params["status_msg"]): print "Predicting results..."
    
    model = net.loadmodel('modelRetrainedRetrained')
    text = ""
    for line in lines:
        for word in line:
            for letter in word:
                result = neural_predict(letter,model)
                text += result
            text += " "
        text += "\n"
    if(params["status_msg"]): print "Finished all operations."
    return text

def main():
    global lines
    im = io.imread("./img/"+global_params["image_name"])

    text = process_img(im, global_params)
    
    f = open("test.txt","w") 
    f.write(text)
    f.close()
    
if __name__ == "__main__":
    main()
    