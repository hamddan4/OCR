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



def neural_predict(im,model):    
    im = im / np.max(im)    
    
    im = utils.rescale(im.astype(float),(64,64),4)
    im = np.reshape(im,(1,64,64,1))
    
    letter = trans_table[net.net_predict(im,model)]
    return letter
#    return "x"
    
def main():
    global lines
    params = {
        "TEST_MODE": {
            "line_detect":False,
            "word_detect":False,
            "char_detect":False,
            
            "im_treatement":True,
            
            "rotation":True,
        },
        
        "new_net":False,
        "image_name": "Practica5-1.png",
        
        "contour" : None
    }
   
#    if(params["new_net"]):
#        nt.net_train()
    plt.close('all') 
    
    im = io.imread("./img/"+params["image_name"])
    if(len(im.shape) != 2):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    
    im = rotation.fix_rotation(im, params)
    im = im_treatement.treatement(im, params)    
    im = morf.apply_morf(im, params)
    
    lines = gc.get_all(im, params)
    
    f = open("test.txt","w") 
    model = net.loadmodel('model')
    
    for line in lines:
        for word in line:
            for letter in word:
                result = neural_predict(letter,model)
                f.write(result)
            f.write(" ")
        f.write("\n")
    f.close()

    
if __name__ == "__main__":
    main()
    