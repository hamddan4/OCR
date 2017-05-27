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

lines = []
trans_table = list(string.digits)+list(string.ascii_uppercase)+list(string.ascii_lowercase)

def neural_predict(im,model):    
    im = im / np.max(im)    
    
    im = utils.rescale(im.astype(float),(64,64),4)
    gc.plt_i(im)
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
            
            "im_treatement":False,
        },
        
        "new_net":False,
        "image_name": "Practica5-1.png",
        
        "contour" : 2200
    }
   
    max_plots = 0
#    if(params["new_net"]):
#        nt.net_train()
    plt.close('all')        
    lines = gc.get_all(params)
    
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
    