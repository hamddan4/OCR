# -*- coding: utf-8 -*-
"""

@author: Dani, richard
"""

#import net_training as nt

import get_chars as gc
import string 
reload(gc)

import matplotlib.pyplot as plt

lines = []
trans_table = range(0,10)+list(string.ascii_uppercase)+list(string.ascii_lowercase)

def neural_predict():
    return "X"
def main():
    global lines
    params = {
        "TEST_MODE": {
            "line_detect":False,
            "word_detect":False,
            "char_detect":False
        },
        
        "new_net":False,
        "image_name": "brut.png",
        
        "threshold":128
    }
   
    max_plots = 0
#    if(params["new_net"]):
#        nt.net_train()
    plt.close('all')        
    lines = gc.get_all(params)
    
    f = open("test.txt","w") 
    
    for line in lines:
        for word in line:
            for letter in word:
                result = neural_predict()
                f.write(result)
            f.write(" ")
        f.write("\n")
    f.close()
    
if __name__ == "__main__":
    main()
    