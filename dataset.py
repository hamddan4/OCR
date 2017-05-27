# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:30:56 2017

@author: richa
"""
import cv2
import os
from PIL import Image
import numpy as np

#read the dataset
def getImagesDatasetFromFile():
    pass

def getImagesDataset():
    print("#################################")    
    print("Getting data from dataset:")    
    x = []
    y = []
    iterator = os.walk('./machine_written/English/Fnt')
    iterator.next()
    i = 0
    for folder_content in iterator:
        folder_path = folder_content[0] 
        y.append((len(folder_content[2])/4)*[i])
        i+=1
        print(str(i)+" de 62")
        half_folder = folder_content[2][0:(len(folder_content[2])/4)]
        for image in half_folder:   
            im = Image.open(folder_path+'/'+image)
            x.append(np.array(im.getdata(),dtype='float32'))
            im.close()
            
    x = np.array(x)
    y = np.reshape(y,np.size(y))
    print("#################################")   
    return x,y
  
#functions used to resize the dataset  
def resizeImage(filename,sizeTup):
    im = Image.open(filename)
    #_im = np.reshape(np.array(im.getdata(),dtype='float32'),(128,128)) 

    imrs = cv2.resize(im,sizeTup)
    
    imsv = Image.fromarray(imrs.astype(np.uint8),mode="L")
    imsv.save(filename,"PNG")
    
def resizeDir():
    iterator = os.walk('./machine_written/English/Fnt')
    iterator.next()
    i = 0
    for folder_content in iterator:
        folder_path = folder_content[0] 
        i+=1
        half_folder = folder_content[2][0:(len(folder_content[2]))]
        for image in half_folder:   
            resizeImage(folder_path+'/'+image,(64,64))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    