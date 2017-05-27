# -*- coding: utf-8 -*-

"""

@author: Dani, richard
"""

from __future__ import print_function
import string

import numpy as np

import keras
import keras.initializers as inits

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop,Adadelta,SGD
from keras.models import model_from_json


batch_size = 128

num_classes = 62

epochs = 15 

def loadmodel(namefile):
    file = open(namefile+'.json','r')
    json_string = ''
    
    for line in file:
        json_string += line
    
    model = model_from_json(json_string)
    model.load_weights(namefile+'config')
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',    
                  optimizer=sgd,    
                  metrics=['accuracy'])  
    return model


def savemodel(namefile,model):
        
    json_model = model.to_json()    
    file = open(namefile+'.json','w')
    file.write(json_model)    
    file.close()

    model.save_weights(namefile+'config')
    
    
def net_predict(im_char,model):
    prediction = model.predict(np.array([im_char][0]))
      
    char_predicted = np.argmax(prediction,axis=1)[0]
    
    return char_predicted
    
def data_split_shuffle(x,y,percent):
    
    randlist = np.arange(0,np.shape(x)[0])
    np.random.shuffle(randlist)
    
    x = x[randlist]
    y = y[randlist]
    
    x_train = x[1:np.shape(x)[0]*percent]
    x_train = np.reshape(x_train,(np.shape(x_train)[0],64,64,1))    
    x_test = x[np.shape(x)[0]*percent:-1]
    x_test = np.reshape(x_test,(np.shape(x_test)[0],64,64,1))
    
    y_train = y[1:np.shape(y)[0]*percent]    
    y_test = y[np.shape(y)[0]*percent:-1]
    
    return x_train,y_train,x_test,y_test
    
def net_train(x,y,augmentation):
        
    x /= 255
    y = keras.utils.to_categorical(y, num_classes)
    
    x_train,y_train,x_test,y_test = data_split_shuffle(x,y,0.7)
    
    print(x_train.shape[0], 'train samples')    
    print(x_test.shape[0], 'test samples')
    
    if(augmentation):
        datagen = ImageDataGenerator(
            rotation_range = 20,
            width_shift_range = 0.15,
            height_shift_range = 0.15,
            shear_range = 0.4,
            zoom_range = 0.3)    
    
    #builds the neural network layers
        
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    
    model.summary()
    
    
#    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',    
                  optimizer='adamax',    
                  metrics=['accuracy'])
    
    
    if(augmentation):
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            samples_per_epoch=len(x_train),
                            epochs=epochs, 
                            validation_data=(x_test, y_test),
                            verbose=1)
    else:
        history = model.fit(x_train, y_train,    
                        batch_size=batch_size,    
                        epochs=epochs,    
                        verbose=1,    
                        validation_data=(x_test, y_test))
    
    savemodel('model',model)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    
    print('Test accuracy:', score[1])
    
    return history
    
    
def net_retrain(x,y,namemodel,augmentation):
    
    model = loadmodel(namemodel)
    
    x /= 255
    y = keras.utils.to_categorical(y, num_classes)
    
    x_train,y_train,x_test,y_test = data_split_shuffle(x,y,0.7)
    
    print(x_train.shape[0], 'train samples')    
    print(x_test.shape[0], 'test samples')    
    
    model.compile(loss='categorical_crossentropy',    
              optimizer='adamax',    
              metrics=['accuracy'])
    
    if(augmentation):
        datagen = ImageDataGenerator(
            rotation_range = 20,
            width_shift_range = 0.15,
            height_shift_range = 0.15,
            shear_range = 0.4,
            zoom_range = 0.3)  
            
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            samples_per_epoch=len(x_train),
                            epochs=epochs, 
                            validation_data=(x_test, y_test),
                            verbose=1)
    else:
        history = model.fit(x_train, y_train,    
                        batch_size=batch_size,    
                        epochs=epochs,    
                        verbose=1,    
                        validation_data=(x_test, y_test))
        
        
    return history
    
    
    
    