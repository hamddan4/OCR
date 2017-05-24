# -*- coding: utf-8 -*-

'''

'''

from __future__ import print_function
import string

import numpy as np

import keras
import keras.initializers as inits

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop,Adadelta

from keras.models import model_from_json


batch_size = 256

num_classes = 62

epochs = 30 

translation_table = range(0,10)+list(string.ascii_uppercase)+list(string.ascii_lowercase)

def __loadmodel(namefile):
    file = open(namefile+'.json','r')
    json_string = ''
    
    for line in file:
        json_string += line
    
    model = model_from_json(json_string)
    model.load_weights(namefile+'config')
    
    model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    
    return model


def __savemodel(namefile,model):
        
    json_model = model.to_json()    
    file = open(namefile+'.json','w')
    file.write(json_model)    
    file.close()

    model.save_weights(namefile+'config')
    
    
def net_predict(im_char):
    model = __loadmodel('model')
    
    prediction = model.predict(np.array([im_char])) #habra que revisar este rollo de los corchetes en el futuro
      
    char_predicted = np.argmax(prediction,axis=1)[0]
    
    return char_predicted
    
    
def net_train(x,y):
    
    # the data, shuffled and split between train and test sets
    
#    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    
    
    x /= 255
    
    y = keras.utils.to_categorical(y, num_classes)
    
    randlist = np.arange(0,np.shape(x)[0])
    np.random.shuffle(randlist)
    
    x = x[randlist]
    y = y[randlist]
    
    x_train = x[1:np.shape(x)[0]*0.7]
    
    x_test = x[np.shape(x)[0]*0.7:-1]
    
    y_train = y[1:np.shape(y)[0]*0.7]
    
    y_test = y[np.shape(y)[0]*0.7:-1]
    
#    x_train = x_train.reshape(60000, 784)
#    
#    x_test = x_test.reshape(10000, 784)
    
#    x_train = x_train.astype('float32')
#    
#    x_test = x_test.astype('float32')
    
#    x_train /= 255
#    
#    x_test /= 255
    
    print(x_train.shape[0], 'train samples')
    
    print(x_test.shape[0], 'test samples')
    
    
    
    # convert class vectors to binary class matrices
    
    

    
    
    
    model = Sequential()
    
    model.add(Dense(1024, activation='relu', input_dim=4096,kernel_initializer=inits.he_normal()))
    model.add(Dropout(0.2))
    
    model.add(Dense(256, activation='relu',kernel_initializer=inits.he_normal()))    
    model.add(Dropout(0.2))
    
    model.add(Dense(1024, activation='relu',kernel_initializer=inits.he_normal()))    
    model.add(Dropout(0.2))
    
    
    model.add(Dense(62, activation='softmax',kernel_initializer=inits.he_normal()))
    
    
    
    model.summary()
    
    
    
    model.compile(loss='categorical_crossentropy',
    
                  optimizer=Adadelta(),
    
                  metrics=['accuracy'])
    
    
    
    history = model.fit(x_train, y_train,
    
                        batch_size=batch_size,
    
                        epochs=epochs,
    
                        verbose=1,
    
                        validation_data=(x_test, y_test))
    
    
    __savemodel('model',model)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    
    print('Test accuracy:', score[1])