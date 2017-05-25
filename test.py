# -*- coding: utf-8 -*-
"""
Created on Wed May 24 19:30:58 2017

@author: richa
"""

import net_training
import get_chars as gt
from keras.models import Model

model = net_training.__loadmodel('model')
layer_name = 'dense_16'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict( np.array([x[0]]))