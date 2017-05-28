# -*- coding: utf-8 -*-
"""

@author: Dani, richard
"""

import dataset
import net_training
import matplotlib.pyplot as plt

x,y = dataset.getImagesDataset()

history = net_training.net_train(x,y,True)
#history = net_training.net_retrain(x,y,'modelRetrained',True)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
