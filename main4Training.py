# -*- coding: utf-8 -*-
"""

@author: Dani, richard
"""

import dataset
import net_training

x,y = dataset.getImagesDataset()

net_training.net_train(x,y)
