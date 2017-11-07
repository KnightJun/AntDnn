# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 00:04:41 2017

@author: jun
"""

from keras.models import Sequential
from keras.layers import Conv2DTranspose

model = Sequential()
model.add(Conv2DTranspose(5, (8, 8),
                padding='valid',
                     input_shape=(32, 32, 3)))