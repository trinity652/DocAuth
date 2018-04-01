#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 01:32:55 2018

@author: abhilasha
import network
import preprocessor
"""

import cv2
import os
import numpy as np




current_dir = os.path.dirname(__file__)

author = '021'
training_folder = os.path.join(current_dir, 'data/training/', author)
test_folder = os.path.join(current_dir, 'data/test/', author)

training_data = []
for filename in os.listdir(training_folder):
    img = cv2.imread(os.path.join(training_folder, filename), 0)
    if img is not None:
        data = np.array(preprocessor.prepare(img))
        data = np.reshape(data, (901, 1))
        result = [[0], [1]] if "genuine" in filename else [[1], [0]]
        result = np.array(result)
        result = np.reshape(result, (2, 1))
        training_data.append((data, result))

test_data = []
for filename in os.listdir(test_folder):
    img = cv2.imread(os.path.join(test_folder, filename), 0)
    if img is not None:
        data = np.array(preprocessor.prepare(img))
        data = np.reshape(data, (901, 1))
        result = 1 if "genuine" in filename else 0
        test_data.append((data, result))

net = network.NeuralNetwork([901, 500, 500, 2])
net.sgd(training_data, 10, 50, 0.01, test_data)
print(net.evaluate(test_data))
