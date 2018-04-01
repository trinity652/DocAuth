#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 02:41:38 2018

@author: abhilasha
"""

import cv2
import numpy as np

img = cv2.imread('Ac2.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
_,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) # threshold
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
dilated = cv2.dilate(thresh,kernel,iterations = 13) # dilate
_, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours


idx =0
# for each contour found, draw a rectangle around it on original image
for contour in contours:

    idx += 1

    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)

    # discard areas that are too large
    if h>300 and w>300:
        continue

    # discard areas that are too small
    if h<10 or w<10:
        continue

    # draw rectangle around contour on original image
    cv2.imwrite('redboxes.png',cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2))

    roi = img[y:y + h, x:x + w]

    cv2.imwrite( str(idx) + '.png', roi)

    cv2.imshow('img',roi)
    
