#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 04:36:37 2018

@author: abhilasha
"""

import cv2 
import numpy as np
import pywt

im_gray = cv2.imread("Ac2.png", cv2.IMREAD_GRAYSCALE)
im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
cv2.imwrite( 'test.png', im_color)
def w2d(img, mode='haar', level=1):
    imArray = cv2.imread(img)
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)
    #Display result
    return imArray_H
    #cv2.imwrite('dct', imArray_H)
    

iparray=w2d("Ac2.png",'db1',8)
cv2.imwrite('dct.png',iparray)
cv2.waitKey(0)
cv2.destroyAllWindows()