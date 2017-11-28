# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:43:09 2017

@author: jarno
"""
import numpy as np

def dividePuzzle(aantal):
    img = #getimage
    size = np.floor(np.sqrt(aantal))
    for i in range(0, size):
        for j in range(0, size):
            #cutoutpiece
            piece = img[(i*len(img))/size:((i+1)*len(img))/size-1,(j*len(img[0]))/size:((j+1)*len(img[0]))/size-1]
            #addimage();