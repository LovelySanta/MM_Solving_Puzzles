# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:43:09 2017

@author: jarno
"""
import numpy as np

def dividePuzzle(aantal):
    #img = #getimage
    size = np.floor(np.sqrt(aantal))
    for i in range(0, size):
        for j in range(0, size):
            #cutoutpiece
            piece = img[(i*len(img))/size:((i+1)*len(img))/size-1,(j*len(img[0]))/size:((j+1)*len(img[0]))/size-1]
            #addimage();

def getMatch(e1, e2):
    dif = len(e1)-len(e2)
    if abs(dif) > 5:
        return float("inf")
    elif dif > 0:
        match = float("inf")
        for i in range(dif):
            match = min(match, sum(cv2.absdiff(e2, e1[i:i+len(e2)])))
        return match*1.0/len(e2)
    else:
        match = float("inf")
        for i in range(0-dif):
            match = min(match, sum(cv2.absdiff(e1, e2[i:i+len(e1)])))
        return match*1.0/len(e1)
    
def getNeighbours(orientated, tiles, pieces):
    
    for i in range(pieces):
        #if(!tiles):
        match = 0
        
        