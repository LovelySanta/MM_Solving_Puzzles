# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 14:57:39 2017

@author: jarno
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import random
from scipy.optimize import curve_fit
from datetime import datetime
random.seed(datetime.now())

def f(x, A, B):
    return A*x+B

def get_rand_color(size = 3):
    return [random.randint(1,255), random.randint(1,255), random.randint(1,255)]


def ransac(image, contour, x, y):
    print("\nCalculating ...")
    T_i = len(contour)/8
    teller = 0
    line = [0,1]
    finished = False
    i = random.randint(1,len(contour)-1)
    while(finished == False):
        line[0] = []
        line[1] = []
        teller = teller+1
        T_d = 0.01+0.01*np.floor(teller*16.0/len(contour))
        img = np.copy(image)
        ### Random start point on first try

        i = (i+17)%(len(contour)-1)
        x1 , y1 = contour[i][0]
        ### Get 2nd point on fixed distance in array
        if(i >= len(contour)/6):
            x2 , y2 = contour[i-len(contour)/6][0]
        else:
            x2 , y2 = contour[i+len(contour)/6][0]

        ### Get Rico of this line < 45 degrees
        if abs(x2-x1) > abs(y2-y1):
            # dy/dx
            r1 = (1.* (y2-y1))/(x2-x1)
            divide = "x"
        elif abs(y2-y1) > abs(x2-x1):
            # dx/dy
            r1 = (1.* (x2-x1))/(y2-y1)
            divide = "y"
        else:
            r1 = 2
            divide = "z"
        print("Rico = " + str(r1))

        ### For all other points, check matching ricos
        for k in range(len(contour)):
            if(divide == "y"):
                # dx/dy
                if contour[k][0][1]-y1 == 0:
                    r2 = r1 + T_d*2
                else:
                    r2 = (1.* (contour[k][0][0]-x1)) / (contour[k][0][1]-y1)
            elif(divide == "x"):
                # dy/dx
                if contour[k][0][0]-x1 == 0:
                    r2 = r1 + T_d*2
                else:
                    r2 = (1.* (contour[k][0][1]-y1)) / (contour[k][0][0]-x1)
            else:
                r2 = r1 + 2*T_d

            # Check d(rico) < T_d
            if(abs(r1-r2) < T_d):
                line[0] = np.append(line[0],contour[k][0][0])
                line[1] = np.append(line[1],contour[k][0][1])
                cv2.circle(img,tuple(contour[k][0]), 2, (255,255,255), 2)
            if(len(line[0]) > T_i):
                finished = True

    if (divide == "x"):
        A,B = np.polyfit(line[0], line[1], 1)
        hoek = np.rad2deg(np.arctan(A))
    else:
        A,B = np.polyfit(line[1], line[0], 1)
        hoek = 90-np.rad2deg(np.arctan(A))

    cv2.circle(img,tuple(contour[i][0]), 2, (255,0,0), 2)
    cv2.circle(img,(x2,y2), 2, (0,0,255), 2)

    cv2.imshow('im', img)

    return hoek


#im = cv2.imread('images/tiles/tiles_scrambled/tiles_scrambled_5x5_04.png')
im = cv2.imread('images/jigsaw/jigsaw_scrambled/jigsaw_scrambled_5x5_04.png')
im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

lut = np.ones(256, dtype = 'uint8')
lut[0] = 0
lut = lut
im_bw = cv2.LUT(im_g,lut)*255


cv2.imshow('im', im)
cv2.imshow('im_bw', im_bw)
cv2.waitKey()
cv2.destroyAllWindows()

cnts = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

#for i in range(len(cnts[0])-1):
#    cv2.line(im, tuple(cnts[0][i][0]), tuple(cnts[0][i+1][0]), (255,0,0), 2)
#cv2.imshow('im_bw', im)
#cv2.waitKey()
#cv2.destroyAllWindows()

for i in range(len(cnts)):
    pts = cnts[i]
    x,y,w,h = cv2.boundingRect(pts)
    if(h > len(im)/10 and w > len(im[0])/10):
        piece = im[y:y+h, x:x+w]
        angle = ransac(im, pts, x, y)
        hoek2 = angle

        empty_piece = np.zeros(piece.shape, dtype = 'uint8')
        piece = np.concatenate((empty_piece, piece, empty_piece))
        empty_piece = np.concatenate((empty_piece, empty_piece, empty_piece))
        piece = np.concatenate((empty_piece, piece, empty_piece), axis = 1)

        M = cv2.getRotationMatrix2D((len(piece[0])*0.5, len(piece)*0.5),hoek2,1)
        piece_r = cv2.warpAffine(piece,M,(len(piece[0]), len(piece)))
        color = get_rand_color()

    #    for j in range(len(approx)-1):
    #        cv2.line(piece_r, tuple(approx[j][0]), tuple(approx[j+1][0]), color, 2)
    #    cv2.line(piece_r, tuple(approx[0][0]), tuple(approx[j+1][0]), color, 2)

        cv2.imshow('piece', piece)
        cv2.imshow('piece_r', piece_r)
        cv2.waitKey()
        cv2.destroyAllWindows()



#objects = {(0,0,0) : ({'xl': 0, 'xu': 0, 'yl': 0, 'yu': 0})}
#connections = list()
#c1 = np.concatenate((np.zeros([len(im_bw),1], dtype = 'uint8'), im_bw[:,1:]), axis = 1)
#c2 = np.concatenate((np.zeros([1,len(im_bw[0])],dtype = 'uint8'), c1[1:]), axis = 0)
#c3 = np.concatenate((np.zeros([1,len(im_bw[0])], dtype = 'uint8'),im_bw[1:]), axis = 0)
#c4 = np.concatenate((c3[:,:-1], np.zeros([len(im_bw),1], dtype = 'uint8')), axis = 1)
#
#for row in range(len(im)):
#    for col in range(len(im[0])):
#        if(im[row][col][0] != 0):
#            w = (c1[row][col] != 0)
#            nw = (c2[row][col] != 0)
#            n = (c3[row][col] != 0)
#            ne = (c4[row][col] != 0)
#
#            if(ne):
#                color = im[row-1][col+1]
#                im[row][col] = color
#            elif(n):
#                color = im[row-1][col]
#                im[row][col] = color
#            elif(nw):
#                color = im[row-1][col-1]
#                im[row][col] = color
#            elif(w):
#                color = im[row][col-1]
#                im[row][col] = color
#            else:
#                color = get_rand_color()
#                objects[(color)] = {'xl': col, 'xu': col, 'yl': row, 'yu': row}
#
#for i in range(len(im_bw)):
#objects = {(5,5,5): 'try'}
