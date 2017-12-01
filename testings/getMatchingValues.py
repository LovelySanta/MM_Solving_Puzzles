# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:43:09 2017

@author: jarno
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def dividePuzzle(img, aantal):
    #img = #getimage
    size = int(np.floor(np.sqrt(aantal)))
    pieces = []
    for i in range(0, size):
        for j in range(0, size):
            #cutoutpiece
            piece = img[(i*len(img))/size:((i+1)*len(img))/size,(j*len(img[0]))/size:((j+1)*len(img[0]))/size]
            pieces.append(piece)
            #addimage();
    return pieces

def getMatch(e1, e2):
    dif = len(e1)-len(e2)
    if abs(dif) > 5:
        return 10000
    elif dif == 0:
        cv2.imshow('diff', abs(cv2.absdiff(e2, e1))[:,0])

        return sum(sum(abs(cv2.absdiff(e2, e1)*1.0)))*256.0/len(e1)
    elif dif > 0:
        match = 10000*len(e2)
        for i in range(dif):
            match = min(match, sum(sum(cv2.absdiff(e2, e1[i:i+len(e2)])*1.0)))
        return match*1.0/len(e2)
    else:
        match = 10000*len(e1)
        for i in range(0-dif):
            match = min(match, sum(sum(cv2.absdiff(e1, e2[i:i+len(e1)])*1.0)))
        return match*1.0/len(e1)

def getedge(piece, edge):
    if edge == "W":
        return(piece[:,0, ])
    if edge == "N":
        return(piece[0,:, ])
    if edge == "E":
        return(piece[:,-1, ])
    if edge == "S":
        return(piece[-1,:, ])

def showPiece(pieces, i):
    cv2.imshow('piece'+str(i), pieces[i])


def getBestMatch(pieces):

    sides = ["N","E","W","S"]
    bestmatch = 10000
    matchid = [0,0,0,0]
    matches = np.empty((len(pieces), len(pieces), len(sides), len(sides)), dtype = 'uint16')
    for i in range(len(pieces)):
        for j in range(i, len(pieces)):
            for k in range(len(sides)):
                for l in range(len(sides)):
                    if i == j:
                        match = 10000
                    else:
                        match = int(getMatch(getedge(pieces[i],sides[k]), getedge(pieces[j],sides[l])))
                    matches[i,j,k,l]= match
                    if match < bestmatch:
                        bestmatch = match
                        showPiece(pieces, i)
                        showPiece(pieces, j)
                        print match
                        matchid = [i,j,k,l]
                        print(matchid)
                        cv2.waitKey()
                        cv2.destroyAllWindows()

    return matches, matchid




puzzleType="tiles"
puzzleArrangement="shuffled"
puzzleSize="5x5"
puzzleNumber="04"
puzzle = cv2.imread('images/'+puzzleType+'/'+puzzleType+'_'+puzzleArrangement+'/'+puzzleType+'_'+puzzleArrangement+'_'+puzzleSize+'_'+puzzleNumber+'.png')

pieces = dividePuzzle(puzzle, 25)

#e = getedge(pieces[5],"W")
#e2 = getedge(pieces[13],"W")
#test = getMatch(e,e2)
#showPiece(pieces, 5)
#showPiece(pieces, 13)
#cv2.waitKey()
#cv2.destroyAllWindows()

sides = ["N","E","W","S"]
bestmatch = float("inf")
matches, matchid = getBestMatch(pieces)

showPiece(pieces, matchid[0])
showPiece(pieces, matchid[1])
print(matchid)

print(matches[matchid[0]][matchid[1]][matchid[2]][matchid[3]])
cv2.waitKey()
cv2.destroyAllWindows()

#cv2.imshow('im', puzzle)
#cv2.waitKey()
#cv2.destroyAllWindows()
