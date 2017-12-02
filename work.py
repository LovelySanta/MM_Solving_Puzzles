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

def mirror(image,op):
    if op == 'H':
        image = cv2.flip(image, 0)
    elif op == 'V': 
        image = cv2.flip(image, 1)
    elif op == 'D':
        image = cv2.transpose(image)
    return image   

def getMatch(e1, e2):
    dif = len(e1)-len(e2)
    if abs(dif) > 5:
        return 50000
    elif dif == 0:
        #cv2.imshow('diff', abs(cv2.absdiff(e2, e1))[:,0])
        
        return sum(sum(abs(cv2.absdiff(e2, e1)*0.1)))*100.0/len(e1)
    elif dif > 0:
        match = 50000
        for i in range(dif):
            match = min(match, sum(sum(cv2.absdiff(e2, e1[i:i+len(e2)])*0.1)))
        return match*100.0/len(e2)
    else:
        match = 50000
        for i in range(0-dif):
            match = min(match, sum(sum(cv2.absdiff(e1, e2[i:i+len(e1)])*0.1)))
        return match*100.0/len(e1)
    
def getedge(piece, edge):
    if type(edge) == type(5):
        sides = ["N","E","S","W"]
        side = sides[edge%4]
    else:
        side = edge
    if side == "W":
        return(piece[:,0, ])
    elif side == "N":
        return(piece[0,:, ])
    elif side == "E":
        return(piece[:,-1, ])
    elif side == "S":
        return(piece[-1,:, ])

    
def showMatch(pieces, matchid): 
    i,j,k,l = matchid
    im = rotatePiece(pieces[i], k)
    im2 = rotatePiece(pieces[j], (l+2)%4)
    im = np.append(im2, im, axis = 0)
    cv2.imshow('piece'+str(i)+str(j), im)

def rotatePiece(piece, i):
    if i == 0: #North side
        return piece
    elif i == 1: #East side
        return mirror(mirror(piece, 'D'), 'H')
    elif i == 2:
        return mirror(mirror(piece, 'H'), 'V')
    elif i == 3:
        return mirror(mirror(piece, 'H'), 'D')
        
    
def getBestMatch(pieces):
    sides = ["N","E","S","W"]
    bestmatch = 50000
    matchid = [0,0,0,0]
    matches = np.empty((len(pieces), len(pieces), len(sides), len(sides)), dtype = 'uint16')
    for i in range(len(pieces)):
        for k in range(len(sides)):
            e1 = getedge(pieces[i],sides[k])
            for j in range(i, len(pieces)):
                if not(i == j):
                    for l in range(len(sides)):
                        match = int(getMatch(e1, getedge(pieces[j],sides[l])))
                        matches[i,j,k,l]= match
                        matches[j,i,l,k]= match
                        if match < bestmatch:
                            bestmatch = match
                            matchid = [i,j,k,l]
                            rev = [j,i,l,k]
#                            cv2.waitKey()
#                            cv2.destroyAllWindows()
                else:
                    matches[i,j,k]= list(50000 for l in range(len(sides)))
    print bestmatch
    print(matchid)
    showMatch(pieces, matchid)
    showMatch(pieces, rev)
    print matches[matchid[1], matchid[0], matchid[3], matchid[2]]
#    print matches[rev]
    return matches, matchid
            
def getNeighbour(pieces, matches, pieceNmr, sideNmr):
    matchid = [0,0,0,0]
    bestmatch = 50000
    for i in range(len(matches)):
        if (not i == pieceNmr):
            for j in range(len(matches[0,0,0])):
                match = matches[i, pieceNmr, j, sideNmr]
                if match < bestmatch:
                    bestmatch = match
                    matchid = [i, pieceNmr, j, sideNmr]
                    showMatch(pieces, matchid)

                    cv2.waitKey()
                    cv2.destroyAllWindows()
    print bestmatch
    print matchid
    return matchid

def getLine(pieces, matches, matchid):
    length = 2
    new_edge = matchid[0, 2]
    old_edge = matchid[1, 3]
    old_match = getNeighbour(matches, old_edge)
    line = list(0 for i in range(np.floor(np.sqrt(len(matches)))))
    line[0] = matchid
#    while length < len(line):
        
    

puzzleType="tiles"
puzzleArrangement="rotated"
puzzleSize="5x5"
puzzleNumber="04"        
puzzle = cv2.imread('images/'+puzzleType+'/'+puzzleType+'_'+puzzleArrangement+'/'+puzzleType+'_'+puzzleArrangement+'_'+puzzleSize+'_'+puzzleNumber+'.png')
pieces = dividePuzzle(puzzle, 25)

sides = ["N","E","S","W"]
matches, matchid = getBestMatch(pieces)
match2 = getNeighbour(pieces, matches, matchid[0], (matchid[2]+2)%4)
match3 = getNeighbour(pieces, matches, matchid[1], (matchid[3]+2)%4)
showMatch(pieces, match2)
showMatch(pieces, match3)
cv2.waitKey()
cv2.destroyAllWindows()

#cv2.imshow('im', puzzle)
#cv2.waitKey()
#cv2.destroyAllWindows()






























