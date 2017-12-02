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
<<<<<<< HEAD:work.py
        #cv2.imshow('diff', abs(cv2.absdiff(e2, e1))[:,0])
        
        return sum(sum(abs(cv2.absdiff(e2, e1)*0.1)))*100.0/len(e1)
=======
        cv2.imshow('diff', abs(cv2.absdiff(e2, e1))[:,0])

        return sum(sum(abs(cv2.absdiff(e2, e1)*1.0)))*256.0/len(e1)
>>>>>>> 5d4eb4e3aacce8bb25d49fea3d130ab72c8717c3:testings/getMatchingValues.py
    elif dif > 0:
        match = 50000
        for i in range(dif):
            match = min(match, sum(sum(cv2.absdiff(e2, e1[i:i+len(e2)])*0.1)))
        return match*100.0/len(e2)
    else:
        match = 50000
        for i in range(0-dif):
<<<<<<< HEAD:work.py
            match = min(match, sum(sum(cv2.absdiff(e1, e2[i:i+len(e1)])*0.1)))
        return match*100.0/len(e1)
    
=======
            match = min(match, sum(sum(cv2.absdiff(e1, e2[i:i+len(e1)])*1.0)))
        return match*1.0/len(e1)

>>>>>>> 5d4eb4e3aacce8bb25d49fea3d130ab72c8717c3:testings/getMatchingValues.py
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
<<<<<<< HEAD:work.py

    
def showMatch(pieces, matchid): 
    i,j,k,l = matchid
    im = rotatePiece(pieces[i], k)
    im2 = rotatePiece(pieces[j], (l+2)%4)
    im = np.append(im2, im, axis = 0)
    cv2.imshow('piece'+str(i)+str(j), im)

def rotatePiece(piece, i):
    if i%4 == 0: #North side
        return piece
    elif i%4 == 1: #East side
        return mirror(mirror(piece, 'D'), 'H')
    elif i%4 == 2:
        return mirror(mirror(piece, 'H'), 'V')
    elif i%4 == 3:
        return mirror(mirror(piece, 'H'), 'D')
        
    
def getBestMatch(pieces):
    sides = ["N","E","S","W"]
    bestmatch = 50000
=======

def showPiece(pieces, i):
    cv2.imshow('piece'+str(i), pieces[i])


def getBestMatch(pieces):

    sides = ["N","E","W","S"]
    bestmatch = 10000
>>>>>>> 5d4eb4e3aacce8bb25d49fea3d130ab72c8717c3:testings/getMatchingValues.py
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
#    print bestmatch
#    print(matchid)
#    showMatch(pieces, matchid)
#    showMatch(pieces, rev)
#    print matches[matchid[1], matchid[0], matchid[3], matchid[2]]
#    print matches[rev]
    return matches, matchid
<<<<<<< HEAD:work.py
            
def getNeighbour( matches, edgeId):
    pieceNmr, sideNmr = edgeId
    matchid = [0,0,0,0]
    bestmatch = 50000
    for i in range(len(matches)):
        if (not i == pieceNmr):
            for j in range(len(matches[0,0,0])):
                match = matches[i, pieceNmr, j, sideNmr%4]
                if match < bestmatch:
                    bestmatch = match
                    matchid = [i, pieceNmr, j, sideNmr%4]
#                    showMatch(pieces, matchid)
#
#                    cv2.waitKey()
#                    cv2.destroyAllWindows()
#    print bestmatch
#    print matchid
    return matchid

def getNeighbourEx( matches, edgeId, exclusions):
    pieceNmr, sideNmr = edgeId
    matchid = [0,0,0,0]
    bestmatch = 50000
    for i in range(len(matches)):
        if (not i == pieceNmr) and i not in exclusions:
            for j in range(len(matches[0,0,0])):
                match = matches[i, pieceNmr, j, sideNmr%4]
                if match < bestmatch:
                    bestmatch = match
                    matchid = [i, pieceNmr, j, sideNmr%4]
    return matchid    

def getLine(pieces, matches, matchid):
    
    #functie werkt naar behoren onder sommige omstandigheden, echter niet genoeg om betrouwbaar te werken
    #kijk naar volgende functie voor een beter werkende puzzelsolver
    
    i,j,k,l = matchid
    im = rotatePiece(pieces[i], k+2)
    im2 = rotatePiece(pieces[j], l)
    im = np.append(im, im2, axis = 0)
    cv2.imshow('line', im)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    length = 2
    up_edge = [matchid[0], matchid[2]+2]
    down_edge = [matchid[1], matchid[3]+2]
    up_match = getNeighbour(matches, up_edge)
    down_match= getNeighbour(matches, down_edge)
    connections = [matchid]
    
    while length < int(np.sqrt(len(matches))):
        length = length+1
        if matches[tuple(up_match)] > matches[tuple(down_match)]:
            print down_match
            connections.append(down_match)
            down_edge = [down_match[0], down_match[2]+2]
            down_match= getNeighbour(matches, down_edge)
            im2 = rotatePiece(pieces[down_edge[0]], down_edge[1]+2)
            im = np.append(im, im2, axis = 0)
#            cv2.imshow('line', im)
#            cv2.waitKey()
#            cv2.destroyAllWindows()            
            
        else:
            print up_match
            connections.append(up_match)
            up_edge = [up_match[0], up_match[2]+2]
            up_match = getNeighbour(matches, up_edge)
            im2 = rotatePiece(pieces[up_edge[0]], up_edge[1])
            im = np.append(im2, im, axis = 0)
    cv2.imshow('line', im)
    cv2.waitKey()
    cv2.destroyAllWindows()   

def combinePuzzle(pieces, placedPieces):
    h,w,chan = np.shape(pieces[0])
    count = int(np.sqrt(len(pieces)))
    image = np.zeros((h*count, w*count, chan), dtype = 'uint8')
    minx = miny = 0
    for piece in placedPieces:
        minx = min(minx,piece[1])
        miny = min(miny,piece[0])
    for i in range(len(placedPieces)):
        y,x,r = placedPieces[i]
        if not(r == 5):
            im = rotatePiece(pieces[i], r)
            x = x*h-minx*h
            y = y*w-miny*w
            image[x:x+h, y:y+w] = im
    cv2.imshow('puzzle', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return image
    
def puzzleSolver(pieces, matches, matchid):
    
    i,j,k,l = matchid
    im = rotatePiece(pieces[i], k+2)
    im2 = rotatePiece(pieces[j], l)
    im = np.append(im, im2, axis = 0)
    cv2.imshow('line', im)
    cv2.waitKey()
    cv2.destroyAllWindows()
    placedPieces = list([0,0,5] for i in range(len(pieces))) #gebruik rotatie 5 voor ongebruikte stukken
    placedPieces[i] = [0,0,(k-2)%4]
    placedPieces[j] = [0,1,l%4]
    used = [i]
    used.append(j)
    edges = [[i,k+1], [i,k+2], [i,k+3], [j,l+1], [j, l+2], [j, l+3]]
    neighbours = []
    minX= maxX = minY = 0
    maxY = 1
    for edge in edges:
        neighbours.append(getNeighbourEx(matches, edge, used))
#    while len(possible > 0):
    #bepaal de best matchende buur, en sla de matchId op in bestPiece
    while(len(used) < len(pieces)):
        bestMatch = 10000
        for i in range(len(neighbours)):
            match = matches[tuple(neighbours[i])]
            if match < bestMatch:
                bestMatch = match
                bestPiece = i
        #nu bepalen we aan welke kant het nieuwe stuk geplaatst moet worden
        #dit gebeurt door de draaiing van het geplaatst stuk 
        # legende: op = OLD piece, np = NEW PIECE, os = OLD side, ns= new side
        print neighbours[bestPiece]
        newP, oldP, ns, os = neighbours[bestPiece]
        oldXYR = placedPieces[oldP]
        side = (os-oldXYR[2])%4
        if placedPieces[newP][2] == 5:
            if side == 1 and oldXYR[0]<(minX+np.sqrt(len(pieces))-1):
                for i in range(4):
                    if not i == ns:
                        neighbours.append(getNeighbourEx(matches, [newP, i], used))
                placedPieces[newP] = [oldXYR[0]+1, oldXYR[1], (ns-2-side)%4]
                maxX = max(maxX, oldXYR[0]+1)
                used.append(newP)
            elif side == 3 and oldXYR[0]>(maxX-np.sqrt(len(pieces))+1):
                for i in range(4):
                    if not i == ns:
                        neighbours.append(getNeighbourEx(matches, [newP, i], used))
                placedPieces[newP] = [oldXYR[0]-1, oldXYR[1], (ns-2-side)%4]
                minX = min(minX, oldXYR[0]-1)
                used.append(newP)
            elif side == 0 and oldXYR[1]>(maxY-np.sqrt(len(pieces))+1):
                for i in range(4):
                    if not i == ns:
                        neighbours.append(getNeighbourEx(matches, [newP, i], used))
                placedPieces[newP] = [oldXYR[0], oldXYR[1]-1, (ns-2-side)%4]
                minY = min(minY, oldXYR[1]-1)
                used.append(newP)
            elif side == 2 and oldXYR[1]<(minY+np.sqrt(len(pieces))-1): 
                for i in range(4):
                    if not i == ns:
                        neighbours.append(getNeighbourEx(matches, [newP, i], used))
                placedPieces[newP] = [oldXYR[0], oldXYR[1]+1, (ns-2-side)%4]
                maxY = max(maxY, oldXYR[1]+1)
                used.append(newP)
        neighbours.pop(bestPiece)
            
#        combinePuzzle(pieces, placedPieces)
    return placedPieces, [minX]
       
=======

>>>>>>> 5d4eb4e3aacce8bb25d49fea3d130ab72c8717c3:testings/getMatchingValues.py

    
puzzleType="tiles"
<<<<<<< HEAD:work.py
puzzleArrangement="rotated"
puzzleSize="4x4"
puzzleNumber="04"        
=======
puzzleArrangement="shuffled"
puzzleSize="5x5"
puzzleNumber="04"
>>>>>>> 5d4eb4e3aacce8bb25d49fea3d130ab72c8717c3:testings/getMatchingValues.py
puzzle = cv2.imread('images/'+puzzleType+'/'+puzzleType+'_'+puzzleArrangement+'/'+puzzleType+'_'+puzzleArrangement+'_'+puzzleSize+'_'+puzzleNumber+'.png')
pieces = dividePuzzle(puzzle, 16)

sides = ["N","E","S","W"]
matches, matchid = getBestMatch(pieces)
#getLine(pieces, matches, matchid)
#print matches[tuple(matchid)]
cv2.waitKey()
cv2.destroyAllWindows()

x, minx = puzzleSolver(pieces, matches, matchid)
im = combinePuzzle(pieces, x)

cv2.imshow('im', im)
cv2.waitKey()
cv2.destroyAllWindows()

<<<<<<< HEAD:work.py





























=======
#cv2.imshow('im', puzzle)
#cv2.waitKey()
#cv2.destroyAllWindows()
>>>>>>> 5d4eb4e3aacce8bb25d49fea3d130ab72c8717c3:testings/getMatchingValues.py
