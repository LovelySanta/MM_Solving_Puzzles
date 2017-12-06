# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 16:05:26 2017

@author: jarno
"""

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
        return sum(sum(abs(cv2.absdiff(e2, e1)**1.5*0.1)))*10.0/len(e1)
    elif dif > 0:
        match = 50000
        for i in range(dif):
            match = min(match, sum(sum(cv2.absdiff(e2, e1[i:i+len(e2)])*0.1)))
        return match*10.0/len(e2)**2
    else:
        match = 50000
        for i in range(0-dif):
            match = min(match, sum(sum(cv2.absdiff(e1, e2[i:i+len(e1)])*0.1)))
        return match*10.0/len(e1)

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
    matchid = [0,0,0,0]
    matches = np.empty((len(pieces), len(pieces), len(sides), len(sides)), dtype = 'uint16')
    for i in range(len(pieces)):
        for k in range(len(sides)):
            e1 = getedge(pieces[i],sides[k])
            for j in range( len(pieces)):
                if not(i == j):
                    for l in range(len(sides)):
                        match = int(getMatch(e1, getedge(pieces[j],sides[l])))
                        matches[i,j,k,l]= match
#                        matches[j,i,l,k]= match
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
        if i not in exclusions:
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

# DISCLAIMER! THE EDGES SHOULD BE SENT CLOCKWISE
# EXAMPLE: FIRST EDGE IS THE RIGHT EDGE OF A PIECE, SECOND ONE IS THE BOTTOM ONE
# IF THIS IS NOT FOLLOWED, MISMATCHES MAY AND WILL HAPPEN
# THE REFERENCE FOR THE MATCH WILL BE DIRECTED TOWARDS THE FIRST NUMBER
def getNeighbour2D(matches, edgeIDs, exclusions): 
    pieceNmr1, sideNmr1 = edgeIDs[0]
    pieceNmr2, sideNmr2 = edgeIDs[1]
    matchid = matchid2 = [-1,0,0,0]
    bestmatch = 5000000
    for i in range(len(matches)):
        if (not i in exclusions):
            for j in range(len(matches[0,0,0])):
                match1 = matches[i, pieceNmr1, j, sideNmr1%4]
                match2 = matches[i, pieceNmr2, (j+1)%4, (sideNmr2+1)%4]
                match = match1**2+match2**2
                if match < bestmatch:
                    bestmatch = match
                    matchid = [i, pieceNmr1, j, sideNmr1%4]
                    matchid2 = [i, pieceNmr2, (j+1)%4, (sideNmr2+1)%4]
#                    print match, match1, match2
#    showMatch(pieces, matchid)
#    showMatch(pieces, matchid2)

#                    cv2.waitKey()
#                    cv2.destroyAllWindows()
#    print bestmatch
#    print matchid
    return matchid

def showsolvedPuzzle(pieces, placedPieces):
    h,w,chan = np.shape(pieces[0])
    count = int(np.sqrt(len(pieces)))
    image = np.zeros((h*count, w*count, chan), dtype = 'uint8')
    for i in range(len(placedPieces)):
        for j in range(len(placedPieces[0])):
            piece, rot = placedPieces[i][j]
            if not piece == -1:
                im = rotatePiece(pieces[piece], rot)
                x = i*h
                y = j*w
                image[x:x+h, y:y+w] = im
    cv2.imshow('puzzle', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return image                
            

def fillBlock(matches, placedPieces, used, matchid, pieces):
    x, y, n, r, d = matchid
    newLoc = [x,y]
    newPieces = [0]
    newPieces[0] = [x,y,n,r]
    used_t = used
    if d == 0 or d == 4:
        oldLoc = [x+1, y-1]
        OL2 = [x+1, y+1]
        dir2 = [0,1]
        dir1 = [0,-1]

    elif d == 3:
        oldLoc = [x+1, y+1]
        OL2 = [x-1, y+1]
        dir2 = [-1,0]
        dir1 = [1,0]
        
    elif d == 2:
        oldLoc = [x-1, y+1]
        OL2 = [x-1, y-1]
        dir2 = [0,-1]
        dir1 = [0,1]
        
    else: 
        oldLoc = [x-1, y-1]
        OL2 = [x+1, y-1]
        dir2 = [1,0]
        dir1 = [-1,0]
        
    print oldLoc
    while -1 < oldLoc[0] < len(placedPieces) and -1 < oldLoc[1] < len(placedPieces) and placedPieces[tuple(oldLoc)][0] > -1:
        oldPiece = np.add(placedPieces[tuple(oldLoc)], [0,d-1])
        newPiece = np.add(placedPieces[tuple(newLoc)], [0,d-1])
        neighbour = getNeighbour2D(matches, [newPiece, oldPiece], used)
        oldLoc = np.add(oldLoc, dir1)
        newLoc = np.add(newLoc, dir1)
        piece = np.append(newLoc, (neighbour[0], neighbour[2]+d*3+3))
        newPieces.append(piece)
#        print piece
        placedPieces[tuple(newLoc)] = [neighbour[0],(neighbour[2]+d*3+3)%4]
        used.append(neighbour[0])
#        showsolvedPuzzle(pieces, placedPieces)
#        print oldLoc
    oldLoc = OL2
    newLoc = [x,y]
    used = used_t
    print oldLoc
    while -1 < oldLoc[0] < len(placedPieces) and -1 < oldLoc[1] < len(placedPieces) and placedPieces[tuple(oldLoc)][0] > -1:
        oldPiece = np.add(placedPieces[tuple(oldLoc)], [0,d])
        newPiece = np.add(placedPieces[tuple(newLoc)], [0,d])
        neighbour = getNeighbour2D(matches, [oldPiece, newPiece], used)
        oldLoc = np.add(oldLoc, dir2)
        newLoc = np.add(newLoc, dir2)
        piece = np.append(newLoc, (neighbour[0], neighbour[2]+d*3+2))
        newPieces.append(piece)
#        print piece
        placedPieces[tuple(newLoc)] = [neighbour[0],(neighbour[2]+d*3+2)%4]
        used.append(neighbour[0])
#        showsolvedPuzzle(pieces, placedPieces)
#        print oldLoc
        
    return newPieces
    
def expandBlock(matches, placedPieces, used):
    #check for max width
    i = j = len(placedPieces[0])-1
    bestMatch = 50000
    matchid = [-1,-1,-1,-1,-1]
    while placedPieces[i][0][0] == -1:
        i = i-1
    while placedPieces[0][j][0] == -1:
        j = j-1
    i = i
    j = j
    print i,j
    if not j >= len(placedPieces)-1:
        for k in range(i+1):
#            print "k in i:"
#            print k
            Mid = getNeighbourEx(matches, [placedPieces[k][0][0], (placedPieces[k][0][1]-1)%4], used)
            newMatch = matches[tuple(Mid)]
            if newMatch < bestMatch:
                bestMatch = newMatch
                matchid = [k,-1, Mid[0], (Mid[2]-1)%4, 3]
                BM = Mid
            Mid = getNeighbourEx(matches, [placedPieces[k][j][0], (placedPieces[k][j][1]+1)%4], used)
            newMatch = matches[tuple(Mid)]
            if newMatch < bestMatch:
                bestMatch = newMatch
                matchid = [k,j+1, Mid[0], (Mid[2]+1)%4, 1] 
                BM = Mid
                
                
    if not i >= len(placedPieces)-1:
        for k in range(j+1):
#            print "k:"
#            print k, j
            Mid = getNeighbourEx(matches, [placedPieces[0][k][0], placedPieces[0][k][1]], used)
            newMatch = matches[tuple(Mid)]
            if newMatch < bestMatch:
                bestMatch = newMatch
                matchid = [-1,k, Mid[0], (Mid[2]-2)%4, 0]
                BM = Mid
            Mid = getNeighbourEx(matches, [placedPieces[i][k][0], (placedPieces[i][k][1]+2)%4], used)
            newMatch = matches[tuple(Mid)]
            if newMatch < bestMatch:
                bestMatch = newMatch
                matchid = [i+1,k, Mid[0], Mid[2], 2] 
                BM = Mid
    return matchid
                  

def solverMK2(pieces, matches, matchid):
    
    i,j,k,l = matchid
    width = int(np.sqrt(len(pieces)))
    placedPieces = np.full((width,width,2), -1, dtype = 'int8')
    placedPieces[0][0] = [i,(k-2)%4]
    placedPieces[1][0] = [j,l]
    used = [i,j]
    while len(used) < 25:
        matchid = expandBlock(matches, placedPieces, used)
        if matchid[0] == -1:
            placedPieces = np.append(np.full((1,len(placedPieces), 2), -1, dtype = 'int8'),placedPieces[0:-1], axis = 0)
            matchid[0] = 0
        elif matchid[1] == -1:
            placedPieces = np.append(np.full((len(placedPieces), 1, 2), -1, dtype = 'int8'),placedPieces[:,0:-1], axis = 1)
            matchid[1] = 0
        print matchid
        showsolvedPuzzle(pieces, placedPieces)
        placedPieces[matchid[0]][matchid[1]] = [matchid[2], matchid[3]]
        showsolvedPuzzle(pieces, placedPieces)
        used.append(matchid[2])
        matchid = fillBlock(matches, placedPieces, used,  matchid, pieces)
        for piece in range(len(matchid)):
            print matchid[piece]
            placedPieces[matchid[piece][0]][matchid[piece][1]] = [matchid[piece][2], matchid[piece][3]]
            used.append(matchid[piece][2])
#        placedPieces[matchid[0][0]][matchid[0][1]] = [matchid[0][2], matchid[0][3]]
        
    showsolvedPuzzle(pieces, placedPieces)
    return placedPieces




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



puzzleType="tiles"
puzzleArrangement="rotated"
puzzleSize="4x4"
puzzleNumber="04"
puzzle = cv2.imread(puzzleType+'_'+puzzleArrangement+'_'+puzzleSize+'_'+puzzleNumber+'.png')
pieces = dividePuzzle(puzzle, 16)

sides = ["N","E","S","W"]
matches, matchid = getBestMatch(pieces)
#getLine(pieces, matches, matchid)
#print matches[tuple(matchid)]
cv2.waitKey()
cv2.destroyAllWindows()


#x, minx = puzzleSolver(pieces, matches, matchid)
y = solverMK2(pieces, matches, matchid)
showsolvedPuzzle(pieces, y)
#im = combinePuzzle(pieces, x)
trial = sum(sum(abs(cv2.absdiff(getedge(pieces[9], 2), getedge(pieces[10],0))*0.1)))*10.0/len(pieces[0])
test2 = sum(abs(cv2.absdiff(getedge(pieces[9], 2), getedge(pieces[10],0))*0.1))*10.0/len(pieces[0])
test3 = abs(cv2.absdiff(getedge(pieces[9], 2), getedge(pieces[10],0))*0.1)*10.0
test4 = abs(cv2.absdiff(getedge(pieces[15], 0), getedge(pieces[12],0))*0.1)*10.0
#cv2.imshow('im', im)
cv2.waitKey()
cv2.destroyAllWindows()

