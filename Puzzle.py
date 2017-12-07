import numpy as np
import cv2
import random
from datetime import datetime
random.seed(datetime.now())
import math

from PuzzlePiece import PuzzlePiece
from basicFunc import getRandomColor

class Puzzle:

    def __init__(self):
        self.sides = ["N","E","S","W"]

        self.image = None
        self.dim = {
            "height" : 0,
            "width" : 0,
            "channels" : 0
        }
        self.mask = None
        self.puzzlePieces = []

        self.usedPieces = []
        self.solvedPuzzle = []

        return None



    def __str__(self):
        string = "   Puzzle class instance   "



    # DESCRIPTION : Read image from file, save it as BGR color image
    # ARGUMENTS   : filename: string with path to file
    # RETURN      : None
    def readImageFromFile(self, filename):
        self.image = cv2.imread(filename, cv2.IMREAD_COLOR)

        h, w, ch = self.image.shape
        self.dim = {
            "height" : h,
            "width" : w,
            "channels" : ch
        }



    # DESCRIPTION : Create instance from another instance
    # ARGUMENTS   : other: other instance
    # RETURN      : None
    def copyFrom(self, other):
        if not self.instance_of(other):
            raise TypeError("Cannot copy if argument is not of same class (" + self.__class__.__name__ + ").")

        self.image = np.copy(other.image)
        self.dim = other.dim.copy()
        self.mask = np.copy(other.image)



    # DESCRIPTION : Generate a Signal instance equivalent to itself
    # ARGUMENTS   : None
    # RETURN      : cpy: a Signal instance containing a copy of itself
    def copy(self):
        cpy = Puzzle()
        cpy.copyFrom(self)
        return cpy



    # DESCRIPTION : Check if an instance is an instance of the Signal class
    # ARGUMENTS   : other: instance to check
    # RETURN      : True: if instance of Signal
    #               False: otherwise
    def instance_of(self, other):
        return True if isinstance(other, self.__class__) else False



    # DESCRIPTION : Show image on screen
    # ARGUMENTS   : windowName: window title
    #               addWaitKey: if true, windows get destroyed when button pressed
    # RETURN      : None
    def showImage(self, windowName="", addWaitKey=True):
        cv2.imshow(windowName, self.image)
        if addWaitKey:
             cv2.waitKey()
             cv2.destroyAllWindows()



    # DESCRIPTION : Show mask on screen
    # ARGUMENTS   : windowName: window title
    #               addWaitKey: if true, windows get destroyed when button pressed
    # RETURN      : None
    def showMask(self, windowName="", addWaitKey=True):
        cv2.imshow(windowName, self.mask)
        if addWaitKey:
             cv2.waitKey()
             cv2.destroyAllWindows()



    def createMask(self):
        lut = np.ones(256, dtype='uint8')
        lut[0] = 0
        lut *= 255

        self.mask = np.copy(self.image)
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        self.mask = cv2.LUT(self.mask, lut)



    def findPuzzlePieces(self):
        cnts = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

        # For each piece, cut out and rotate
        for i in range(len(cnts)):

            pts = cnts[i]
            x, y, w, h = cv2.boundingRect(pts)
            if(h > len(self.image)/10 and w > len(self.image[0])/10):

                # Cut out the piece
                piece = self.image[y:y+h, x:x+w]

                # Add extra space to picture so it won't get out of bound when rotating
                empty = np.zeros(piece.shape, dtype = 'uint8')
                # Add left and right
                piece = np.concatenate((empty, piece, empty))
                # Add above and beneath
                empty = np.concatenate((empty, empty, empty))
                piece = np.concatenate((empty, piece, empty), axis = 1)

                # Find the rotation using ransac
                hoek = self.rotateRansac(pts)

                # Rotate the picture over given angle using a tranformation matrix
                M = cv2.getRotationMatrix2D((len(piece[0])*0.5, len(piece)*0.5),hoek,1)
                rotatedPiece = cv2.warpAffine(piece,M,(len(piece[0]), len(piece)))

                # Create new piece instance and add it to the Puzzle:pieces
                puzzlePiece = PuzzlePiece(rotatedPiece)
                self.puzzlePieces.append(puzzlePiece)



    def cutPuzzlePieces(self, numberOfPieces):
        size = int(math.ceil(math.sqrt(numberOfPieces)))
        for i in range(0, size):
            for j in range(0, size):
                #cutoutpiece
                piece = self.image[(i*len(self.image))/size:((i+1)*len(self.image))/size-1,(j*len(self.image[0]))/size:((j+1)*len(self.image[0]))/size-1]
                puzzlePiece = PuzzlePiece(piece)
                self.puzzlePieces.append(puzzlePiece)



    def showAllPieces(self):
        for puzzlePiece in self.puzzlePieces:
            puzzlePiece.showPuzzlePiece()

    def cutAllPiecesOut(self):
        for puzzlePiece in self.puzzlePieces:
            puzzlePiece.cutOut()



    def showMatch(self, matchId):
        i,j,k,l = matchId
        # First matching image
        piece1 = self.puzzlePieces[i].rotatePiece(k)
        # Second matching image
        piece2 = self.puzzlePieces[j].rotatePiece((l+2)%4)

        match = np.append(piece2, piece1, axis = 0)
        cv2.imshow('pieces'+str(i)+' and '+str(j), match)
        cv2.waitKey()
        cv2.destroyAllWindows()



    def rotateRansac(self, contour):
        T_i = len(contour)/10
        teller = 0
        line = [0,1]
        i = random.randint(1,len(contour)-1)
        finished = False

        while(finished == False):
            line[0] = []
            line[1] = []
            # Change T_d dynamicly
            teller = teller+1
            T_d = 0.01+0.01*np.floor(teller*16.0/len(contour))

            # Random start point on first try, next time, just increment
            i = (i+17)%(len(contour)-1)
            x1 , y1 = contour[i][0]

            # Get 2nd point on fixed distance in array
            if(i >= len(contour)/6):
                x2 , y2 = contour[i-len(contour)/6][0]
            else:
                x2 , y2 = contour[i+len(contour)/6][0]

            # Get Rico of this line < 45 degrees
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

            if (divide != "z"):
                # For all other points, check matching ricos
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
                    # Check if we met T_i
                    if(len(line[0]) > T_i):
                        finished = True

        # Calculate the angle
        if (divide == "x"):
            A,B = np.polyfit(line[0], line[1], 1)
            hoek = np.rad2deg(np.arctan(A))
        else:
            A,B = np.polyfit(line[1], line[0], 1)
            hoek = 90-np.rad2deg(np.arctan(A))
        # Return the angle in degrees
        return hoek



    def getMatchingValue(self, edge1, edge2):
        dif = len(edge1)-len(edge2)

        # If dimensions mismatch too much (rectangle), it won't match at all
        if abs(dif) > 5:
            return 50000

        # Perfect match
        elif dif == 0:
            #cv2.imshow('diff', abs(cv2.absdiff(edge2, edge1))[:,0])
            return sum(sum(abs(cv2.absdiff(edge2, edge1)*0.1)))*100.0/len(edge1)

        # Small mismatch (rotation error)
        elif dif > 0:
            match = 50000
            for i in range(dif):
                match = min(match, sum(sum(cv2.absdiff(edge2, edge1[i:i+len(edge2)])*0.1)))
            return match*100.0/len(edge2)
        else:
            match = 50000
            for i in range(0-dif):
                match = min(match, sum(sum(cv2.absdiff(edge1, edge2[i:i+len(edge1)])*0.1)))
            return match*100.0/len(edge1)



    def getBestMatch(self):
        noMatchValue = 50000
        bestmatch = noMatchValue
        matchid = [0,0,0,0]

        numberOfPieces = len(self.puzzlePieces)
        numberOfSides = len(self.sides)
        matches = np.empty((numberOfPieces, numberOfPieces, numberOfSides, numberOfSides), dtype = 'float64')

        # Iterate over all edges of all pieces...
        for i in range(numberOfPieces):
            for k in range(numberOfSides):
                edge1 = self.puzzlePieces[i].getEdge(self.sides[k])
                # Match this side to all other sides of other pieces.
                for j in range(numberOfPieces):
                    if not (i == j):
                        for l in range(numberOfSides):
                            # Get the matching value
                            match = int(self.getMatchingValue(
                                edge1,
                                self.puzzlePieces[j].getEdge(self.sides[l])  ))
                            matches[i,j,k,l] = match
                            matches[j,i,l,k] = match

                            # Check if the match is the best match up till now
                            if match < bestmatch:
                                bestmatch = match
                                matchid = [i,j,k,l]
                                rev = [j,i,l,k]
                    else:
                        matches[i,j,k]= list(noMatchValue for l in range(len(self.sides)))

        # Return all matching values we calculated and the index of the two best matchings
        return matches, matchid



    def showSolvedPuzzle(self):
        h,w,chan = np.shape(self.puzzlePieces[self.solvedPuzzle[0][0][0]].rotatePiece(self.solvedPuzzle[0][0][1]))
        puzzleSize = int(np.sqrt(len(self.puzzlePieces)))

        image = np.zeros((h*puzzleSize, w*puzzleSize, chan), dtype = 'uint8')
        for i in range(len(self.solvedPuzzle)):
            for j in range(len(self.solvedPuzzle)):
                piece, rotation = self.solvedPuzzle[i][j]
                if not piece == -1:
                    puzzlePiece = self.puzzlePieces[piece].rotatePiece(rotation)
                    posX = i*h
                    posY = j*w
                    image[posX:posX+h, posY:posY+w] = puzzlePiece

        cv2.imshow("Solved Puzzle", image)
        cv2.waitKey()
        cv2.destroyAllWindows()



    def getNextBestMatches1D(self, pieceToMatch, matches):
        matchPiece, matchSide = pieceToMatch
        matchId = [-1,0,0,0]
        bestMatchValue = 50000
        # For each unused piece
        for i in range(len(matches)):
            if i not in self.usedPieces:
                # For each side of the piece
                for j in range(len(matches[0,0,0])):
                    matchValue = matches[i, matchPiece, j, matchSide%4]
                    if matchValue < bestMatchValue:
                        bestMatchValue = matchValue
                        matchId = [i, matchPiece, j, matchSide%4]
        return matchId



    def getNextBestMatches2D(self, piecesToMatch, matches):
        matchPiece1, matchSide1 = piecesToMatch[0]
        matchPiece2, matchSide2 = piecesToMatch[1]

        matchId1 = matchId2 = [-1,0,0,0]
        bestMatchValue = 5000000
        # For each unused piece
        for i in range(len(matches)):
            if i not in self.usedPieces:
                # For each side of the piece
                for j in range(len(matches[0,0,0])):
                    matchValue1 = matches[i, matchPiece1, j, matchSide1%4]
                    matchValue2 = matches[i, matchPiece2, (j+1)%4, matchSide2%4]
                    matchValue = matchValue1**2 + matchValue2**2
                    if matchValue < bestMatchValue:
                        bestMatchValue = matchValue
                        matchId1 = [i, matchPiece1, j, matchSide1%4]
                        matchId2 = [i, matchPiece2, j+1, matchSide2%4]
        self.showMatch(matchId1)
        self.showMatch(matchId2)
        return matchId1


    def expandRectangle(self, matches):
        #check for max width
        puzzleSize = len(self.solvedPuzzle)
        i = j = puzzleSize - 1
        bestMatch = 50000
        matchId = [-1,-1,-1,-1,-1]

        # Check size of the current rectangle
        while self.solvedPuzzle[i][0][0] == -1:
            i = i-1
        while self.solvedPuzzle[0][j][0] == -1:
            j = j-1
        print i,j

        # If not length of rows full size
        if j != (puzzleSize-1):
            # Check horizontal edges for each row
            for k in range(i+1):
                # Check the left side
                newMatchId = self.getNextBestMatches1D([self.solvedPuzzle[k][0][0], (self.solvedPuzzle[k][0][1]-1)%4], matches)
                newMatch = matches[tuple(newMatchId)]
                if newMatch < bestMatch:
                    bestMatch = newMatch
                    matchId = [k,-1, newMatchId[0], (newMatchId[2]-1)%4, 3]
                    bestMatchId = newMatchId

                # Check the right side
                newMatchId = self.getNextBestMatches1D([self.solvedPuzzle[k][j][0], (self.solvedPuzzle[k][j][1]+1)%4], matches)
                newMatch = matches[tuple(newMatchId)]
                if newMatch < bestMatch:
                    bestMatch = newMatch
                    matchId = [k,j+1, newMatchId[0], (newMatchId[2]+1)%4, 1]
                    bestMatchId = newMatchId

        # If not colums full size
        if i != (puzzleSize-1):
            for k in range(j+1):
                # Check the top side
                newMatchId = self.getNextBestMatches1D([self.solvedPuzzle[0][k][0], (self.solvedPuzzle[0][k][1]-0)%4], matches)
                newMatch = matches[tuple(newMatchId)]
                if newMatch < bestMatch:
                    bestMatch = newMatch
                    matchId = [-1,5, newMatchId[0], (newMatchId[2]+2)%4, 0]
                    bestMatchId = newMatchId

                # Check the bottom side
                newMatchId = self.getNextBestMatches1D([self.solvedPuzzle[i][k][0], (self.solvedPuzzle[i][k][1]+2)%4], matches)
                newMatch = matches[tuple(newMatchId)]
                if newMatch < bestMatch:
                    bestMatch = newMatch
                    matchId = [i+1,k, newMatchId[0], (newMatchId[2]-0)%4, 2]
                    bestMatchId = newMatchId

        return matchId



    def fillRectangle(self, matches, matchId):
        x, y, n, r, d = matchId
        newLoc = [x,y]

        # Added piece on the top, fill left/right
        if d == 0:
            oldLoc1 = [x+1, y-1]
            dir1 = [0,-1]
            oldLoc2 = [x+1, y+1]
            dir2 = [0, 1]
        # Added piece on the bottom, fill left/right
        elif d == 2:
            oldLoc1 = [x-1, y+1]
            dir1 = [0, 1]
            oldLoc2 = [x-1, y-1]
            dir2 = [0,-1]
        # Added piece to the left, fill top/bottm
        elif d == 3:
            oldLoc1 = [x+1, y+1]
            dir1 = [1, 0]
            oldLoc2 = [x-1, y+1]
            dir2 = [-1,0]
        # Added piece to the right, fill top/bottom
        elif d == 1:
            oldLoc1 = [x-1, y-1]
            dir1 = [-1,0]
            oldLoc2 = [x+1, y-1]
            dir2 = [1, 0]
        else:
            print("Invalid fill direction!")

        # Check the first direction
        while (-1 < oldLoc1[0] and oldLoc1[0] < len(self.solvedPuzzle)) and \
              (-1 < oldLoc1[1] and oldLoc1[1] < len(self.solvedPuzzle)) and \
              self.solvedPuzzle[tuple(oldLoc1)][0] > -1:
            # Get the two pieces with edges to check ready to make a 2D match
            newPiece = np.add(self.solvedPuzzle[tuple(newLoc)], [0,d-1])
            oldPiece = np.add(self.solvedPuzzle[tuple(oldLoc1)], [0, d])
            newNeighbour = self.getNextBestMatches2D([newPiece, oldPiece], matches)
            # We found a match, add it to the pieces to match, and do again in same direction
            newLoc = np.add(newLoc, dir1)
            self.solvedPuzzle[tuple(newLoc)] = [newNeighbour[0],(newNeighbour[2])]
            self.usedPieces.append(newNeighbour[0])
            oldLoc1 = np.add(oldLoc1, dir1)

        # Check the other direction
        newLoc = [x,y]
        while (-1 < oldLoc2[0] and oldLoc2[0] < len(self.solvedPuzzle)) and \
              (-1 < oldLoc2[1] and oldLoc2[1] < len(self.solvedPuzzle)) and \
              self.solvedPuzzle[tuple(oldLoc2)][0] > -1:
            # Get the two pieces with edges to check ready to make a 2D match
            newPiece = np.add(self.solvedPuzzle[tuple(newLoc)], [0,d+1])
            oldPiece = np.add(self.solvedPuzzle[tuple(oldLoc2)], [0, d])
            newNeighbour = self.getNextBestMatches2D([newPiece, oldPiece], matches)
            # We found a match, add it to the pieces to match, and do again in same direction
            newLoc = np.add(newLoc, dir2)
            self.solvedPuzzle[tuple(newLoc)] = [newNeighbour[0],(newNeighbour[2])]
            self.usedPieces.append(newNeighbour[0])
            oldLoc2 = np.add(oldLoc2, dir1)



    def solvePuzzle(self):
        matches, matchid = self.getBestMatch()
        piece1, piece2, orientation1, orientation2 = matchid

        puzzleSize = int(np.sqrt(len(self.puzzlePieces)))
        # Reference to the solved puzzle, each coordinate contains a piece and an orentation of that piece
        self.solvedPuzzle = np.full((puzzleSize,puzzleSize,2), -1, dtype = 'int8')
        self.solvedPuzzle[0][0] = [piece1,(orientation1-2)%4]
        self.solvedPuzzle[1][0] = [piece2,orientation2]
        self.usedPieces=[piece1,piece2]

        self.showSolvedPuzzle()

        # Now we got the start of our puzzle, now lets solve all other pieces
        #while len(usedPieces) < len(self.puzzlePieces):
            # The puzzle is rectangular, now we add an extra piece to make an L shape
        matchId = self.expandRectangle(matches)
        #print(matchId)
        # Check if the piece to add is left or on top, if so, make room
        if matchId[0] == -1:
            self.solvedPuzzle = np.append(np.full((1,len(self.solvedPuzzle), 2), -1, dtype = 'int8'),self.solvedPuzzle[0:-1], axis = 0)
            matchId[0] = 0
            self.showSolvedPuzzle()
        elif matchId[1] == -1:
            self.solvedPuzzle = np.append(np.full((len(self.solvedPuzzle), 1, 2), -1, dtype = 'int8'),self.solvedPuzzle[:,0:-1], axis = 1)
            matchId[1] = 0
            self.showSolvedPuzzle()

        self.solvedPuzzle[matchId[0]][matchId[1]] = [matchId[2], matchId[3]]
        self.usedPieces.append(matchId[2])
        self.showSolvedPuzzle()

        # We added an extra piece, now we have an L shaped, fill it up to a rectangle again
        self.fillRectangle(matches, matchId)
        self.showSolvedPuzzle()


### End Of File ###
