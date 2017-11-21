import numpy as np
import cv2
import random
from datetime import datetime
random.seed(datetime.now())

from PuzzlePiece import PuzzlePiece
from basicFunc import getRandomColor

class Puzzle:

    def __init__(self):
        self.image = None
        self.dim = {
            "height" : 0,
            "width" : 0,
            "channels" : 0
        }
        self.mask = None
        self.puzzlePieces = []

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
        cnts = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

        # For each piece
        for i in range(len(cnts)):

            # Cut the piece out, depending on the bounding box
            pts = cnts[i]
            x, y, w, h = cv2.boundingRect(pts)
            piece = self.image[y:y+h, x:x+w]

            # Find the rotation using ransac
            hoek = self.rotateRansac(pts)

            # Rotate the picture over given angle using a tranformation matrix
            M = cv2.getRotationMatrix2D((len(piece[0])*0.5, len(piece)*0.5),hoek,1)
            rotatedPiece = cv2.warpAffine(piece,M,(len(piece[0])*2, len(piece)*2))

            # Create new piece instance and add it to the Puzzle:pieces
            puzzlePiece = PuzzlePiece(rotatedPiece)
            self.puzzlePieces.append(puzzlePiece)



    def showAllPieces(self):
        for puzzlePiece in self.puzzlePieces:
            puzzlePiece.showPuzzlePiece()

    def cutAllPiecesOut(self):
        for puzzlePiece in self.puzzlePieces:
            puzzlePiece.cutOut()



    def rotateRansac(self, contour):
        T_i = len(contour)/10
        T_d = 0.015

        i = random.randint(1, len(contour)-1)

        finished = False
        while(finished == False):
            number_of_points = 0

            # Random start point on first try, next time, just increment
            i = (i+17)%(len(contour)-1)
            x1 , y1 = contour[i][0]

            # Get 2nd point on fixed distance in array
            if(i >= len(contour)/5):
                x2 , y2 = contour[i-len(contour)/5][0]
            else:
                x2 , y2 = contour[i+len(contour)/5][0]

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
                    number_of_points = number_of_points + 1
                if(number_of_points > T_i):
                    finished = True
        # Calculate the angle
        hoek = np.arctan2(x1-x2,y1-y2)

        # Return the angle in degrees
        return -(np.rad2deg(hoek))%90

### End Of File ###
