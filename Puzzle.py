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

### End Of File ###
