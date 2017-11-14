import numpy as np
import cv2

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
        pieces = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

        for pieceIndex in range(12):
            pts = pieces[pieceIndex]
            rect = cv2.minAreaRect(pts)
            angle = rect[2]-90

            # Cut the piece out of the image (+ space for outline)
            piece = self.image[min(pts[:, 0 ,1])-2:max(pts[:,0,1])+2, min(pts[:,0,0])-2:max(pts[:,0,0])+2]
            # Rotate the picture over given angle using a tranformation matrix
            transformMatrix = cv2.getRotationMatrix2D((len(piece[0])*0.75, len(piece)*0.75), angle, 1)
            rotatedPiece = cv2.warpAffine(piece, transformMatrix, (len(piece[0])*2, len(piece)*2))

            # Get the mask of the rotated piece
            rotatedPieceGray = cv2.cvtColor(rotatedPiece, cv2.COLOR_BGR2GRAY)
            im_bw = cv2.threshold(rotatedPieceGray, 1, 255, cv2.THRESH_BINARY)[1]

            # Get the countour of the rotated image and approximate it
            cnt = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
            peri = cv2.arcLength(pts, True)
            approx = cv2.approxPolyDP(cnt[0], 0.05*peri, True)
            hull = cv2.convexHull(cnt[0])

            # Draw the outline of the piece
            color = getRandomColor()
            for point in range(len(hull)-1):
                cv2.line(rotatedPiece, tuple(hull[point][0]), tuple(hull[point+1][0]), color, 2)
            cv2.line(rotatedPiece, tuple(hull[0][0]), tuple(hull[point+1][0]), color, 2)

            # Create new piece instance and add it to the Puzzle:pieces
            puzzlePiece = PuzzlePiece(piece, rotatedPiece)
            self.puzzlePieces.append(puzzlePiece)



    def showAllPieces(self):
        for puzzlePiece in self.puzzlePieces:
            puzzlePiece.showPuzzlePiece()

### End Of File ###
