import numpy as np
import cv2

class PuzzlePiece:

    def __init__(self, piece):
        self.sides = ["N","E","S","W"]

        self.piece = piece
        self.h, self.w, self.channels = self.piece.shape

        return None

    def showPuzzlePiece(self, windowName="", addWaitKey=True, rotation=0):
        cv2.imshow(windowName, self.rotatePiece(rotation))
        #cv2.imshow('Rotated Piece', self.rotatedPiece)
        if addWaitKey:
            cv2.waitKey()
            cv2.destroyAllWindows()

    def cutOut(self):
        # Cut out the picture so there is no background
        x, y, w, h = cv2.boundingRect(self.createMask())
        self.piece = self.piece[y:y+h, x:x+w]


    def createMask(self):
        lut = np.ones(256, dtype='uint8')
        lut[0] = 0
        lut *= 255

        mask = np.copy(self.piece)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.LUT(mask, lut)

        return mask

    def getEdge(self, edge):
        # Match integer to string
        if type(edge) == type(5):
            side = self.sides[edge%4]
        else:
            side = edge

        # Get the edge
        if edge == "W":
            return(self.piece[:,0, ])
        elif edge == "N":
            return(self.piece[0,:, ])
        elif edge == "E":
            return(self.piece[:,-1, ])
        elif edge == "S":
            return(self.piece[-1,:, ])
        else:
            return None

    def mirrorPiece(self, axis, piece):
        if axis == 'H':
            return cv2.flip(piece, 0)
        elif axis == 'V':
            return cv2.flip(piece, 1)
        elif axis == 'D':
            return cv2.transpose(piece)

    def rotatePiece(self, rotation):
            if rotation % 4 == 0: # North side
                return self.piece
            elif rotation % 4 == 1: # East side
                return self.mirrorPiece('H', self.mirrorPiece('D', self.piece))
            elif rotation % 4 == 2: # South side
                return self.mirrorPiece('V', self.mirrorPiece('H', self.piece))
            elif rotation % 4 == 3: # West side
                return self.mirrorPiece('D', self.mirrorPiece('H', self.piece))

### End Of File ###
