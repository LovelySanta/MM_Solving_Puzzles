import numpy as np
import cv2

class PuzzlePiece:

    def __init__(self, piece):
        self.piece = piece
        self.h, self.w, self.channels = self.piece.shape

        return None

    def showPuzzlePiece(self, windowName="", addWaitKey=True):
        cv2.imshow(windowName, self.piece)
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

### End Of File ###
