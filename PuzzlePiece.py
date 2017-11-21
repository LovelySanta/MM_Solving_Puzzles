import numpy as np
import cv2

class PuzzlePiece:

    def __init__(self, piece):
        self.piece = piece

        return None

    def showPuzzlePiece(self):
        cv2.imshow('Original Piece', self.piece)
        #cv2.imshow('Rotated Piece', self.rotatedPiece)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def cutOut(self):
        # Cut out the picture so there is no background

        x, y, w, h = cv2.boundingRect(self.createMask())
        self.piece = self.piece[y+2:y+h-2, x+1:x+w-2]

    def createMask(self):
        lut = np.ones(256, dtype='uint8')
        lut[0] = 0
        lut *= 255

        mask = np.copy(self.piece)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.LUT(mask, lut)

        return mask

### End Of File ###
