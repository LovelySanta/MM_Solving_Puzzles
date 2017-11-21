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

### End Of File ###
