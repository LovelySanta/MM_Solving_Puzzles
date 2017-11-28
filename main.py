
from Puzzle import Puzzle


puzzle = Puzzle()

#puzzle.readImageFromFile('images/jigsaw/jigsaw_scrambled/jigsaw_scrambled_5x5_04.png')
puzzle.readImageFromFile('images/tiles/tiles_scrambled/tiles_scrambled_5x5_04.png')

### Create the mask to get pieces out of the background
puzzle.createMask()
puzzle.showImage(windowName="original", addWaitKey=False)

### Now we got the mask, let's get the pieces of the puzzle out
puzzle.findPuzzlePieces()
puzzle.cutAllPiecesOut()

puzzle.showAllPieces()
