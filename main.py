from Puzzle import Puzzle

puzzle = Puzzle()
puzzle.readImageFromFile('images/tiles/tiles_scrambled/tiles_scrambled_5x5_04.png')

### Create the mask to get pieces out of the background
puzzle.createMask()
puzzle.showImage(windowName="original", addWaitKey=False)
puzzle.showMask(windowName="mask")

### Now we got the mask, let's get the pieces of the puzzle out
puzzle.findPuzzlePieces()
puzzle.showAllPieces()
