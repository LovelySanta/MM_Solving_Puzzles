
from Puzzle import Puzzle


puzzleType="jigsaw"
puzzleArrangement="rotated"
puzzleSize="5x5"
puzzleNumber="04"

puzzle = Puzzle()
puzzle.readImageFromFile('images/'+puzzleType+'/'+puzzleType+'_'+puzzleArrangement+'/'+puzzleType+'_'+puzzleArrangement+'_'+puzzleSize+'_'+puzzleNumber+'.png')

## Create the mask to get pieces out of the background
puzzle.createMask()
puzzle.showImage(windowName="original", addWaitKey=False)

## Now we got the mask, let's get the pieces of the puzzle out
puzzle.findPuzzlePieces()
puzzle.cutAllPiecesOut()

puzzle.showAllPieces()


#puzzle.cutPuzzlePieces(4)
#puzzle.showAllPieces()
