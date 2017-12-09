
from Puzzle import Puzzle
import cv2


puzzleType="tiles"
puzzleArrangement="rotated"
puzzleSize="4x4"
puzzleNumber="04"




### STEP 1: READ THE PUZZLE INPUT and determine settings
puzzle = Puzzle()
puzzle.readImageFromFile('images/'+puzzleType+'/'+puzzleType+'_'+puzzleArrangement+'/'+puzzleType+'_'+puzzleArrangement+'_'+puzzleSize+'_'+puzzleNumber+'.png')

#TODO: determine settings
puzzleHasBackground = False
puzzlePiecesAmount = 16 # Only needed if not with background
puzzleHasJigSaw = False


### STEP 2: CUT OUT ALL THE PIECES
if puzzleHasBackground: ### Puzzle with background
    ## Create the mask to get pieces out of the background
    puzzle.createMask()
    puzzle.showImage(windowName="original", addWaitKey=False)

    ## Now we got the mask, let's get the pieces of the puzzle out
    puzzle.findPuzzlePieces()
    puzzle.cutAllPiecesOut()

    puzzle.showAllPieces()

else: ### Puzzle without background
    puzzle.cutPuzzlePieces(puzzlePiecesAmount)

    #puzzle.showAllPieces()



### STEP 3: MATCH THE PIECES
if puzzleHasJigSaw: # JigSaw
    print("Not Implemented")

else: #No JigSaw
    puzzle.solvePuzzle()
    ##print(matches[matchid[0]][matchid[1]][matchid[2]][matchid[3]])
