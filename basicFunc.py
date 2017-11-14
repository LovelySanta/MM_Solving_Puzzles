import random
from datetime import datetime
random.seed(datetime.now())

def getRandomColor(size = 3):
    if size == 3:
        return [random.randint(1,255), random.randint(1,255), random.randint(1,255)]
    elif size == 1:
        return [random.randint(1,255)]
    else:
        return nil
