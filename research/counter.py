import dotenv
import os
import cv2
from PIL import Image

# TEST SCRIPT
# idea : count all labels and images in the repo
dotenvFile = dotenv.find_dotenv()
dotenv.load_dotenv(dotenvFile)

counter0 = 0
counter1 = 0
counter2 = 0
counterX = 0
fileCounter = 0
# parameters
# datapath : path to the data
# DATAPATH='/full/path/from/root/to/data/folder'
dataPath = os.getenv('DATAPATH')

# pourcours train test valid/ ...
for partPath in ["train", 'test', "valid"] :
    pathWork = os.path.join(dataPath, partPath)
    # train test valid/ ... chaque image
    for imageFile in os.listdir(os.path.join(pathWork, "images")) :
        f = open(os.path.join(os.path.join(os.path.join(pathWork, "labels")), imageFile[:-4] + '.txt'))
        # train test valid/ ... chaque image ... chaque label d'image
        for line in f :
            classification, coordX, coordY, sizeX, sizeY  = line.split(' ')
            classification = int(classification)
            if classification == 0 :
                counter0 += 1
            elif classification == 1 :
                counter1 += 1
            elif classification == 2 :
                counter2 += 1
            else :
                counterX += 1
        fileCounter += 1

        f.close()
    print(partPath, counter0, counter1, counter2, counterX, fileCounter)
    counter0 = 0
    counter1 = 0
    counter2 = 0
