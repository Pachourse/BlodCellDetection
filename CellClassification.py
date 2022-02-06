# CellClassification.py
#example : python CellClassification.py ../BloodCell-Detection-Datatset/test/images/BloodImage_00386_jpg.rf.c708422b2d9c642f200a853e70012850.jpg
#example : python CellClassification.py ~/BloodCell-Detection-Datatset/test/images/ ~/yolov5/runs/detect/exp4

import sys
import os
import cv2
import matplotlib.pyplot as plt
import wget
import zipfile
import numpy as np
from PIL import Image
from lib import WhiteBloodCellClassification as WBCC
wbcc = WBCC()

#params
cutter = True # True if you need to crope images from Yolo5 and import them in tranfom folder
analyse = True # True if you need to apply AI on cropped images
url = 'http://viallet.me/model.zip'
dataPath = "~/BloodCell-Detection-Datatset/test/images/"
resultsPath = "~/yolov5/runs/detect/exp4"
projectPath = ""
def findSplit(result):
    m = 0.0
    i = 0
    j = 0
    for e in result :
        if e > m:
            m = e
            i = j
        j += 1
    return i

def checkInstall() :
    if "model" in os.listdir() :
        print("model is installed")
        return True
    else :
        print("model isn't installed")
    if "model.zip" in os.listdir() :
        print("zip file already installed")
    else :
        print("downloading model")
        wget.download(url)
    
    #unzip model
    print(os.getcwd(), "model.zip")
    print("unzipping model")
    with zipfile.ZipFile(os.path.join(os.getcwd(), "model.zip"), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(os.getcwd(), ""))
    return True

if __name__ == "__main__" :
    argv = sys.argv

    if (len(argv)) < 3 :
        raise Exception("ERROR : please add absolute path for data and results")

    if argv[1][0] == '~' :
        dataPath = os.path.join(os.path.expanduser('~'), argv[1][1:])
    else :    
        dataPath = argv[1]
    
    if argv[2][0] == '~' :
        resultsPath = os.path.join(os.path.expanduser('~', argv[2][1:]))
    else :
        resultsPath = argv[2]

    projectPath = os.getcwd()

    if cutter :
        for imageFile in os.listdir(resultsPath) :
            if imageFile == "labels" :
                continue
            print(imageFile)
            f = open(os.path.join(os.path.join(resultsPath, "labels"), imageFile[:-4] + '.txt'))
            print(f)
            im = Image.open(os.path.join(os.path.join(dataPath, imageFile)))
            width, height = im.size
            number=0
            for label in f :
                print("----")
                catego, x, y, dx, dy = label.split(' ')

                if float(dx) == 0 or float(dy) == 0 : 
                    print("COUCOU")
                    continue

                catego = int(catego)
                if catego != 2 : #only need WBC
                    continue

                # position calc
                x = int(float(x) * width)
                y = int(float(y) * height)
                dx = int((float(dx) * width) / 2)
                dy = int((float(dy) * height) / 2)

                left = x - dx
                right = x + dx
                top = y - dy #- because 0:0 on the left top corner
                bottom = y + dy
                
                im1 = im.crop((left, top, right, bottom))
                #im1 = im.crop((x, y, x + dx, x + dy))
                im1.save(projectPath + "/transform/"  + str(catego) + "--" + str(imageFile) + str(number) + ".cropped.jpeg", "JPEG")
                im1.close()

            f.close()
            im.close()

    if analyse :
        checkInstall()
        for image in os.listdir("transform/") :
            path = os.path.join(os.path.join(projectPath, "transform/"), image)
            data = wbcc.get_core(wbcc.resize(path, open_file=True, crop=False, resize=True))
            data = np.expand_dims(data, axis=0)
            r = wbcc.predict(data)
            r = str(r)[2:-2].split(' ')
            for i in range(len(r)) :
                r[i] = float(r[i])
            res = wbcc.get_type(findSplit(r))
            a, b = path.split("cropped")
            res = a + res + b
            os.rename(path, res)
