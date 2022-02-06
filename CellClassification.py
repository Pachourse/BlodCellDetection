# CellClassification.py
#example : python CellClassification.py ../BloodCell-Detection-Datatset/test/images/BloodImage_00386_jpg.rf.c708422b2d9c642f200a853e70012850.jpg

import sys
import os
import cv2
import matplotlib.pyplot as plt
import wget
import zipfile
import numpy as np
from lib import WhiteBloodCellClassification as WBCC
wbcc = WBCC()

#params
customResizeBool = True
url = 'http://viallet.me/model.zip'
resultsPath = "/home/pavielschertzer/yolov5/runs/detect/exp4"

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

def customResize(img) : 
    # resizing image in 128 by 128
    img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
    #plt.show(img)
    print('New Dimensions : ', img.shape)

    if img.shape != (128, 128, 3) :
        raise Exception("ERROR : resize didn't work")
    else :
        print("OK - image resized")

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

    if (len(argv)) < 2 :
        raise Exception("ERROR : please add file path as input")
    path = argv[1]
    


    # list files in result folder

    # calling AI on image
    checkInstall()
    #wwc.get_core(resize(raw_test + c + '/' + f, open_file=True))
    data = wbcc.get_core(wbcc.resize(path, open_file=True, crop=False, resize=True))
    data = np.expand_dims(data, axis=0)
    r = wbcc.predict(data)
    r = str(r)[2:-2].split(' ')
    for i in range(len(r)) :
        r[i] = float(r[i])
    # print(r)
    # print(max(r))
    print(wbcc.get_type(findSplit(r)))
    
    # # returns information
    


    # #image openning and resizing
    # try : 
    #     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # except Exception :
    #     raise Exception("ERROR : can't open file")
    # print('Original Dimensions : ',img.shape)

    # if customResizeBool :
    #     img = customResize(img)
    #     img, old = WBCC.get_grayscale(img)



    # for imageFile in os.listdir(resultsPath) :
    #     print(imageFile)
    #     f = open(os.path.join(os.path.join(os.path.join(resultsPath, "labels")), imageFile[:-4] + '.txt'))
    #     im = Image.open(os.path.join(os.path.join(os.path.join(pathWork, "images")), imageFile))
    #     width, height = im.size
    #     number=0
    #     for label in f :
    #         print("----")
    #         catego, x, y, dx, dy = label.split(' ')

    #         if float(dx) == 0 or float(dy) == 0 : 
    #             print("COUCOU")
    #             continue

    #         catego = int(catego)
    #         if catego != 
    #         x = int(float(x) * width)
    #         y = int(float(y) * height)
    #         dx = int((float(dx) * width) / 2)
    #         dy = int((float(dy) * height) / 2)

    #         left = x - dx
    #         right = x + dx
    #         top = y - dy #- because 0:0 on the left top corner
    #         bottom = y + dy


    #         print(str((os.path.join(os.path.join(os.path.join(pathWork, "images")), imageFile))))
    #         print(label)
    #         print(width, height, left, right, top, bottom)
    #         im1 = im.crop((left, top, right, bottom))
    #         #im1 = im.crop((x, y, x + dx, x + dy))
    #         im1.save("transform/" + str(partPath) + "--" + str(catego) + "--" + str(imageFile) + str(number) + ".cropped", "JPEG")
    #         number += 1

    #     f.close()
    #     im1.close()