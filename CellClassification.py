# CellClassification.py
#example : python CellClassification.py ../BloodCell-Detection-Datatset/test/images/BloodImage_00386_jpg.rf.c708422b2d9c642f200a853e70012850.jpg

import sys
import os
import cv2
import matplotlib.pyplot as plt
import wget
import zipfile
from lib import WhiteBloodCellClassification as WBCC
wbcc = WBCC()

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
        url = 'http://viallet.me/model.zip'
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
    
    #image openning and resizing
    try : 
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception :
        raise Exception("ERROR : can't open file")
    print('Old dimensions : ',img.shape)

    # image processing
    img, old = WBCC.resize(img)
    print('New dimensions : ',img.shape)

    # calling AI on image
    checkInstall()
    r = wbcc.predict(img)

    # returns information
    print(wbcc.get_type(r))