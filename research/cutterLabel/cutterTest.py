from PIL import Image
import dotenv
import os
import cv2

dotenvFile = dotenv.find_dotenv()
dotenv.load_dotenv(dotenvFile)

dataPath = os.getenv('DATAPATH')

if 'DATAPATH' in os.environ:
    print("env path var : OK")
else:
    raise Exception("ERROR : il faut préciser le chemin vers le dossier de Data dans le .ENV : DATAPATH=\'votre/chemin/abosolu/vers/data\'")

if not os.path.exists("transform") :
    os.mkdir("transform")


# pourcours train test valid/ ...
for partPath in ["train", 'test', "valid"] :
    pathWork = os.path.join(dataPath, partPath)
    # train test valid/ ... chaque image
    for imageFile in os.listdir(os.path.join(pathWork, "images")) :
        f = open(os.path.join(os.path.join(os.path.join(pathWork, "labels")), imageFile[:-4] + '.txt'))
        # train test valid/ ... chaque image ... chaque label d'image
        im = Image.open(os.path.join(os.path.join(os.path.join(pathWork, "images")), imageFile))
        width, height = im.size
        number=0
        for label in f :
            print("----")
            catego, x, y, dx, dy = label.split(' ')

            if float(dx) == 0 or float(dy) == 0 : 
                print("COUCOU")
                continue

            catego = int(catego)
            x = int(float(x) * width)
            y = int(float(y) * height)
            dx = int((float(dx) * width) / 2)
            dy = int((float(dy) * height) / 2)

            left = x - dx
            right = x + dx
            top = y - dy #- because 0:0 on the left top corner
            bottom = y + dy


            print(str((os.path.join(os.path.join(os.path.join(pathWork, "images")), imageFile))))
            print(label)
            print(width, height, left, right, top, bottom)
            im1 = im.crop((left, top, right, bottom))
            #im1 = im.crop((x, y, x + dx, x + dy))
            im1.save("transform/" + str(partPath) + "--" + str(catego) + "--" + str(imageFile) + str(number) + ".cropped", "JPEG")
            number += 1

        f.close()
        im1.close()