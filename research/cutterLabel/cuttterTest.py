from PIL import Image
import dotenv
import os
import cv2

dotenvFile = dotenv.find_dotenv()
dotenv.load_dotenv(dotenvFile)

dataPath = os.getenv('DATAPATH')

# TO DELETE
#im = Image.open(r"img.jpg")
#width, height = im.size


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
            catego, x, y, dx, dy = label.split(' ')
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
            im1 = im.crop((left, top, right, bottom))
            im1.save("transform/" + str(partPath) + "--" + str(catego) + "--" + str(imageFile) + str(number) + ".cropped", "JPEG")
            number += 1

        f.close()
        im1.close()