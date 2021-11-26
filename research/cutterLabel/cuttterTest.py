from PIL import Image

im = Image.open(r"img.jpg")
width, height = im.size

label = "2 0.7403846153846154 0.7620192307692307 0.18389423076923078 0.25600961538461536"
catego, x, y, dx, dy = label.split(' ')
catego = int(catego)
x = int(float(x) * width)
y = int(float(y) * height)
dx = int((float(dx) * width) / 2)
dy = int((float(dy) * height) / 2)

print(x)
print(y)
print(dx)
print(dy)
print(width)
print(height)

left = x - dx
right = x + dx
top = y - dy #- because 0:0 on the left top corner
bottom = y + dy
im1 = im.crop((left, top, right, bottom))
im1.save(str(catego) + ".cropped", "JPEG")