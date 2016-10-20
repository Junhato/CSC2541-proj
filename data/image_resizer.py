from PIL import Image
import glob, os, sys

path = "calming/images/"
dirs = os.listdir(path)

def resize():
    for item in dirs:
        print path + item
        if os.path.isfile(path+item):
            im = Image.open(path + item)
            f, e = os.path.splitext(path + item)
            imResize = im.resize((32,32), Image.ANTIALIAS)
            imResize.save(path + item)

def rename():
    for i in range(len(dirs)):
        item = dirs[i]
        if os.path.isfile(path+item):
            f, e = os.path.splitext(path + item)
            os.rename(path + item, path + "image" + str(i) + e)       

if __name__ == "__main__":
    resize()
