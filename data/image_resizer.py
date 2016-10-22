from PIL import Image
import glob, os, sys

path = "unlabeled/images/"
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

def get_smallest_size():
	width = 500
	height = 500
	for item in dirs:
		if os.path.isfile(path+item):
			im = Image.open(path + item)
			w, h = im.size
			if w < width:
				width = w
			if h < height:
				height = h
	print "w: %d, h: %d" % (width, height)

if __name__ == "__main__":
    get_smallest_size()
