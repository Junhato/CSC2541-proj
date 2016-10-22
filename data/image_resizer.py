from PIL import Image
import glob, os, sys

path = "unlabeled/images/"
dirs = os.listdir(path)

def crop_images():
    for item in dirs:
        print path + item
        if ".jpg" in item and os.path.isfile(path+item):
            im = Image.open(path + item)
            imResize = im.crop((0,0, 200, 83))
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
		if ".jpg" in item and os.path.isfile(path+item):
			im = Image.open(path + item)
			w, h = im.size
			if w < width:
				width = w
			if h < height:
				height = h
	print "w: %d, h: %d" % (width, height)

if __name__ == "__main__":
    crop_images()
