import urllib

if __name__ == "__main__":
	for i in range(5221, 5401, 5):
		url = "http://i0.wp.com/www.caps.media/201/3-now-you-see-me/full/now-you-see-me-movie-screencaps.com-%d.jpg?w=200" % i
		urllib.urlretrieve(url, "now-you-see-me-" + str(i) + ".jpg")