# This programme is to generate the RGB histograms for all the images in the training data for all the three buoys

import cv2
from matplotlib import pyplot as plt
import glob
import imageio

# Histogram plots for Green

for image_path in glob.glob("buoy_Green\\Train\\*.png"):
	image_name_g =  image_path[17:-4]
	img = imageio.imread(image_path)
	# grab the image channels, initialize the tuple of colors,
	# the figure and the flattened feature vector
	chans = cv2.split(img)
	colors = ("b", "g", "r")
	plt.figure()
	plt.title("'Flattened' Color Histogram")
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")
	features = []
	# loop over the image channels
	for (chan, color) in zip(chans, colors):
		# create a histogram for the current channel and
		# concatenate the resulting histograms for each
		# channel
		hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
		features.extend(hist)
		# plot the histogram
		plt.plot(hist, color = color)
		plt.xlim([0, 256])
	# here we are simply showing the dimensionality of the
	# flattened color histogram 256 bins for each channel
	# x 3 channels = 768 total values -- in practice, we would
	# normally not use 256 bins for each channel, a choice
	# between 32-96 bins are normally used, but this tends
	# to be application dependent

	filename_g = "buoy_Green//rgb_hist_plots//%s.png" % image_name_g
	plt.savefig(filename_g)

# Histogram plots for Yellow

for image_path in glob.glob("buoy_Yellow\\Train\\*.png"):
	image_name_y =  image_path[18:-4]
	img = imageio.imread(image_path)
	# grab the image channels, initialize the tuple of colors,
	# the figure and the flattened feature vector
	chans = cv2.split(img)
	colors = ("b", "g", "r")
	plt.figure()
	plt.title("'Flattened' Color Histogram")
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")
	features = []
	# loop over the image channels
	for (chan, color) in zip(chans, colors):
		# create a histogram for the current channel and
		# concatenate the resulting histograms for each
		# channel
		hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
		features.extend(hist)
		# plot the histogram
		plt.plot(hist, color = color)
		plt.xlim([0, 256])
	# here we are simply showing the dimensionality of the
	# flattened color histogram 256 bins for each channel
	# x 3 channels = 768 total values -- in practice, we would
	# normally not use 256 bins for each channel, a choice
	# between 32-96 bins are normally used, but this tends
	# to be application dependent

	filename_y = "buoy_Yellow//rgb_hist_plots//%s.png" % image_name_y
	plt.savefig(filename_y)

# Histogram plots for Orange

for image_path in glob.glob("buoy_Orange\\Train\\*.png"):
	image_name_o =  image_path[18:-4]
	print(image_name_o)
	img = imageio.imread(image_path)
	# grab the image channels, initialize the tuple of colors,
	# the figure and the flattened feature vector
	chans = cv2.split(img)
	colors = ("b", "g", "r")
	plt.figure()
	plt.title("'Flattened' Color Histogram")
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")
	features = []
	# loop over the image channels
	for (chan, color) in zip(chans, colors):
		# create a histogram for the current channel and
		# concatenate the resulting histograms for each
		# channel
		hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
		features.extend(hist)
		# plot the histogram
		plt.plot(hist, color = color)
		plt.xlim([0, 256])
	# here we are simply showing the dimensionality of the
	# flattened color histogram 256 bins for each channel
	# x 3 channels = 768 total values -- in practice, we would
	# normally not use 256 bins for each channel, a choice
	# between 32-96 bins are normally used, but this tends
	# to be application dependent

	filename_o = "buoy_Orange//rgb_hist_plots//%s.png" % image_name_o
	plt.savefig(filename_o)
