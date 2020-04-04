# This programme is to generate the RGB histograms for all the images in the training data for all the three buoys

import cv2
from matplotlib import pyplot as plt
import glob
import imageio

# Histogram plots for Green

count= 0
for image_path in glob.glob("buoy_Green\\Train\\*.png"):
	image_name_g =  image_path[17:-4]
	img = imageio.imread(image_path)
	img  = cv2.resize(img,(30,30),interpolation=cv2.INTER_LINEAR)
	filename_resized_green = "buoy_Green//RESIZED//file_green_%d.png" % count
	cv2.imwrite(filename_resized_green, img)
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

# Individual Graphs
	filename_g = "buoy_Green//rgb_hist_plots//%s.png" % image_name_g
	plt.savefig(filename_g)

	color = ('b','g','r')
	for i,col in enumerate(color):
		if i == 0:
			plt.figure()
			histr = cv2.calcHist([img],[i],None,[256],[0,256])
			plt.plot(histr,color = col)
			plt.xlim([0,256])
			filename_g = "buoy_Green//B_channel//B_%s.png" % image_name_g
			plt.savefig(filename_g)

		if i == 1:
			plt.figure()
			histr = cv2.calcHist([img],[i],None,[256],[0,256])
			plt.plot(histr,color = col)
			plt.xlim([0,256])
			filename_g = "buoy_Green//G_channel//G_%s.png" % image_name_g
			plt.savefig(filename_g)

		if i == 2:
			plt.figure()
			histr = cv2.calcHist([img],[i],None,[256],[0,256])
			plt.plot(histr,color = col)
			plt.xlim([0,256])
			filename_g = "buoy_Green//R_channel//R_%s.png" % image_name_g
			plt.savefig(filename_g)

		count+=1

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

# Individual Graphs
	filename_y = "buoy_Yellow//rgb_hist_plots//%s.png" % image_name_y
	plt.savefig(filename_y)

	color = ('b', 'g', 'r')
	for i, col in enumerate(color):
		if i == 0:
			plt.figure()
			histr = cv2.calcHist([img], [i], None, [256], [0, 256])
			plt.plot(histr, color=col)
			plt.xlim([0, 256])
			filename_y = "buoy_Yellow//B_channel//B_%s.png" % image_name_y
			plt.savefig(filename_y)

		if i == 1:
			plt.figure()
			histr = cv2.calcHist([img], [i], None, [256], [0, 256])
			plt.plot(histr, color=col)
			plt.xlim([0, 256])
			filename_y = "buoy_Yellow//G_channel//G_%s.png" % image_name_y
			plt.savefig(filename_y)

		if i == 2:
			plt.figure()
			histr = cv2.calcHist([img], [i], None, [256], [0, 256])
			plt.plot(histr, color=col)
			plt.xlim([0, 256])
			filename_y = "buoy_Yellow//R_channel//R_%s.png" % image_name_y
			plt.savefig(filename_y)

# Histogram plots for Orange

for image_path in glob.glob("buoy_Orange\\Train\\*.png"):
	image_name_o =  image_path[18:-4]
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

# Individual Graphs
	filename_o = "buoy_Orange//rgb_hist_plots//%s.png" % image_name_o
	plt.savefig(filename_o)

	color = ('b', 'g', 'r')
	for i, col in enumerate(color):
		if i == 0:
			plt.figure()
			histr = cv2.calcHist([img], [i], None, [256], [0, 256])
			plt.plot(histr, color=col)
			plt.xlim([0, 256])
			filename_o = "buoy_Orange//B_channel//B_%s.png" % image_name_o
			plt.savefig(filename_o)

		if i == 1:
			plt.figure()
			histr = cv2.calcHist([img], [i], None, [256], [0, 256])
			plt.plot(histr, color=col)
			plt.xlim([0, 256])
			filename_o = "buoy_Orange//G_channel//G_%s.png" % image_name_o
			plt.savefig(filename_o)

		if i == 2:
			plt.figure()
			histr = cv2.calcHist([img], [i], None, [256], [0, 256])
			plt.plot(histr, color=col)
			plt.xlim([0, 256])
			filename_o = "buoy_Orange//R_channel//R_%s.png" % image_name_o
			plt.savefig(filename_o)