import cv2
import pylab as pl
from roipoly import roipoly
import glob
import imageio

count = 1

for image_path in glob.glob("Training_Frames\*.png"):
    img = imageio.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # for buoy 1
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # for buoy 2
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # for buoy 3
    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    pl.imshow(img1, interpolation='nearest', cmap="Greys")
    pl.colorbar()
    pl.title("left click: line segment         right click: close region")

    # let user draw first ROI
    ROI1 = roipoly(roicolor='r')  # let user draw first ROI
    # show the image with the first ROI
    pl.imshow(img1, interpolation='nearest', cmap="Greys")
    pl.colorbar()
    # ROI1.displayROI()
    points_1 = [x.get_vertices(img1) for x in [ROI1]][0]

    pl.imshow(img2, interpolation='nearest', cmap="Greys")
    pl.colorbar()
    pl.title("left click: line segment         right click: close region")
    # let user draw second ROI
    ROI2 = roipoly(roicolor='b')  # let user draw ROI
    # show the image with both ROIs and their mean values
    pl.imshow(img2, interpolation='nearest', cmap="Greys")
    pl.colorbar()
    # ROI2.displayROI()
    points_2 = [x.get_vertices(img2) for x in [ROI2]][0]

    pl.imshow(img3, interpolation='nearest', cmap="Greys")
    pl.colorbar()
    pl.title("left click: line segment         right click: close region")
    ROI3 = roipoly(roicolor='g')  # let user draw ROI
    # show the image with both ROIs and their mean values
    pl.imshow(img3, interpolation='nearest', cmap="Greys")
    pl.colorbar()
    points_3 = [x.get_vertices(img3) for x in [ROI3]][0]

    # cropping the images

    # Buoy 1
    max_x_1 = 0
    min_x_1 = 10000
    min_y_1 = 10000
    max_y_1 = 0

    # to get the coordinates for getting the cropped
    for (x_coordinate, y_coordinate) in points_1:
        if x_coordinate > max_x_1:
            max_x_1 = int(x_coordinate)
        if x_coordinate < min_x_1:
            min_x_1 = int(x_coordinate)
        if y_coordinate > max_y_1:
            max_y_1 = int(y_coordinate)
        if y_coordinate < min_y_1:
            min_y_1 = int(y_coordinate)
    # cropping the image
    cropped_img_1 = img[min_y_1:max_y_1, min_x_1:max_x_1, :]
    # cv2.imshow('cropped_img_1', cropped_img_1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    filename_yellow = "buoy_Yellow//Train//file_yellow_%d.png" % count
    cv2.imwrite(filename_yellow, cropped_img_1)

    # Buoy 2
    max_x_2 = 0
    min_x_2 = 10000
    min_y_2 = 10000
    max_y_2 = 0
    # to get the coordinates for getting the cropeed
    for (x_coordinate, y_coordinate) in points_2:
        if x_coordinate > max_x_2:
            max_x_2 = int(x_coordinate)
        if x_coordinate < min_x_2:
            min_x_2 = int(x_coordinate)
        if y_coordinate > max_y_2:
            max_y_2 = int(y_coordinate)
        if y_coordinate < min_y_2:
            min_y_2 = int(y_coordinate)
    # cropping the image
    cropped_img_2 = img[min_y_2:max_y_2, min_x_2:max_x_2, :]
    # cv2.imshow('cropped_img_2', cropped_img_2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    filename_orange = "buoy_Orange//Train//file_orange_%d.png" % count
    cv2.imwrite(filename_orange, cropped_img_2)

    # Buoy 3
    max_x_3 = 0
    min_x_3 = 10000
    min_y_3 = 10000
    max_y_3 = 0
    for (x_coordinate, y_coordinate) in points_3:
        if x_coordinate > max_x_3:
            max_x_3 = int(x_coordinate)
        if x_coordinate < min_x_3:
            min_x_3 = int(x_coordinate)
        if y_coordinate > max_y_3:
            max_y_3 = int(y_coordinate)
        if y_coordinate < min_y_3:
            min_y_3 = int(y_coordinate)
    # cropping the image
    cropped_img_3 = img[min_y_3:max_y_3, min_x_3:max_x_3, :]
    # cv2.imshow('cropped_img_3', cropped_img_3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    filename_green = "buoy_Green//Train//file_green_%d.png" % count
    cv2.imwrite(filename_green, cropped_img_3)
    # count incremented to chnage the file names while saving the cropped images
    count += 1
