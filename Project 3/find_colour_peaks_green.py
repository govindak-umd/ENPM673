import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import imageio


def find_peaks_green(a):
    x = []
    for i in a:
        x.append(i[0])
    x = np.array(x)
    max = np.max(x)
    length = len(a)
    ret = []
    for i in range(length):
        ispeak = True
        if i - 1 > 0:
            ispeak &= (x[i] > 1.2 * x[i - 1])  # 1.15 greater than the neighbour
        if i + 1 < length:
            ispeak &= (x[i] > 1.2 * x[i + 1])  # 1.15 greater than the neighbour

        ispeak &= (x[i] > 0.2 * max)  # percentage greater than the max pixel value
        if ispeak:
            ret.append(i)
    return ret


final_peaks_green_list = []
count = 0
k_list = []
all_green = []
# final_list = []
for image_path in glob.glob("buoy_Green\\Train\\*.png"):
    image_name_g = image_path[17:-4]
    img = imageio.imread(image_path)
    green_values = img[:,:,1]

    for green_pix_intensity in green_values:
        all_green.append(green_pix_intensity[0])

    img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_LINEAR)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):

        if i == 1:
            # plt.figure()
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            intensities = find_peaks_green(histr)
            k_list.append(len(intensities))
            for i in intensities:
                final_peaks_green_list.append(i)
k = sum(k_list)/len(k_list)
print(final_peaks_green_list)
