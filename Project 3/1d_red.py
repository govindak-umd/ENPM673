import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
import time
import pylab as pl
from roipoly import roipoly
from PIL import Image
import glob
from os import listdir
from PIL import Image as PImage
import imageio
import imutils
from imutils import contours


from final_colour_peaks_red import final_peaks_red_list
from final_colour_peaks_red import k
from final_colour_peaks_red import all_red

X  = np.array (final_peaks_red_list)
k = int(k)
flag = 0

def pdf(points, mean_value, cov):
    cov=[cov]
    cov_inv = 1/cov[0]
    cov_inv = [cov_inv]
    diff = points - mean_value
    # print("diff=",diff.shape)

    diff_trans =np.transpose(diff)

    N = (2.0 * np.pi) ** (-1 / 2.0) * (1.0 / cov[0] ** 0.5) * \
        np.exp(-0.5 * np.multiply(np.multiply(diff_trans, cov_inv), diff))

    return N

def gaussian():
    global X

    np.random.shuffle(X)


    bins = np.linspace(np.min(X), np.max(X), len(X))

    return bins,X

def em():
    global k
    bins, X = gaussian()
    weights = np.ones((k)) / k
    means = np.random.choice(X, k)
    variances = np.random.random_sample(size=k)
    # visualize the training data
    z=0

    eps = 0.00001

    iterations = 1000
    iterations_completed = 0
    log_likelihood = []


    while iterations_completed<iterations:
        global flag
    # calculate the maximum likelihood of each observation xi
        likelihood_list = []
        log_likelihood_list = []

        # Expectation step
        for j in range(k):
            likelihood_list.append(pdf(X, means[j], np.sqrt(variances[j])))
            log_likelihood_list.append(pdf(X, means[j], np.sqrt(variances[j]))*weights[j])

        likelihood_list = np.array(likelihood_list)
        log_likelihood_list = np.array(log_likelihood_list).T
        likelihood_sum = np.sum(log_likelihood_list,axis=1)
        log_value =  np.sum(np.log(likelihood_sum))
        log_likelihood.append(log_value)


        b = []

        # Maximization step
        for j in range(k):
            # use the current values for the parameters to evaluate the posterior
            # probabilities of the data to have been generanted by each gaussian
            b.append((likelihood_list[j] * weights[j]) / (np.sum([likelihood_list[i] * weights[i] for i in range(k)], axis=0) + eps))

            # updage mean and variance
            means[j] = np.sum(b[j] * X) / (np.sum(b[j]+eps))
            variances[j] = np.sum(b[j] * np.square(X - means[j])) / (np.sum(b[j] + eps))

            # update the weights
            weights[j] = np.mean(b[j])

        if len(log_likelihood) < 2:
            continue
        if np.abs(log_value - log_likelihood[-2]) < eps:
            flag = 1
            plt.figure(figsize=(10, 6))
            plt.xlabel("$x$")
            plt.ylabel("pdf")
            plt.title(" Best result reached at Iteration {}".format(iterations_completed))
            plt.scatter(X, [0.005] * len(X), color='navy', s=30, marker=2, label="Randomly generated data points")
            for index in range(k):
                plt.plot(bins, pdf(bins, means[index], variances[index]), color='red')
            plt.legend(loc='upper left')
            plt.show()
            plt.savefig('EM.png')
            return means,variances,weights
            break
        iterations_completed+=1

def image_testing():
    global k
    global X
    img = cv2.imread('buoy_Orange//Test//frame6.png', 1)
    img_rows = img.shape[0]
    img_cols = img.shape[1]

    likelihood_list_test = []
    img_red_channel = img[:, :, 2]
    img_red_channel = np.reshape(img_red_channel,(img_rows*img_cols,))
    print(img_red_channel.shape)
    means, variances, weights = em()
    for j in range(k):
        likelihood_list_test.append(pdf(img_red_channel, means[j], np.sqrt(variances[j])) * weights[j])
        likelihood_list_test_arr = np.array(likelihood_list_test).T
        likelihood_sum_test = np.sum(likelihood_list_test_arr, axis=1)

    probs = np.reshape(likelihood_sum_test,(img_rows,img_cols))
    probs[probs > np.max(probs) / 1.9] = 255

    test_out = np.zeros_like(img)
    test_out[:,:,0] = probs
    test_out[:,:,1] = probs
    test_out[:,:,2] = probs

    blur = cv2.GaussianBlur(test_out,(3,3),5)
    cv2.imshow("out",test_out)
    edged = cv2.Canny(blur,50,255 )

    cnts,h = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (cnts_sorted, boundingBoxes) = contours.sort_contours(cnts, method="left-to-right")
    hull = cv2.convexHull(cnts_sorted[0])
    (x,y),radius = cv2.minEnclosingCircle(hull)
    if radius > 4: cv2.circle(img,(int(x),int(y)),int(radius),(0,255,0),4)
    return img

fin = image_testing()

print(fin.shape)
cv2.imshow("Final output - RED", fin)
cv2.waitKey(0)
cv2.destroyAllWindows()