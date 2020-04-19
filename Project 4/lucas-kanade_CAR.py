# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:07:10 2020

@author: nsraj
"""

import numpy as np
import cv2
import glob
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
img_array = []
for filename in glob.glob('Car4/Car4/img/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)


first_image = img_array[0]
# # now let's initialize the list of reference point 
ref_point = []
crop = False

def LucasKanadeAffine(It, It1):
    '''
    [input]
    * It - Template image
    * It1 - Current image
    * threshold - Threshold for error convergence (default: 0.005)
    * iters - Number of iterations for error convergence (default: 50)

    [output]
    * M - Affine warp matrix [2x3 numpy array]
    '''

    # Initial parameters
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    threshold = 1
    dp = 10
    p = np.zeros(6)
    Iy, Ix = np.gradient(It1)
    x1, y1, x2, y2 = 0, 0, It.shape[1], It.shape[0]
    while np.square(dp).sum() > threshold:

        W = np.array([[1.0 + p[0], p[1], p[2]],
                      [p[3], 1.0 + p[4], p[5]]])

        x1_w = W[0, 0] * x1 + W[0, 1] * y1 + W[0, 2]
        y1_w = W[1, 0] * x1 + W[1, 1] * y1 + W[1, 2]
        x2_w = W[0, 0] * x2 + W[0, 1] * y2 + W[0, 2]
        y2_w = W[0, 0] * x2 + W[0, 1] * y2 + W[0, 2]

        x = np.arange(0, It.shape[0], 1)
        y = np.arange(0, It.shape[1], 1)

        c = np.linspace(x1, x2, It.shape[1])
        r = np.linspace(y1, y2, It.shape[0])
        cc, rr = np.meshgrid(c, r)

        cw = np.linspace(x1_w, x2_w, It.shape[1])
        rw = np.linspace(y1_w, y2_w, It.shape[0])
        ccw, rrw = np.meshgrid(cw, rw)

        spline = RectBivariateSpline(x, y, It)
        T = spline.ev(rr, cc)

        spline1 = RectBivariateSpline(x, y, It1)
        warpImg = spline1.ev(rrw, ccw)

        # compute error image
        # errImg is (n,1)
        err = T - warpImg
        errImg = err.reshape(-1, 1)

        # compute gradient
        spline_gx = RectBivariateSpline(x, y, Ix)
        Ix_w = spline_gx.ev(rrw, ccw)

        spline_gy = RectBivariateSpline(x, y, Iy)
        Iy_w = spline_gy.ev(rrw, ccw)
        # I is (n,2)
        I = np.vstack((Ix_w.ravel(), Iy_w.ravel())).T

        # evaluate delta = I @ jac is (n, 6)
        delta = np.zeros((It.shape[0] * It.shape[1], 6))

        for i in range(It.shape[0]):
            for j in range(It.shape[1]):
                # I is (1,2) for each pixel
                # Jacobiani is (2,6)for each pixel
                I_indiv = np.array([I[i * It.shape[1] + j]]).reshape(1, 2)

                jac_indiv = np.array([[j, 0, i, 0, 1, 0],
                                      [0, j, 0, i, 0, 1]])
                delta[i * It.shape[1] + j] = I_indiv @ jac_indiv

        # compute Hessian Matrix
        # H is (6,6)
        H = delta.T @ delta

        # compute dp
        # dp is (6,6)@(6,n)@(n,1) = (6,1)
        dp = np.linalg.inv(H) @ (delta.T) @ errImg

        # update parameters
        p[0] += dp[0, 0]
        p[1] += dp[1, 0]
        p[2] += dp[2, 0]
        p[3] += dp[3, 0]
        p[4] += dp[4, 0]
        p[5] += dp[5, 0]

    M = np.array([[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]]])
    return M



##bolt
# rect_coordinates = [(int(0.25*269),int(0.25*75)), (int(0.25*300), int(0.25*139))]
##dragon baby
rect_coordinates = [(int(0.5*70),int(0.5*51)), (int(0.5*177), int(0.5*138))]
# rect_coordinates = [(int(70),int(51)), (int(177), int(138))]
# rect_coordinates = [(142, 67), (224, 166)]
#car
# rect_coordinates = [(int(scale_DOWN*70), int(scale_DOWN*51)), (int(scale_DOWN*177), int(scale_DOWN*138))]
rect = np.array([rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])
rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))
first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)

first_image = cv2.pyrDown(first_image)


count = 0

img_list = []
# first_image = cv2.resize(first_image, dimen, interpolation=cv2.INTER_AREA)


for next_img in img_array:
    next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    next_img_untouched = next_img.copy()

    next_img = cv2.pyrDown(next_img)

    layers = 2
    # next_img = cv2.pyrDown(next_img)
    p = LucasKanadeAffine(first_image, next_img)
    # print('p : ', p)
    newrect1 = np.matmul(p, rect1)
    newrect2 = np.matmul(p, rect2)
    cv2.rectangle(next_img_untouched, (int(layers*newrect1[0]), int(layers*newrect1[1])), (int(layers*newrect2[0]), int(layers*newrect2[1])), (0, 255, 0), 2)
    cv2.imwrite("all_new_imgs_CAR/%d.jpg" % count, next_img_untouched)
    img_list.append(next_img_untouched)
    print(count)
    count += 1



out = cv2.VideoWriter('all_new_imgs_CAR/CAR4.avi', cv2.VideoWriter_fourcc(*'XVID'), 5.0, (360, 240))
for image in img_list:
    out.write(image)
    cv2.waitKey(10)
out.release()