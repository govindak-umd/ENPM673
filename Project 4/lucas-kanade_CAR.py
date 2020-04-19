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

def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)

def Average(lst):
    return sum(lst) / len(lst)

def LucasKanadeAffine(It, It1, threshold=0.005, iters=15):
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
    M = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.asarray([0.0] * 6)
    I = M

    # Iterate
    # for i in range(iters):
        # Step 1 - Warp image
    for i in range(iters):
        warp_img = affine_transform(It1, np.flip(M)[..., [1, 2, 0]])

        # Step 2 - Compute error image with common pixels
        mask = affine_transform(np.ones(It1.shape), np.flip(M)[..., [1, 2, 0]])
        error_img = (mask * It) - (mask * warp_img)
        #error_img = It-warp_img
        # Step 3 - Compute and warp the gradient
        gradient = np.dstack(np.gradient(It1)[::-1])
        gradient[:, :, 0] = affine_transform(gradient[:, :, 0], np.flip(M)[..., [1, 2, 0]])
        gradient[:, :, 1] = affine_transform(gradient[:, :, 1], np.flip(M)[..., [1, 2, 0]])
        warp_gradient = gradient.reshape(gradient.shape[0] * gradient.shape[1], 2).T

        # Step 4 - Evaluate jacobian parameters
        H, W = It.shape
        Jx = np.tile(np.linspace(0, W - 1, W), (H, 1)).flatten()
        Jy = np.tile(np.linspace(0, H - 1, H), (W, 1)).T.flatten()

        # Step 5 - Compute the steepest descent images
        steepest_descent = np.vstack([warp_gradient[0] * Jx, warp_gradient[0] * Jy,
                                      warp_gradient[0], warp_gradient[1] * Jx, warp_gradient[1] * Jy,
                                      warp_gradient[1]]).T

        # Step 6 - Compute the Hessian matrix
        hessian = np.matmul(steepest_descent.T, steepest_descent)

        # Step 7/8 - Compute delta P
        delta_p = np.matmul(np.linalg.inv(hessian), np.matmul(steepest_descent.T, error_img.flatten()))

        # Step 9 - Update the parameters
        p = p + delta_p
        M = p.reshape(2, 3) + I
        # Test for convergence
        if np.linalg.norm(delta_p) <= threshold:
            break


    # print('%d %.4f'%(i, np.linalg.norm(delta_p)))
    return M



reduction = 0.25
## car
rect_coordinates = [(int(reduction*70),int(reduction*51)), (int(reduction*177), int(reduction*138))]

rect = np.array([rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])
rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))
first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
# first_image = gamma_correction(first_image, 0.8)
first_image = cv2.pyrDown(first_image)
first_image = cv2.pyrDown(first_image)
# first_image = cv2.pyrDown(first_image)
# first_image = cv2.pyrDown(first_image)
# first_image = gamma_correction(first_image, 0.7)

count = 0

img_list = []
# first_image = cv2.resize(first_image, dimen, interpolation=cv2.INTER_AREA)

gamma = 0.7
for next_img in img_array:
    next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    next_img_untouched = next_img.copy()
    # next_img = gamma_correction(next_img, 0.8)
    next_img = cv2.pyrDown(next_img)
    next_img = cv2.pyrDown(next_img)
    # next_img = cv2.pyrDown(next_img)
    # next_img = cv2.pyrDown(next_img)
    next_img = gamma_correction(next_img, gamma)
    # next_img = cv2.pyrDown(next_img)

    layers = 4
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # cl1 = clahe.apply(next_img)

    p = LucasKanadeAffine(first_image, next_img)
    newrect1 = np.matmul(p, rect1)
    newrect2 = np.matmul(p, rect2)
    cv2.rectangle(next_img_untouched, (int(layers*newrect1[0]), int(layers*newrect1[1])), (int(layers*newrect2[0]), int(layers*newrect2[1])), (0, 255, 0), 2)

    # small_count  = 0
    # if int(layers*newrect1[0]) > int(layers*newrect2[0]):
    #     a = int(layers*newrect2[0])
    #     b = int(layers*newrect1[0])
    # else:
    #     a = int(layers*newrect1[0])
    #     b = int(layers*newrect2[0])
    # if int(layers*newrect1[1])> int(layers*newrect2[1]):
    #     c = int(layers*newrect2[1])
    #     d = int(layers*newrect1[1])
    # else:
    #     c = int(layers*newrect1[1])
    #     d = int(layers*newrect2[1])
    #
    # val = []
    #
    # for i in next_img_untouched[a:b,c:d]:
    #     for c in i:
    #         val.append(c)
    # try:
    #     avg = Average(val)
    #     print(count ,avg,gamma)
    # except:
    #     avg = 150.0
    #     print(count,avg,gamma)
    val2 = []
    for i in next_img_untouched:
        for c in i:
            val2.append(c)
    try:
        avg2 = Average(val2)
        print('count > ', count, 'IMAGE_AVG = ',avg2)
    except:
        avg2 = 150.0
        print('count > ', count, 'IMAGE_AVG = ',avg2)
    if avg2 < 110.0 or avg2 == 150.0:
        gamma = 0.7
    elif avg2 >
    else:
        gamma = 1.0
    # if avg < 110.0 or avg == 150.0:
    #     gamma = 0.7
    # else:
    #     gamma = 1.0


    # print('small_count' , small_count)
        # print(count , '> > ' , Average(i))
    cv2.imwrite("all_new_imgs_CAR/%d.jpg" % count, next_img_untouched)
    img_list.append(next_img_untouched)

    count += 1



out = cv2.VideoWriter('all_new_imgs_CAR/CAR4.avi', cv2.VideoWriter_fourcc(*'XVID'), 5.0, (360, 240))
for image in img_list:
    out.write(image)
    cv2.waitKey(10)
out.release()