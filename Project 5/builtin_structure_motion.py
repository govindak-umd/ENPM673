# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:44:21 2020

@author: nsraj
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ReadCameraModel import *
from UndistortImage import *
import random
import time

#Read camera intrinsic parameters
fx, fy, cx, cy, camera_image, LUT = ReadCameraModel('model/')
K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1])
K = np.reshape(K, (3, 3))

#read dataset
images = np.load('image_list.npy')
count = 0
fig = plt.figure('Camera Pose estimation')
plt.title('Trajectory of the car')
list_x = []
list_z = []

#initialize R-T matrix
H_curr = np.identity(4)
start_time = time.time()


#Read images sequentially
for b in range(20,len(images)-1):
    print("------- Frame-----:  "+str(b))

    # Initiate SIFT detector
   
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = images[b]
    img2 = images[b+1]
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    #Implement BF matcher
    best_matches = []
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
     # ratio test as per Lowe's paper
    for m,n in matches:
        if m.distance < 0.7*n.distance: best_matches.append(m)
    matchesMask = [[0, 0] for i in range(len(matches))]

    features_1 = []
    features_2 = []

    #Save the best matches
    for i, match in enumerate(best_matches):
        matchesMask[i] = [1, 0]
        features_1.append(kp1[match.queryIdx].pt)
        features_2.append(kp2[match.trainIdx].pt)
              
    #Find essential matrix using matched features and K
    E,mask = cv2.findEssentialMat(np.array(features_1), np.array(features_2), focal = K[0, 0], pp = (K[0, 2], K[1, 2]), method = cv2.RANSAC, prob = 0.999, threshold = 0.5)
    
    #Recover the best R and T matrices
    points,R, T, mask = cv2.recoverPose(E, np.array(features_1), np.array(features_2),K)
    
    #check for negative condition
    if(np.linalg.det(R) < 0):
        R = -R
        T = -T
    
    #Construct a homogeneous R-T matrix    
    H = np.hstack([R,T])
    H_norm = np.vstack([H,np.array([[0,0,0,1]]).reshape((1,4))])
    
    #Multiply the current matrix with pose of the previous frame to get the estimate of new R and T
    H_curr = H_curr @ H_norm
    
    #Read the last column x (lateral) and z (depth) estimate values
    x_trans = H_curr[0,-1]
    z_trans = H_curr[2,-1]
    
    #plot x and z coordinates
    plt.scatter(x_trans, -z_trans, color='g')
    plt.pause(0.01)
    plt.savefig("Output/"+str(count)+".png")
    
    count += 1

print("Time taken->",time.time()-start_time)
