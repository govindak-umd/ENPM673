# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:44:21 2020

@author: nsraj
"""

import cv2
import numpy as np
import glob
import pandas as pd
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

#Normalized fundamental matrix
def fundamental_matrix(feat_1,feat_2): 
    
    #compute the centroids
    feat_1_mean_x = np.mean(feat_1[:, 0])
    feat_1_mean_y = np.mean(feat_1[:, 1])
    feat_2_mean_x = np.mean(feat_2[:, 0])
    feat_2_mean_y = np.mean(feat_2[:, 1])
    
    #Recenter the coordinates by subtracting mean
    feat_1[:,0] = feat_1[:,0] - feat_1_mean_x
    feat_1[:,1] = feat_1[:,1] - feat_1_mean_y
    feat_2[:,0] = feat_2[:,0] - feat_2_mean_x
    feat_2[:,1] = feat_2[:,1] - feat_2_mean_y
    
        
    
    #Compute the scaling terms which are the average distances from origin
    s_1 = np.sqrt(2.)/np.mean(np.sum((feat_1)**2,axis=1)**(1/2)) 
    s_2 = np.sqrt(2.)/np.mean(np.sum((feat_2)**2,axis=1)**(1/2))
    
     
    #Calculate the transformation matrices
    T_a_1 = np.array([[s_1,0,0],[0,s_1,0],[0,0,1]])
    T_a_2 = np.array([[1,0,-feat_1_mean_x],[0,1,-feat_1_mean_y],[0,0,1]])
    T_a = T_a_1 @ T_a_2
    
    
    T_b_1 = np.array([[s_2,0,0],[0,s_2,0],[0,0,1]])
    T_b_2 = np.array([[1,0,-feat_2_mean_x],[0,1,-feat_2_mean_y],[0,0,1]])
    T_b = T_b_1 @ T_b_2
    

    #Compute the normalized point correspondences
    x1 = ( feat_1[:, 0].reshape((-1,1)))*s_1
    y1 = ( feat_1[:, 1].reshape((-1,1)))*s_1
    x2 = (feat_2[:, 0].reshape((-1,1)))*s_2
    y2 = (feat_2[:, 1].reshape((-1,1)))*s_2
    
    #-point Hartley
    #A is (8x9) matrix
    A = np.hstack((x2*x1, x2*y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones((len(x1),1))))    
        
        
    #Solve for A using SVD    
    A = np.array(A)
    U,S,V = np.linalg.svd(A)
    V = V.T
    
    #last col - soln
    sol = V[:,-1]
    F = sol.reshape((3,3))
    U_F, S_F, V_F = np.linalg.svd(F)
    
    #Rank-2 constraint
    S_F[2] = 0
    S_new = np.diag(S_F)
    
    #Recompute normalized F
    F_new = U_F @ S_new @ V_F
    F_norm = T_b.T @ F_new @ T_a
    F_norm = F_norm/F_norm[-1,-1]
    return F_norm
        

#RANSAC to estimate the best fundamental matrix corresponding to the inliers
def ransac_best_fundamental_matrix(feat_1,feat_2):
    #RANSAC parameters
    threshold =0.05
    inliers_present= 0
    F_best = []
    #probability of selecting only inliers
    p = 0.99
    N = 2000
    count = 0
    while count < N:
        inlier_count= 0
        random_8_feat_1 = []
        random_8_feat_2 = []
        #generate a set of random 8 points
        random_list = np.random.randint(len(feat_1), size = 8)
        for k in random_list:
            random_8_feat_1.append(feat_1[k])
            random_8_feat_2.append(feat_2[k])
        #Perform 8-point Hartley algorithm to determine F using the generated 8 random points
        F = fundamental_matrix(np.array(random_8_feat_1), np.array(random_8_feat_2))
        ones = np.ones((len(feat_1),1))
        x1 = np.hstack((feat_1,ones))
        x2 = np.hstack((feat_2,ones))
        e1, e2 = x1 @ F.T, x2 @ F
        error = np.sum(e2* x1, axis = 1, keepdims=True)**2 / np.sum(np.hstack((e1[:, :-1],e2[:,:-1]))**2, axis = 1, keepdims=True)
        inliers = error<=threshold
        inlier_count = np.sum(inliers)
        #Record the best F
        if inliers_present <  inlier_count:
            inliers_present = inlier_count
            F_best = F 
        #Iterations to run the RANSAC for every frame
        inlier_ratio = inlier_count/len(feat_1)
        if np.log(1-(inlier_ratio**8)) == 0: continue
        N = np.log(1-p)/np.log(1-(inlier_ratio**8))
        count += 1
    return F_best
    
 #Compute Essential matrix using Fundamental matrix and camera intrinsic parameters   
def Essential_Matrix(F, K):
    #print("F=",F)
    E = K.T@F@K
    U, S, V = np.linalg.svd(E)

    S[0] = 1
    S[1] = 1
    S[2] = 0
    S_new = np.diag(S)
    E_new = U@S_new@V
    #E_new = E_new / E_new[2, 2]

    return (E_new)

#Determine the Projection matrix parameters - R and T
def cameraposestimation(E):
    U_decompose, S_decompose, V_decompose = np.linalg.svd(E)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    #Compute the four R-T solutions
    R1 = U_decompose @ W @ V_decompose
    R2 = U_decompose @ W @ V_decompose
    R3 =  U_decompose @ W.T @ V_decompose
    R4 = U_decompose @ W.T @ V_decompose
    
    
    C1 = U_decompose[:, 2]
    C2 = -U_decompose[:, 2]
    C3 = U_decompose[:, 2]
    C4 = -U_decompose[:, 2]

    if (np.linalg.det(R1) < 0):
        R1 = -R1
        C1 = -C1
    if (np.linalg.det(R2) < 0):
        R2 = -R2
        C2 = -C2
    if (np.linalg.det(R3) < 0):
        R3 = -R3
        C3 = -C3
    if (np.linalg.det(R4) < 0):
        R4 = -R4
        C4 = -C4
    C1 = C1.reshape((3,1))
    C2 = C2.reshape((3,1))
    C3 = C3.reshape((3,1))
    C4 = C4.reshape((3,1))

    return [R1, R2, R3, R4], [C1, C2, C3, C4]

#Given Projection matrix and point correspondences, estimate 3-D point
def point_3d(pt,pt_,R2,C2,K):
    #Find the projection matrices for respective frames
    C1 = [[0],[0],[0]]
    R1 = np.identity(3)
    R1C1 = -R1@C1
    R2C2 = -R2@C2
    #Current frame has no Rotation and Translation
    P1 = K @ np.hstack((R1, R1C1))
    
    #Estimate the projection matrix for second frame using returned R and T values
    P2 = K @ np.hstack((R2, R2C2))
    #P1_T = P1.T
    #P2_T = P2.T	
    X = []
    
    #Solve linear system of equations using cross-product technique, estimate X using least squares technique
    for i in range(len(pt)):
        x1 = pt[i]
        x2 = pt_[i]
        A1 = x1[0]*P1[2,:]-P1[0,:]
        A2 = x1[1]*P1[2,:]-P1[1,:]
        A3 = x2[0]*P2[2,:]-P2[0,:]
        A4 = x2[1]*P2[2,:]-P2[1,:]		
        A = [A1, A2, A3, A4]
        
        # A1 = x1[1]*P1[2,:] - P1[0,:]
        # A2 = P1[0:,] - x1[0]*P1[2,:]
        # A3 = x2[1]*P2[2:,] - P2[0:,]
        # A4 = P2[0:,] - x1[0]*P2[2:,]
        # A = np.vstack([A1,A2,A3,A4])
        U,S,V = np.linalg.svd(A)
        V = V[3]
        V = V/V[-1]
        X.append(V)
    return X

#cheirality condition
def linear_triangulation(pt,pt_, R,C,K):
    #Check if the reconstructed points are in front of the cameras using cheilarity equations
    X1 = point_3d(pt,pt_,R,C,K)
    X1 = np.array(X1)	
    count = 0
    #r3(X-C)>0
    for i in range(X1.shape[0]):
        x = X1[i,:].reshape(-1,1)
        if R[2]@np.subtract(x[0:3],C) > 0 and x[2] > 0: 
            count += 1
    return count




count = 0
fig = plt.figure('Camera Pose estimation using user defined functions')
plt.title('Trajectory of the car')
list_x = []
list_z = []
H_curr = np.identity(4)
start_time = time.time()

for b in range(35,len(images)-1):
    print("------- Frame-----:  "+str(b))

    # Initiate SIFT detector
   
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = images[b]
    
    #print('shape of img1 is :', img1.shape)
    img1_orig = img1
    img2 = images[b+1]
    #print('shape of img2 is :', img2.shape)
    img2_orig = img2
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    #Implement BF matcher
    best_matches = []
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    #Extract best matches 
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
            
    #Perform RANSAC to compute Fundamental matrix for two point correspondences using 8-point Hartley algorithm
    Best_F = ransac_best_fundamental_matrix(features_1, features_2)
    
    #Find Essential matrix from F and K
    E = Essential_Matrix(Best_F, K)
    
    #Estiamate the 4possible camera poses

    R,C = cameraposestimation(E)
    ispoint = 0
    for p in range(4):
        Z = linear_triangulation(features_1,features_2,R[p], C[p],K)
        if ispoint < Z : 
            ispoint, ind = Z, p
     
    #Choose the best R and T
    R = R[ind]
    t = C[ind]
    if t[2] > 0: 
        t = -t
     
    #Construct a homogeneous R-T matrix  
    H = np.hstack([R,t])
    H_norm = np.vstack([H,np.array([[0,0,0,1]]).reshape((1,4))])
    
    #Multiply the current matrix with pose of the previous frame to get the estimate of new R and T
    H_curr = H_curr @ H_norm
    
     #Read the last column x (lateral) and z (depth) estimate values
    x_trans = H_curr[0,-1]
    z_trans = H_curr[2,-1]
    
    #plot x and z coordinates
    plt.scatter(x_trans, -z_trans, color='r') 
    plt.pause(0.01)
    plt.savefig("Output_user/"+str(count)+".png")
    count += 1

print("Time taken->",time.time()-start_time)
# np.save('data_x.npy', np.array(list_x))
# np.save('data_y.npy',np.array(list_z))