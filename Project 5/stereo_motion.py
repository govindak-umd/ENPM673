# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:39:05 2020

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
import sys
import time
#from skimage.measure import ransac

fx, fy, cx, cy, camera_image, LUT = ReadCameraModel('model/')
Translation = np.zeros((3, 1))
Rotation = np.eye(3)


K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1])
K = np.reshape(K, (3, 3))
#print(K)
images = np.load('image_list.npy')


def normalize_points_transform(feat_1,feat_2):
    
   
    
    feat_1 = np.array(feat_1)
    feat_2 = np.array(feat_2)
    
    feat_1_mean_x = np.mean(feat_1[:, 0])
    feat_1_mean_y = np.mean(feat_1[:, 1])
    feat_2_mean_x = np.mean(feat_2[:, 0])
    feat_2_mean_y = np.mean(feat_2[:, 1])
    
    s_1 = np.sqrt(2.)/ (np.sum(((feat_1[:, 0] - feat_1_mean_x) ** 2 + (feat_1[:, 1] - feat_1_mean_y) ** 2))/len(feat_1)) ** (1 / 2)
    s_2 = np.sqrt(2.) / (np.sum(((feat_2[:, 0] - feat_2_mean_x) ** 2 + (feat_2[:, 1] - feat_2_mean_y) ** 2))/len(feat_2)) ** (1 / 2)
    
    T_a_1 = np.array([[s_1,0,0],[0,s_1,0],[0,0,1]])
    T_a_2 = np.array([[1,0,-feat_1_mean_x],[0,1,-feat_1_mean_y],[0,0,1]])
    
    T_a = T_a_1 @ T_a_2
    
    
    
    T_b_1 = np.array([[s_2,0,0],[0,s_2,0],[0,0,1]])
    T_b_2 = np.array([[1,0,-feat_2_mean_x],[0,1,-feat_2_mean_y],[0,0,1]])
    
    
    T_b = T_b_1 @ T_b_2
    
    # for ind in range(0, len(feat_1)):
    #     feat_1[ind][0] = (feat_1[ind][0] - feat_1_mean_x) * s_1
    #     feat_1[ind][1] = (feat_1[ind][1] - feat_1_mean_y) * s_1
        
    # for ind in range(0, len(feat_2)):
    #     feat_2[ind][0] = (feat_2[ind][0] - feat_2_mean_x) * s_2
    #     feat_2[ind][1] = (feat_2[ind][1] - feat_2_mean_y) * s_2
    new_feat_1 = []
    new_feat_2 = []
    
    feat_1_arr = []
    feat_2_arr = []
    for ind in range(0,len(feat_1)):
        feat_1_arr.append(T_a @ (np.array([[feat_1[ind][0],feat_1[ind][1],1]]).reshape((3,1))))
        feat_2_arr.append(T_b @ (np.array([[feat_2[ind][0],feat_2[ind][1],1]]).reshape((3,1))))
            
   
        
        
    for ind in range(0,len(feat_1)):
        new_feat_1.append([feat_1_arr[ind][0][0],feat_1_arr[ind][1][0]])
        new_feat_2.append([feat_2_arr[ind][0][0],feat_2_arr[ind][1][0]])
        
    
   
    
    
    
    
    
    return new_feat_1,new_feat_2,T_a,T_b


def fundamental_matrix(feat_1,feat_2,T_a,T_b):
    A = []
    
    for k in range(len(feat_1)):
        x1 = feat_1[k][0]
        y1 = feat_1[k][1]
        x2 = feat_2[k][0]
        y2 = feat_2[k][1]
        A.append([x1*x2,x2*y1,x2,y2*x1,y1*y2,y2,x1,y1,1])
    A = np.array(A)
    U,S,V = np.linalg.svd(A)
    V = V.T
    sol = V[:,-1]
    F = sol.reshape((3,3))
    U_F, S_F, V_F = np.linalg.svd(F)
    S_F[2] = 0
    S_new = np.diag(S_F)
    F_new = U_F @ S_new @ V_F
    F_norm = T_b.T @ F_new @ T_a
    F_norm = F_norm/F_norm[-1,-1]
    return F_norm


def ransac_best_fundamental_matrix(feat_1,feat_2,T_a,T_b):
    prob_inliers = 0.99
    outlier_ratio = 0.5
    threshold = 0.5
    inlier_present = 0
    #times = np.log(1 - p) / np.log(1 - ((1 - outlier_ratio) ** 8))
    times = 2000
    Best_F = fundamental_matrix(feat_1[:8], feat_2[:8], T_a, T_b)
    final_feat_1 = []
    final_feat_2 = []
    count = 0
    F_best = []
    confidence = 0.99
    N =1000000
    for i in range(50):
    # while N > count:
        random_8_feat_1 = []
        random_8_feat_2 = []
        best_feat_1 = []
        best_feat_2 = []
        counter = 0
        random_list = np.random.randint(len(feat_1), size = 8)
        for k in random_list:
            random_8_feat_1.append(feat_1[k])
            random_8_feat_2.append(feat_2[k])
        
        f_mat = fundamental_matrix(random_8_feat_1,random_8_feat_2,T_a,T_b)
        
        inlier_count = 0
        for j in range(len(feat_1)):
            new_list = np.array([feat_1[j][0], feat_1[j][1], 1])
            new_list = np.reshape(new_list, (3, 1))
            new_list_2 = np.array([feat_2[j][0], feat_2[j][1], 1])
            new_list_2 = np.reshape(new_list_2, (1, 3))
            distance = abs(new_list_2@f_mat@new_list)
            #print("distance=",distance)
            if distance < threshold:
                inlier_count += 1
                best_feat_1.append(feat_1[j])
                best_feat_2.append(feat_2[j])
        #print("best_feat_1=",feat_1)      
        if inlier_count > inlier_present:
            inlier_present = inlier_count
            Best_F = f_mat
            final_feat_1 = np.array(best_feat_1)
            final_feat_2 = np.array(best_feat_2)
        # I_O_ratio = len(best_feat_1)/len(feat_1)
        # #print("I_)=",I_O_ratio*100)
        # if np.log(1-(I_O_ratio**8)) == 0: 
        #     continue
        # N = np.log(1-confidence)/np.log(1-(I_O_ratio**8))
        # print("N value=", N)
        # count += 1
            
    return final_feat_1,final_feat_2,Best_F

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

def cameraposestimation(E):
    U_decompose, S_decompose, V_decompose = np.linalg.svd(E)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

    R1 = np.matmul(np.matmul(U_decompose, W), V_decompose)
    R2 = np.matmul(np.matmul(U_decompose, W), V_decompose)
    R3 = np.matmul(np.matmul(U_decompose, W.T), V_decompose)
    R4 = np.matmul(np.matmul(U_decompose, W.T), V_decompose)

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

def linear_trainagulation(pt,pt_,R2,C2,K):
    
    C1 = [[0],[0],[0]]
    R1 = np.identity(3)
    P1 = K @ np.hstack((R1, -R1 @ C1))
    P2 = K @ np.hstack((R2, -R2 @ C2))
    #P1_T = P1.T
    #P2_T = P2.T	
    X = []
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

def extract_Rot_and_Trans(pt,pt_, R,C,K):
    X1 = linear_trainagulation(pt,pt_,R,C,K)
    X1 = np.array(X1)	
    count = 0
    for i in range(X1.shape[0]):
        x = X1[i,:].reshape(-1,1)
        if R[2]@np.subtract(x[0:3],C) > 0 and x[2] > 0: count += 1
    return count


first_position = np.identity(4)
original_base_pose = np.identity(4)

Translation = np.zeros((3, 1))
Rotation = np.eye(3)
count = 0
fig = plt.figure('Fig',figsize=(7,5))
fig.suptitle('Project 5 - Visual Odometry')
ax1 = fig.add_subplot(111)
ax1.set_title('Trajectory of the car')
list_x = []
list_z = []
H_curr = np.identity(4)
start_time = time.time()

for b in range(35,len(images)-1):
    print("#######################Frame  "+str(b)+"###############")

# Initiate SIFT detector
    #orb = cv2.xfeatures2d.SIFT_create()
    orb = cv2.xfeatures2d.SIFT_create()
    img1 = images[b]
    
    #print('shape of img1 is :', img1.shape)
    img1_orig = img1
    #img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = images[b+1]
    #print('shape of img2 is :', img2.shape)
    img2_orig = img2
    #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # img1 = img1[150:750, :]
    # img2 = img2[150:750, :]
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2,k=2)
    
    best_matches = []
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    for m,n in matches:
        if m.distance < 0.7*n.distance: best_matches.append(m)

    # # FLANN parameters
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)  # or pass empty dictionary

    # flann = cv2.FlannBasedMatcher(index_params, search_params)

    # des1 = np.float32(des1)
    # des2 = np.float32(des2)

    # matches = flann.knnMatch(des1, des2, k=2)
    # matches = sorted(matches, key = lambda x:x.distance)
    # print(matches)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    features_1 = []
    features_2 = []
    # ratio test as per Lowe's paper
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < (0.7 * n.distance):
    #         matchesMask[i] = [1, 0]
    #         features_1.append(kp1[m.queryIdx].pt)
    #         features_2.append(kp2[m.trainIdx].pt)
    for i, match in enumerate(best_matches):
        features_1.append(kp1[match.queryIdx].pt)
        features_2.append(kp2[match.trainIdx].pt)
            
    feat_1,feat_2,T_a,T_b = normalize_points_transform(features_1,features_2)
    
    inliers_1,inliers_2, Best_F = ransac_best_fundamental_matrix(feat_1, feat_2, T_a, T_b)
    
    E = Essential_Matrix(Best_F, K)
    
    R,C = cameraposestimation(E)
    flag = 0
    for p in range(4):
        Z = extract_Rot_and_Trans(features_1,features_2,R[p], C[p],K)
        if flag < Z : flag, reg = Z, p
        
    R = R[reg]
    t = C[reg]
    if t[2] > 0: t = -t
    
    
    # x_cf = Translation[0]
    # z_cf = Translation[2]
    # Translation += Rotation.dot(t)
    # Rotation = R.dot(Rotation)
    # x_nf = Translation[0]
    # z_nf = Translation[2]
    # list_x.append(x_nf)
    # list_z.append(z_nf)
    H = np.hstack([R,t])
    H_norm = np.vstack([H,np.array([[0,0,0,1]]).reshape((1,4))])
    H_curr = H_curr @ H_norm
    x_trans = H_curr[0,-1]
    z_trans = H_curr[2,-1]
    ax1.scatter(x_trans, -z_trans, color='r')

    #plt.plot([x_cf, x_nf],[-z_cf, -z_nf],'o')
    if count%50 == 0: 
        plt.pause(1)
        plt.savefig("Output/"+str(count)+".png")
    else: plt.pause(0.001)
    count += 1

print("Time taken->",time.time()-start_time)
cv2.waitKey(0)
plt.show()
cv2.destroyAllWindows()
# np.save('data_x.npy', np.array(list_x))
# np.save('data_y.npy',np.array(list_z))