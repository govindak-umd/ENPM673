# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 01:15:03 2020

@author: nsraj
"""

import cv2
import numpy as np
from PIL import Image
import glob
from os import listdir
import imageio
import imutils
from imutils import contours


# In[8]:


yellow_mean = np.load('yellow_means.npy')
yellow_covar = np.load('yellow_covar.npy')
yellow_weights = np.load('yellow_weights.npy')
k_yellow = 2
rad_yellow = 5
thresh_yellow = 1.2


# In[9]:


red_mean = np.load('red_means.npy')
red_covar = np.load('red_covar.npy')
red_weights = np.load('red_weights.npy')
k_red = 3
thresh_red = 1.8
rad_red = 5


# In[10]:


green_mean = np.load('green_means.npy')
green_covar = np.load('green_covar.npy')
green_weights = np.load('green_weights.npy')
k_green = 3
rad_green_1 = 7
rad_green_2 = 9
thresh_green = 1.6


# In[11]:


prev_cnts = 0


# In[12]:
def gaussian_pdf(data,mean,covar):
    data_mean = np.matrix(data-mean)
    covar_inv = np.linalg.pinv(covar)
    pdf = (2.0 * np.pi) ** (-len(data[1]) / 2.0) * (1.0 /np.linalg.det(covar) ** 0.5) *\
            np.exp(-0.5 * np.sum(np.multiply(data_mean*covar_inv,data_mean),axis=1))
    return pdf

def yellow_gmm(frame,K,updated_mean,updated_covar,updated_weights):
    curr_image = frame
    img_x = curr_image.shape[0]
    img_y = curr_image.shape[1]
    image_1 = curr_image[:,:,1].ravel()
    image_2 = curr_image[:,:,2].ravel()
    image = np.concatenate((image_1,image_2),axis=0)
    img = np.reshape(image,(image.shape[0],1))
    prob = np.zeros((img.shape[0],K))
    for j in range(K):
        #calculate the likelihood
        prob[:,j:j+1] = updated_weights[j]*gaussian_pdf(img,updated_mean[j],updated_covar[j])
    #calculate the prob sum which is a 1d array
    sum_prob = np.sum(prob, axis = 1)
    green_prob = sum_prob[:img_x*img_y]
    red_prob = sum_prob[img_x*img_y:]    
    combined_prob = np.add(green_prob,red_prob)    
    combined_prob[red_prob>np.max(red_prob)/1.2] = 255
    output = np.zeros_like(frame)
    pixel_prob =  np.reshape(combined_prob,(img_x,img_y))
    #Assign the calulated probablities to every pixel in the red channel and green channel
    output[:,:,2]= pixel_prob
    output[:,:,1]= pixel_prob
    #blur = cv2.GaussianBlur(output,(5,5),0)
    #Do filtering and edge detection to detect the buoys
    blur = cv2.medianBlur(output,5)
    edged = cv2.Canny(blur,20,255 )
    #Detect contours with circle shape
    cnts,h = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:  
        (cnts_sorted, boundingBoxes) = contours.sort_contours(cnts)
        prev_cnts = cnts
    except:
        (cnts_sorted, boundingBoxes) = contours.sort_contours(prev_cnts)
    hull = cv2.convexHull(cnts_sorted[0])
    (x,y),radius = cv2.minEnclosingCircle(hull)
    #if greater than a threshold radius, display the detected buoy
    if radius > 5:
        #draw circle over the buoy
        cv2.circle(curr_image,(int(x),int(y)),int(radius+3),(0,200,255),8)
        reshaped = cv2.resize(curr_image,(640,480),interpolation=cv2.INTER_LINEAR)
        return reshaped
    else:
        reshaped = cv2.resize(curr_image,(640,480),interpolation=cv2.INTER_LINEAR)
        return reshaped


# In[13]:


def green_gmm(frame,K,updated_mean,updated_covar,updated_weights):
    curr_image = frame
    img_x = curr_image.shape[0]
    img_y = curr_image.shape[1]
    img = curr_image[:,:,1]
    #Take the red channel and reshape it to 1-d array of pixel intensities
    img = np.reshape(img, (img_x*img_y,1))
    prob = np.zeros((img.shape[0],K))
    for j in range(K):
        #calculate the likelihood
        prob[:,j:j+1] = updated_weights[j]*gaussian_pdf(img,updated_mean[j],updated_covar[j])
    #calculate the prob sum which is a 1d array
    sum_prob = np.sum(prob, axis = 1)
    pixel_probabilities = np.reshape(sum_prob,(img_x,img_y))
    #Choose a threshold prob for assigning to the pixels
    pixel_probabilities[pixel_probabilities>np.max(pixel_probabilities)/1.9] = 255
    #create an empty image of size of current frame
    output = np.zeros_like(frame)
    #Assign the calulated probablities to every pixel in the red channel
    output[:,:,1] = pixel_probabilities
    #blur = cv2.GaussianBlur(output,(5,5),0)
    #Do filtering and edge detection to detect the buoys
    blur = cv2.medianBlur(output,5)
    edged = cv2.Canny(blur,15,255 )
    #Detect contours with circle shape
    cnts,h = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (cnts_sorted, boundingBoxes) = contours.sort_contours(cnts,method="right-to-left")
    hull = cv2.convexHull(cnts_sorted[0])
    (x,y),radius = cv2.minEnclosingCircle(hull)
    #if greater than a threshold radius, display the detected buoy
    if 7< radius < 7.2:
     
        #draw circle over the buoy
        cv2.circle(curr_image,(int(x),int(y)),int(radius+3),(0,255,0),8)
        reshaped = cv2.resize(curr_image,(640,480),interpolation=cv2.INTER_LINEAR)
        images.append(reshaped)
        return reshaped
    else:
        reshaped = cv2.resize(curr_image,(640,480),interpolation=cv2.INTER_LINEAR)
        images.append(reshaped)
        return reshaped


# In[14]:


def red_gmm(frame,K,updated_mean,updated_covar,updated_weights):
    global prev_cnts
    curr_image = frame
#     prev_cnts = 0
    img_x = curr_image.shape[0]
    img_y = curr_image.shape[1]
    img = curr_image[:,:,2]
    #Take the red channel and reshape it to 1-d array of pixel intensities
    img = np.reshape(img, (img_x*img_y,1))
    prob = np.zeros((img_x*img_y,K))
    for j in range(K):
        #calculate the likelihood
        prob[:,j:j+1] = updated_weights[j]*gaussian_pdf(img,updated_mean[j],updated_covar[j])
    #calculate the prob sum which is a 1d array
    sum_prob = np.sum(prob, axis = 1)
    pixel_probabilities = np.reshape(sum_prob,(img_x,img_y))
    #Choose a threshold prob for assigning to the pixels
    pixel_probabilities[pixel_probabilities>np.max(pixel_probabilities)/1.8] = 255
    #create an empty image of size of current frame
    output = np.zeros_like(frame)
    #Assign the calulated probablities to every pixel in the red channel
    output[:,:,2] = pixel_probabilities
    #blur = cv2.GaussianBlur(output,(5,5),0)
    #Do filtering and edge detection to detect the buoys
    blur = cv2.medianBlur(output,5)
    edged = cv2.Canny(blur,20,255 )
    #Detect contours with circle shape
    cnts,h = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:  
        (cnts_sorted, boundingBoxes) = contours.sort_contours(cnts)
        prev_cnts = cnts
    except:
        (cnts_sorted, boundingBoxes) = contours.sort_contours(prev_cnts)
    hull = cv2.convexHull(cnts_sorted[0])
    (x,y),radius = cv2.minEnclosingCircle(hull)
    #if greater than a threshold radius, display the detected buoy
    if radius > 5:
        #draw circle over the buoy
        cv2.circle(curr_image,(int(x),int(y)),int(radius+3),(0,0,255),8)
        reshaped = cv2.resize(curr_image,(640,480),interpolation=cv2.INTER_LINEAR)
        return reshaped
    else:
        reshaped = cv2.resize(curr_image,(640,480),interpolation=cv2.INTER_LINEAR)
        return reshaped


# In[ ]:


#video starting 
cap = cv2.VideoCapture("detectbuoy.avi")
images = []
while (cap.isOpened()):
    #check if frame read is true
    ret, frame = cap.read()
    if ret == False:
        print("Exit!")
        cv2.destroyAllWindows()
        break    
    curr_image = frame
    curr_image_RED = curr_image[:,:,2][0]
    curr_image_GREEN = curr_image[:,:,1][0]
    R = red_gmm(curr_image,k_red,red_mean,red_covar,red_weights)
    Y = yellow_gmm(curr_image,k_yellow,yellow_mean,yellow_covar,yellow_weights)
    G =  green_gmm(curr_image,k_green,green_mean,green_covar,green_weights)

    final_combined = G
    cv2.imshow("Combined",final_combined)
    
    if cv2.waitKey(10) == 27:
        cv2.destroyAllWindows()# exit if Escape is hit
        break
# Save the video file in .avi
out = cv2.VideoWriter('combined_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 15, (640, 480))
for image in images:
    out.write(image)
    cv2.waitKey(10)

out.release()

cap.release()