#!/usr/bin/env python
# coding: utf-8

# # ENPM 673 | Project 2 | Question 1

# In[1]:


#importing necessary libraries
import cv2
import numpy as np


# In[2]:


#fucntion to adjust the gamma value of the function
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


# In[10]:


#getting the video feed
camera = cv2.VideoCapture('Night Drive - 2689.mp4')
# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1920,1080))

#checking if video is being played
while(camera.isOpened()):
    ret, frame = camera.read()
    
    #blurring the image
    blurred_img = cv2.GaussianBlur(frame,(7,7),0)
    
    #converting the image to HSV 
    img2hsv = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
    hsv_v = img2hsv[:,:,2]
    #finding the CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))#increased values may cause noise
    cl1 = clahe.apply(hsv_v)
    #setting the gamma value, increased values may cause noise
    gamma = 1.4
    cl1= adjust_gamma(cl1, gamma=gamma)
    #adding the last V layer back to the HSV image
    img2hsv[:,:,2] = cl1
    #converting back from HSV to BGR format
    improved_image = cv2.cvtColor(img2hsv, cv2.COLOR_HSV2BGR)
    #showing the image
    cv2.imshow('improved_image',improved_image)
    #writing the video
    out.write(improved_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#releasing the video feed
out.release()
camera.release()
cv2.destroyAllWindows()


# In[ ]:




