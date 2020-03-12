#!/usr/bin/env python
# coding: utf-8

# # ENPM 673 | Project 2 | Question 2 | Part 1

# In[1]:


#importing all necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
import time


# In[2]:


#source points to be warped
pts_src = np.array([[353,402], [490,323], [742,323],[805,402]])
#destination points to be warped towards
pts_dst = np.array([[410,511],[410, 0],[780, 0],[780,511]])


# In[3]:


#fucntion to generate videos
image_folder = 'C:\\Users\\govin\\OneDrive - University of Maryland\\UMD DOCS\Semester 2\\ENPM673\\Project 2\\data\\'
#list of all the image names
images = [image for image in os.listdir(image_folder) if image.endswith(".png")]
#new empty array in which the images after all processes will be stored
new_arr=[]
#for every image in the list 'images'
for image in images:
    #read the image
    img = cv2.imread(image,1)
    #find homography of the image
    h, status = cv2.findHomography(pts_src, pts_dst)
    #warp the image
    warped = cv2.warpPerspective(img, h, (img.shape[1],img.shape[0]))
    #grayed the warped image
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    #threshed the warped image
    ret,thresh = cv2.threshold(gray_warped,240,255,cv2.THRESH_BINARY)
    #number of column numbers of the thresholded image
    original_column = thresh.shape[1]
    #starting point of the relevant working area of the thresholded image
    col_start = 3*thresh.shape[1]//10
    #ending point of the relevant working area of the thresholded image
    col_end =60*thresh.shape[1]//100
    #new size of the thresholded image | cropped thresholded image
    thresh = thresh[:,col_start:col_end]
    #the part of the thresholded image that was cut-off on the left side
    start_img = np.zeros((thresh.shape[0],col_start,3),np.uint8)
    #the part of the thresholded image that was cut-off on the right side
    end_img = np.zeros((thresh.shape[0],original_column-col_end,3),np.uint8)
    #finding the histogram
    histogram = np.sum(thresh, axis=0)
    try:
        #checking the base points
        
            #checking the midpoint
        midpoint = np.int(histogram.shape[0]//2)
        
            #checking the left side
        leftx_base = np.argmax(histogram[0:midpoint])
        
            #checking the right side
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        #points that are non-zero
        nonzero_points=thresh.nonzero()
        #empty list of the coordinate points on the left side, that is the left lane
        list_of_coord_left=[]
        #empty list of the coordinate points on the right side, that is the right lane
        list_of_coord_right=[]
        len_nonzero_points=len(nonzero_points[0])#can be len(nonzero_points[1] as well
        
        #sorting out the points accordingly and adding them to either the left side
        #or the right list respectively
        for i in range(len_nonzero_points):
            if nonzero_points[1][i]>midpoint:
                list_of_coord_right.append((nonzero_points[1][i],nonzero_points[0][i]))
            elif nonzero_points[1][i]<midpoint:
                list_of_coord_left.append((nonzero_points[1][i],nonzero_points[0][i]))
        #converting the thresholded image from gray to bgr layer    
        thresh_converted= cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR) 
        #taking out points to draw the rectangles, along the line of the lane on the left side 
        #of  the road
        list_of_coord_left_skipped=np.linspace(list_of_coord_left[0],list_of_coord_left[-1],9)
        #taking out points to draw the rectangles, along the line of the lane on the right side 
        #of  the road
        list_of_coord_right_skipped=np.linspace(list_of_coord_right[0],list_of_coord_right[-1],9)
        
        #initializing an empty list
        final_left_skipped=[]
        for coord_l in list_of_coord_left_skipped:
            final_left_skipped.append((int(coord_l[0]),int(coord_l[1])))

        final_right_skipped=[]
        for coord_r in list_of_coord_right_skipped:
            final_right_skipped.append((int(coord_r[0]),int(coord_r[1])))
            
        #initializing an empty list
        L_H=[]
        for y_l in final_left_skipped:
            L_H.append(y_l[1])
        L_H.sort()
        h_rectangle_l=max(L_H)-min(L_H)
        h_rectangle_l=h_rectangle_l//9
    
        #initializing an empty list
        R_H=[]
        for r_l in final_right_skipped:
            R_H.append(r_l[1])
        R_H.sort()
        #declaring the height of the rectangle
        h_rectangle_r=max(R_H)-min(R_H)
        h_rectangle_r=h_rectangle_r//9
        #width of the rectangle is chosen experimentally
        width=14#experimantally
        
        #drawing the rectangle on the left lane
        for i in final_left_skipped:
            cv2.rectangle(thresh_converted,(i[0]-width,i[1]-(h_rectangle_l//2)),(i[0]+width,i[1]+(h_rectangle_l//2)),[0,0,255],2,cv2.LINE_AA)
        
        #drawing the rectangle on the right lane
        for j in final_right_skipped:
            cv2.rectangle(thresh_converted,(j[0]-width,j[1]-(h_rectangle_r//2)),(j[0]+width,j[1]+(h_rectangle_r//2)),[0,255,0],2,cv2.LINE_AA)
        
        #drawing lines along the path
        left_points = np.array(final_left_skipped)
        right_points = np.array(final_right_skipped)    
        left_points = left_points.reshape((-1,1,2))
        right_points = right_points.reshape((-1,1,2))
        cv2.polylines(thresh_converted,[left_points],True,(0,255,255))
        cv2.polylines(thresh_converted,[right_points],True,(0,255,255))    
        #declaring the points for creating a shaded polygon on the road
        points = np.array([[final_left_skipped[0][0],final_left_skipped[0][1]], 
                           [final_right_skipped[0][0],final_right_skipped[0][1]], 
                           [final_right_skipped[-1][0],final_right_skipped[-1][1]],
                           [final_left_skipped[-1][0],final_left_skipped[-1][1]]])
        # points.dtype => 'int64'
        cv2.fillPoly(thresh_converted, np.int32([points]), [255,0,255])  
        #inverse of h, the homography matrix, necessary for the
        #unwarping
        h_inv = np.linalg.inv(h)
        #extending the image shape, by adding np.zeros or blank image
        #equivalent to the sizes of the image that was cut-off earlier
        thresh_converted = np.concatenate((start_img,thresh_converted,end_img), axis=1)
        #unwarping the image
        unwarp = cv2.warpPerspective(thresh_converted, h_inv, (thresh_converted.shape[1], thresh_converted.shape[0]))
        #performing weighted image addiition
        unwarped_final = cv2.addWeighted(img, 1, unwarp, 0.5, 0)
        #adding the images to an array  oustide for further processing
        new_arr.append(unwarped_final)
    except:
        pass


# In[4]:


#iterator
d=0
for i in new_arr:
    #declaring the file name of every file that is to
    #added to the path specified by the user
    filename = "proj_2_q_2_1/file_%d.png"%d
    #write the image in a folder, named as 'grayed'
    cv2.imwrite(filename,i)
    #increment the iterator
    d+=1


# In[5]:


#folder to take every image from
image_folder = 'C:\\Users\\govin\\OneDrive - University of Maryland\\UMD DOCS\Semester 2\\ENPM673\\Project 2\\proj_2_q_2_1\\'
#name of the video file that is to be output
video_name = 'project2_lane_detection_straight_line.avi'
#list of images
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
#video file
video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    #writing the video onto the disk
    video.write(cv2.imread(os.path.join(image_folder, image)))
#closing the video file
cv2.destroyAllWindows()
#releasing the captured frames
video.release()


# In[ ]:




