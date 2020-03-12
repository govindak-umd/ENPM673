#!/usr/bin/env python
# coding: utf-8

# In[ ]:


ENPM 673 | Project 2 | Question 2 | Part 2


# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv


# In[2]:


#remove distortion using intrinsic camera parameters
def undistorted_version(cropped_img):
    k = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02], [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    dist = np.array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])

    undistorted_img = cv2.undistort(cropped_image, k, dist, None, k)
    return undistorted_img


# In[3]:


#detect white lanes and yello lanes by converting img to HLS space.
def preprocess(undistorted):
    img_hls = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HLS)

    #Yellow lines
    lower_yellow = np.array([20, 120, 80], dtype='uint8')
    upper_yellow = np.array([45, 200, 255], dtype='uint8')
    mask_yellow = cv2.inRange(img_hls, lower_yellow, upper_yellow)

    yellow_line = cv2.bitwise_and(img_hls, img_hls, mask=mask_yellow).astype(np.uint8)

    #White lines
    lower_white = np.array([0, 200, 0], dtype='uint8')
    upper_white = np.array([255, 255, 255], dtype='uint8')
    mask_white = cv2.inRange(img_hls, lower_white, upper_white)

    white_line = cv2.bitwise_and(img_hls, img_hls, mask=mask_white).astype(np.uint8)

    # Bitwise OR
    preprocessed_hls = cv2.bitwise_or(yellow_line, white_line)


    #convert back to BGR to perform warping and histogram
    preprocessed_bgr = cv2.cvtColor(preprocessed_hls, cv2.COLOR_HLS2BGR)
    return preprocessed_bgr


# In[4]:


def hist_slide_wind(warped,preprocessed_bgr,H_inv,start_img,end_img):
    # Choose the number of sliding windows
    
    nwindows = 10
    # Set the width of the windows +/- margin
    margin=35 
    # Set minimum number of pixels found to recenter window
    minpix=50
    histogram = np.sum(warped, axis=0)
    out_img = np.dstack((warped,warped,warped))*255

    midpoint = np.int(histogram.shape[0]/2)

    # Compute the left and right max pixels
    leftx_ = np.argmax(histogram[:midpoint])
    rightx_ = np.argmax(histogram[midpoint:]) + midpoint


    left_lane_pos = leftx_
    right_lane_pos = rightx_
    image_center = int(warped.shape[1]/2)

    offset = predict_turn(leftx_,rightx_,center_img)

    # Use the lane pixels to predict the turn
    #prediction = turn_predict(image_center, right_lane_pos, left_lane_pos)
    # Set height of windows
    window_height = np.int(warped.shape[0]/nwindows)

    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Update current position for each window
    leftx_p = leftx_
    rightx_p = rightx_

    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_down = warped.shape[0] - (window+1)*window_height
        win_y_up = warped.shape[0] - window*window_height
        win_x_left_down = leftx_p - margin
        win_x_left_up = leftx_p + margin
        win_x_right_down = rightx_p - margin
        win_x_right_up = rightx_p + margin 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_left_down) & (nonzerox < win_x_left_up)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_right_down) & (nonzerox < win_x_right_up)).nonzero()[0]
        # Append these indices to the list
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_p = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_p = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 


    if (np.sum(leftx) == 0 or np.sum(lefty) == 0 or np.sum(rightx) == 0 or np.sum(righty) == 0):
        pass

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Extract points from fit
    left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx,
                              ploty])))])

    image_center = preprocessed_bgr.shape[0]/2


    pts = np.hstack((left_line_pts, right_line_pts))
    pts = np.array(pts, dtype=np.int32)
    
    #Cropped image

    color_blend = np.zeros_like(cropped_image).astype(np.uint8)
    cv2.fillPoly(color_blend, pts, (0,255, 0))
    color_blend = np.concatenate((start_img,color_blend,end_img), axis=1)
    #inverse warping
    unwarped = cv2.warpPerspective(color_blend, H_inv, (undistorted.shape[1], undistorted.shape[0]))
    final_img = cv2.addWeighted(undistorted, 1, unwarped, 0.5, 0)
    return final_img,offset
#     except:
#         pass
    #return final_img


# In[5]:


def predict_turn(left_x,right_x,center_img):
    mean_distance_x = left_x + (right_x-left_x)/2
    
    center_offset = center_img - mean_distance_x
    if(center_offset>0):
        return("Right")
    elif(center_offset<0):
        return("left")
    elif((center_offset>8)):
        return("Straight")


# In[6]:


#Define Homography points
pts_src = np.array([[678,141], [796,68], [952,68],[1037,141]])
pts_dst = np.array([[720,361],[720, 0],[995, 0],[995,361]])


# In[7]:


#performing homography
h, status = cv2.findHomography(pts_src, pts_dst)
# for unwarping the image from camera coordinates to world
H_inv = inv(h)


# In[8]:


#getting the video feed
camera = cv2.VideoCapture(r'C:\Users\govin\OneDrive - University of Maryland\UMD DOCS\Semester 2\ENPM673\Project 2\data_2\challenge_video.mp4')
out = cv2.VideoWriter('curved_road_line_detection.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,288))
global center_img
#checking if video is being played
while(camera.isOpened()):
    ret, img = camera.read()
    try:
        cropped_image=img[3*img.shape[0]//5:,:]
        undistorted = undistorted_version(cropped_image)
        preprocessed_bgr = preprocess(undistorted)
        warped = cv2.warpPerspective(preprocessed_bgr, h, (preprocessed_bgr.shape[1],preprocessed_bgr.shape[0]))
        center_img = int(warped.shape[1]/2)
        ##crop the warped image to remove possible adjacent white lines
        original_column = warped.shape[1]
        col_start = 3*warped.shape[1]//10
        col_end =75*warped.shape[1]//100
        warped = warped[:,col_start:col_end]
        start_img = np.zeros((warped.shape[0],col_start,3),np.uint8)
        end_img = np.zeros((warped.shape[0],original_column-col_end,3),np.uint8)
        warped= cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped = cv2.GaussianBlur(warped,(5,5),0)
        final_img,turn = hist_slide_wind(warped,preprocessed_bgr,H_inv,start_img,end_img)
        cv2.putText(final_img, turn, (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,255),2, cv2.LINE_AA)
        cv2.imshow('final_img',final_img)
        out.write(final_img)
        
    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#releasing the video feed
camera.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




