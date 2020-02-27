#!/usr/bin/env python
# coding: utf-8

# # PROJECT 1 | Perception for Autonomous Robotics | ENPM673

# # Importing Libraries

# In[1]:


#Importing the necessary Libraries
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')
#not needed if not being run on Jupyter Notebook
import matplotlib.pyplot as plt
from functools import reduce
import operator
import math
import operator
from collections import Counter
import os


# # Function to compute Homography

# In[2]:


# p1 and p2 to be input in the following format
# for example,
# p2 = np.array([[0,0], [200,0], [200,200],[0,200] ])

#function to calculate the homography matrix
def findHomography(p1,p2):
    try:
        A = []
        for i in range(0, len(p1)):
            x, y = p1[i][0], p1[i][1]
            u, v = p2[i][0], p2[i][1]
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
        A = np.asarray(A)
        U, S, Vh = np.linalg.svd(A) #Using SVD file
        L = Vh[-1,:] / Vh[-1,-1]
        H = L.reshape(3, 3)
        return(H)
    except:
        pass


# # Distance Calculation between two points

# In[3]:


def distanceCalc(a,b,c,d): #Calculation of the distance bewteen two points
    dist = np.sqrt(((a-b) ** 2) + ((c-d) ** 2)) 
    return (dist) 


# # Function to compute AR TAG CONTOUR

# In[4]:


#Function to make a list of tuple points into clockwise order
def toClockwise(tup):
    coords =tup
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    t = (sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::1]))) % 360))
    return (t)


# In[5]:


#Function to return contour over the AR tag
def drawARContour(img): 
    img2 =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,threshold = cv2.threshold(img2, 240, 250, 
                                cv2.THRESH_BINARY) 
    contours,_=cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #finding contours
    #using cv2.RETR_TREE 
    contours.pop(0)
    len_of_all_contours = []
    len_of_all_contours = []
    for c in contours:
        len_of_all_contours.append(len(c))
    try: 
        max_ind = len_of_all_contours.index(max(len_of_all_contours))
    except ValueError:
        pass
    #to get the min x and min y of the rectangles outside the QR tag
    all_x= []
    all_y=[]
    for i in range(len(contours[max_ind])):
        for c in contours[max_ind][i]:
            all_x.append(c[0])
            all_y.append(c[1])   
    tup = [(min(all_x),min(all_y)),((max(all_x),min(all_y))),((min(all_x),max(all_y))),((max(all_x),max(all_y)))]  
    rect = toClockwise(tup) 
    for cnt in contours : 
        area = cv2.contourArea(cnt) 
        approx = cv2.approxPolyDP(cnt, 
                                0.009 * cv2.arcLength(cnt, True), True) 
        cv2.drawContours(img, [approx], 0, (0, 0, 255), 2) 
    return(img,rect) #rect is the rectangle coordinates


# # Function to draw the 8 grid

# In[6]:


#Function to draw the 8 grid around the April Tag
def draw8Grid(img): 
    
    quart_x = int(img.shape[0]/8)
    half_x = int(img.shape[0]/2)
    full_x = int(img.shape[0])
    quart_y = int(img.shape[1]/8)
    half_y = int(img.shape[1]/2)
    full_y = int(img.shape[1])
    
    #drawing vertical lines across the tag
    for i in range(1,9):
        cv2.line(img, (quart_x*i,0), (quart_x*i, full_x), (125, 0, 0), 1, 1) 
    #drawing horizontal lines across the tag
    for i in range(1,9):
        cv2.line(img, (0,quart_y*i), (full_y,quart_y*i), (125, 0, 0), 1, 1) 
    return(img)


# # Thershold and maintain the grid

# In[7]:


#Fuction to threshold the 8 grid image and returns it
def thresholdAndDraw(img):
    ret,threshed = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
    resized_image = cv2.resize(threshed, (200, 200))
    eight_thresh = draw8Grid(resized_image)
    return(eight_thresh)



# In[8]:


#Function to determine the points for warping and perspective transformation
def determinePoints(out):
    tl = out[0]
    tr = out[1]
    br = out[2] #Function to cacluate the distance between the points for size of the destination image.
    bl = out[3]
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

    dst = np.array([     #create a destination image of corresponding size
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    return (tl,tr,br,bl,maxWidth,maxHeight,dst)


# # Detects if the bottom right is a black square

# In[9]:


def bottomIsBlackSquare(img):
    val = 0
    bottom_square= img[150:200,150:200] #to check if tag is rotated or not
    list_black_or_white_1 = []
    for i in range(bottom_square.shape[0]):
        for j in range(bottom_square.shape[1]):
                list_black_or_white_1.append(bottom_square[i][j])
    count_black_1  = list_black_or_white_1.count(0)#checking count of black
    count_white_1  = list_black_or_white_1.count(255) #checking count of white
    if count_black_1>3*count_white_1: #changing 2.5 will increase harshness to check for rotation
#         print('Image might be straight here') 
        val==1
#         cv2.imshow('ORIGINAL IMAGE AT THIS POINT',img)
        return(img,1)
    else:
        val==0
        return(img,val)


# # Detects if the 2by2 white square is oriented correctly

# In[10]:


def bottomIsWhiteSquare(img):
    white_orientation= img[100:150,100:150]
    list_black_or_white_1 = []
    for i in range(white_orientation.shape[0]):
        for j in range(white_orientation.shape[1]):
                list_black_or_white_1.append(white_orientation[i][j])
    count_black_1  = list_black_or_white_1.count(0)#checking count of black
    count_white_1  = list_black_or_white_1.count(255) #checking count of white
    if 3*count_black_1<count_white_1:
        return(True)
    else:
        return(False)
        
    
    


# # Function for TAG ID

# In[11]:


#Function to determine TAG ID
def Tag_ID(img):
    tag = [] #empty tag
    center_img = img[75:125,75:125] #choosing the center of the TAG
    for i in range(2):
        for j in range(2):
            small_sq = center_img[(25*i):(25*(i+1)),(25*j):(25*(j+1))] #checking each quarter region of the center
            avg_list = []
            for c in range(25):
                for r in range(25):
                    avg_list.append(small_sq[r][c]) #appedning the pixel values
            #mode to check which pixel is largely present here
            count_black  = avg_list.count(0) 
            count_white  = avg_list.count(255)       
            if count_white>0.8*count_black:
                tag.append(1)
            else:
                tag.append(0)       
    tag[2],tag[3] = tag[3],tag[2]
    return(tag)


# # Calculating Projection matrix
# 
# 

# In[12]:


#Function to take in the  homogeneous matrix and camera pareameters and returns the 
#Projection Matrix
def projectionMatrix(h, K):  
    h1 = h[:,0]          #taking column vectors h1,h2 and h3
    h2 = h[:,1]
    h3 = h[:,2]
    #calculating lamda
    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
    b_t = lamda * np.matmul(np.linalg.inv(K),h)

    #check if determinant is greater than 0 ie. has a positive determinant when object is in front of camera
    det = np.linalg.det(b_t)

    if det > 0:
        b = b_t
    else:                    #else make it positive
        b = -1 * b_t  
        
    row1 = b[:, 0]
    row2 = b[:, 1]                      #extract rotation and translation vectors
    row3 = np.cross(row1, row2)
    
    t = b[:, 2]
    Rt = np.column_stack((row1, row2, row3, t))
#     r = np.column_stack((row1, row2, row3))
    P = np.matmul(K,Rt)  
    return(P,Rt,t)


# # Warping Function

# In[13]:


#Function for warping, equivalent to cv2.warpPerspective
def warpPerspective(H,img,maxHeight,maxWidth):
    H_inv=np.linalg.inv(H)
    warped=np.zeros((maxHeight,maxWidth,3),np.uint8)
    for a in range(maxHeight):
        for b in range(maxWidth):
            f = [a,b,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            warped[a][b] = img[int(y/z)][int(x/z)]
    return(warped)


# # Rotate Image

# In[14]:

#function to rotate the image by 90
def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# #  - - - - Tag Detection Video - - - -

# In[20]:


print('Video Running to detect Tag ID')
print('note: might take a while ..... (approx : 15 seconds)')
#reading a video file
cap = cv2.VideoCapture('Tag1.mp4')
count=0
bottom_is_square_count=0
if (cap.isOpened() == False):
    print('Please check the file name again!')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
cam_out = cv2.VideoWriter('MULTIPLE TAG_ID.mp4',0x7634706d, 10.0, (1280,720))
while(cap.isOpened()):
    ret,frame = cap.read()
    ret,frame1 = cap.read()
    if ret == True:
          #convert to Grayscale
        img2 =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #Threshold the frame
        _,threshold = cv2.threshold(img2, 240, 250, 
                                    cv2.THRESH_BINARY)
        #Detect the contours
        contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #Select the appropriate contours by giving the right hierarchy and choosing the right contour from the list
        try: 
            hierarchy = hierarchy[0][2]
        except: 
            hierarchy = []
        min_x, min_y = 200,200
        max_x = max_y = 0      
        (x,y,w,h) = cv2.boundingRect(contours[1])#Select required contours from list of list.
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 80 and h > 80:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,25), 2)
            cv2.putText(frame,'('+str(x)+','+str(y)+')',(x,y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(255,25,25),2)
            pts2 = np.float32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])#these are the points where
            cv2.imshow('Frame Original',frame)
            frame_new = frame1[y:y+h,x:x+w]
            contour_img,out = drawARContour(frame_new)
            tl,tr,br,bl,maxWidth,maxHeight,d= determinePoints(out) #determine the end points of contour
            H_2 = findHomography(np.array([tl,tr,br,bl],np.float32), d) #find homography between the src and dst points
            warped_new = warpPerspective(H_2,contour_img,maxWidth, maxHeight) #perform forward warping on the detected contour
            frame_2 = cv2.resize(frame,(1280,720))
            cam_out.write(frame_2)
            eight_grid_threshed = thresholdAndDraw(warped_new)
            eight_grid_threshed = cv2.cvtColor(eight_grid_threshed,cv2.COLOR_BGR2GRAY)
            target = frame[y:y+h,x:x+w]#points of the contours
            cv2.imshow('target',target)

            new_target = cv2.resize(target, (200, 200)) #resizing image to perform encoding
            cv2.imshow('new_target > resized to 200 by 200',new_target)
#function to check the orientation of the tag. Take the frame which has the most black pixels  of the 4 bottommost right squares.
#Check the 4x4 padded sqaures if they are white. Rotate the image if this condition is not met until it satisfies
#Then detect the inner 2x2 grid pixels for value of tag
            new_threshold = cv2.cvtColor(new_target,cv2.COLOR_BGR2GRAY)
            new_threshold = thresholdAndDraw(new_threshold)      #function to check the orientation of the tag. Take the 
            cv2.imshow('new_threshold',new_threshold)
            tagstraightORnot,val = bottomIsBlackSquare(new_threshold)
            if bottom_is_square_count==0 and val==1:
                print('Found a match to calculate .... ')
                cv2.imshow('tagstraightORnot',tagstraightORnot)
                if bottomIsWhiteSquare(tagstraightORnot)==False:
                    print('Rotate')
                    rotated = rotateImage(tagstraightORnot,90)
                    cv2.imshow('ROTATED',rotated)
                    if bottomIsWhiteSquare(rotated)==False:
                        print('Rotate')
                        rotated2 = rotateImage(rotated,90)
                        cv2.imshow('ROTATED',rotated2)
                        if bottomIsWhiteSquare(rotated2)==False:
                            print('Rotate')
                            rotated3 = rotateImage(rotated2,90)
                            cv2.imshow('ROTATED',rotated3)
                            tag = Tag_ID(rotated3)
                            print('Tag ID is > ',tag)
#                            
                        else:
                            tag = Tag_ID(rotated2)
                            print('Tag ID is > ',tag)
                        
                    else:
                        tag = Tag_ID(rotated)
                        print('Tag ID is > ',tag)
                       
                else:
                    tag = Tag_ID(tagstraightORnot)
                    print('Tag ID is > ',tag)
               
                
                bottom_is_square_count+=1
    if cv2.waitKey(1)== 27:
        break
cap.release()
cam_out.release()
cv2.destroyAllWindows()




