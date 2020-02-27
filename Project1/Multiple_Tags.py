#!/usr/bin/env python
# coding: utf-8

# # PROJECT 1 | Perception for Autonomous Robotics | ENPM673

# # Importing Libraries

# In[37]:


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

# In[38]:


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

# In[39]:


def distanceCalc(a,b,c,d): #Calculation of the distance bewteen two points
    dist = np.sqrt(((a-b) ** 2) + ((c-d) ** 2)) 
    return (dist) 


# # Function to compute AR TAG CONTOUR

# In[40]:


#Function to make a list of tuple points into clockwise order
def toClockwise(tup):
    coords =tup
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    t = (sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::1]))) % 360))
    return (t)


# In[41]:


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

# In[42]:


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

# In[43]:


#Fuction to threshold the 8 grid image and returns it
def thresholdAndDraw(img):
    ret,threshed = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
    resized_image = cv2.resize(threshed, (200, 200))
    eight_thresh = draw8Grid(resized_image)
    return(eight_thresh)


# # Function to initiate the warp and perspective transform

# In[44]:


#Function to determine the points for warping and perspective transformation
def determinePoints(out):
    tl = out[0]
    tr = out[1]
    br = out[2]
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

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    return (tl,tr,br,bl,maxWidth,maxHeight,dst)


# # Detecting TAG's straightness

# In[45]:


#Function to check for Tag Straightness 
def detectARTagStraightness(img):
    bottom_square= img[150:200,150:200]
    list_black_or_white = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
                list_black_or_white.append(img[i][j])
    count_black  = list_black_or_white.count(0)
    count_white  = list_black_or_white.count(255)
    if count_black > count_white:
        #NOW CHECKING IF THE ORIENTATION IS PROPER
        img = img [50:150,50:150] #GEETING RID OF THE BLACK PADDING
        bottom_square= img[50:100,50:100] #THE ROI FOR THE WHITE ORIENTATION CHECK
        new_list_black_or_white = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                    new_list_black_or_white.append(img[i][j])
        count_black  = new_list_black_or_white.count(0) 
        count_white  = new_list_black_or_white.count(255)
        if count_white>(0.50*len(new_list_black_or_white)):
            return(True)


# # Function for TAG ID

# In[46]:


#Function to determine TAG ID
def Tag_ID(img):
    tag = [] #empty tag
    center_img = img[75:125,75:125]
    for i in range(2):
        for j in range(2):
            small_sq = center_img[(25*i):(25*(i+1)),(25*j):(25*(j+1))]
            avg_list = []
            for c in range(25):
                for r in range(25):
                    avg_list.append(small_sq[c][r])
            #mode to check which pixel is largely present here
            data = Counter(avg_list) 
            get_mode = dict(data) 
            mode = [k for k, v in get_mode.items() if v == max(list(data.values()))]        
            if mode[0] == 255:
                tag.append(1)
            else:
                tag.append(0)       
    tag[2],tag[3] = tag[3],tag[2]
    print('the tag ID is: ', tag)
    return(tag)


# # Calculating Projection matrix
# 
# 

# In[47]:


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

# In[48]:


#Function for warping, equivalent to cv2.warpPerspective
def warpPerspective(H,img,maxHeight,maxWidth):
    H_inv=np.linalg.inv(H)
    im_out=np.zeros((maxHeight,maxWidth,3),np.uint8)
    for a in range(maxHeight):
        for b in range(maxWidth):
            f = [a,b,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            if (int(y/z) < 1080 and int(y/z) > 0) and (int(x/z) < 1920 and int(x/z) > 0):
                im_out[a][b] = contour_img[int(y/z)][int(x/z)]
    return(im_out)


# #  - - - - Tag Detection Video - - - -

# In[50]:


print('Video Running to detect Tag ID')
print('note: might take a while ..... (approx : 15 seconds)')
#reading a video file
cap = cv2.VideoCapture('multipleTags.mp4')
count=0
if (cap.isOpened() == False):
    print('Please check the file name again!')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
cam_out = cv2.VideoWriter('Multiple AR Tags Video.mp4',0x7634706d, 10.0, (1280,720))
while(cap.isOpened()):
    img_1 = cv2.imread('Lena.png',1)
    ret,frame = cap.read()
    ret,frame1 = cap.read()
    if ret == True:
        img2 =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        _,threshold = cv2.threshold(img2, 240, 250, 
                                    cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        try: 
            hierarchy = hierarchy[0]
        except: 
            hierarchy = []
        min_x, min_y = 200,200
        max_x = max_y = 0
        for contour, hier in zip(contours, hierarchy):
            (x,y,w,h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x+w, max_x)
            min_y, max_y = min(y, min_y), max(y+h, max_y)
            if w > 80 and h > 80:
                rows , cols ,ch = frame.shape
                back_rows , back_cols ,back_ch = img_1.shape
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,25), 2)
                cv2.putText(frame,'('+str(x)+','+str(y)+')',(x,y),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(255,25,25),2)
                pts2 = np.float32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])#these are the points where
                cv2.imshow('Frame Original',frame)
                try:
                    frame_new = frame1[y:y+h,x:x+w]
                    try:
                        contour_img,out = drawARContour(frame_new)
                        tl,tr,br,bl,maxWidth,maxHeight,d= determinePoints(out)
                        H_2 = findHomography(np.array([tl,tr,br,bl],np.float32), d)
                        warped_new = warpPerspective(H_2,contour_img,maxWidth, maxHeight)
                        cv2.imshow('Warped',warped_new)
                        frame_2 = cv2.resize(frame,(1280,720))
                        cam_out.write(frame_2)
                        eight_grid_threshed = thresholdAndDraw(warped_new)
                        eight_grid_threshed = cv2.cvtColor(eight_grid_threshed,cv2.COLOR_BGR2GRAY)
                        cv2.imshow('eight_grid_threshed',eight_grid_threshed)
                        g = detectARTagStraightness(eight_grid_threshed)
                        if g == True:
                            tagid_returned = Tag_ID(frame([],[]))  
                            cv2.putText(frame,'TAG ID: '+tagid_returned,(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,4,(0,255,0),3)

                    except:
                        pass
                except:
                    pass
    if cv2.waitKey(1)== 27:
        break
cap.release()
cam_out.release()
cv2.destroyAllWindows()


# # - - - -LENA VIDEO- - - -

# In[21]:


print('Showing Lena on the April Tag')
#reading a video file
cap = cv2.VideoCapture('multipleTags.mp4')
count=0
img_1 = cv2.imread('Lena.png',1)
if (cap.isOpened() == False):
    print('Please check the file name again!')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
cam_out = cv2.VideoWriter('Lena_Video.mp4',0x7634706d, 5.0, (1280,720))
while(cap.isOpened()):
    img_1 = cv2.imread('Lena.png',1)
    ret,frame = cap.read()
    ret,frame1 = cap.read()
    if ret == True:
        img2 =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        _,threshold = cv2.threshold(img2, 240, 250, 
                                    cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        try: 
            hierarchy = hierarchy[0]
        except: 
            hierarchy = []
        min_x, min_y = 200,200
        max_x = max_y = 0
        for contour, hier in zip(contours, hierarchy):
            (x,y,w,h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x+w, max_x)
            min_y, max_y = min(y, min_y), max(y+h, max_y)
            if w > 80 and h > 80:
                rows , cols ,ch = frame.shape
                back_rows , back_cols ,back_ch = img_1.shape
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,25), 2)
                cv2.putText(frame,'('+str(x)+','+str(y)+')',(x,y),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(255,25,25),2)
                pts2 = np.float32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])#these are the points where
                try:
                    frame_new = frame1[y:y+h,x:x+w]
                    try:
                        k,out = drawARContour(frame_new)
                        tl,tr,br,bl,maxWidth,maxHeight,d= determinePoints(out)
                        H_2 = findHomography(np.array([tl,tr,br,bl],np.float32), d)
                        warped = warpPerspective(H_2,k, maxWidth, maxHeight)  
                        count+=1
                        if count%2!=0:
                            cv2.imshow('Warped',warped)
                            img_1 = cv2.resize(img_1, (warped.shape[0],warped.shape[1]))
                            rows , cols ,ch = img_1.shape
                            pts2 = np.array([[0,0],[rows,0],[rows,cols],[0,cols]])
                            pts1 = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
                            h_new = findHomography(pts1,pts2)
                            h_inv = np.linalg.inv(h_new)  
                            for j in range(0,warped.shape[1]):
                                for k in range(0,warped.shape[0]):
                                    x_lena, y_lena, z_lena = np.dot(h_inv,[j,k,1])
                                    if (int(y_lena/z_lena) < np.shape(threshold)[0] and int(y_lena/z_lena) > 0) and (int(x_lena/z_lena) < np.shape(threshold)[1] and int(x_lena/z_lena) > 0):
                                        frame[int(y_lena/z_lena)][int(x_lena/z_lena)] = img_1[j][k]
                            cv2.imshow('Frame Original',frame)
                            frame_2 = cv2.resize(frame,(1280,720))
                            cam_out.write(frame_2)
                        else:
                            pass
                    except:
                        pass
                except:
                    pass
    if cv2.waitKey(1)== 27:
        break
cap.release()
cam_out.release()
cv2.destroyAllWindows()


# # CUBE VIDEO

# In[15]:


#Camera Intrinsic Paramters
K =np.array([[1406.08415449821,0,0],
   [ 2.20679787308599, 1417.99930662800,0],
   [ 1014.13643417416, 566.347754321696,1]])
K = K.T


# In[17]:


##Cube
#reading a video file
cap = cv2.VideoCapture('multipleTags.mp4')
count=0
img_1 = cv2.imread('Lena.png',1)
if (cap.isOpened() == False):
    print('Please check the file name again!')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
cam_out = cv2.VideoWriter('cube_video.mp4',0x7634706d, 10.0, (1280,720))
while(cap.isOpened()):
    img_1 = cv2.imread('Lena.png',1)
    ret,frame = cap.read()
    ret,frame1 = cap.read()
  
    if ret == True:
        img2 =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        _,threshold = cv2.threshold(img2, 240, 250, 
                                    cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        try: 
            hierarchy = hierarchy[0][2]
        except: 
            hierarchy = []
        min_x, min_y = 200,200
        max_x = max_y = 0 
        (x,y,w,h) = cv2.boundingRect(contours[1])
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 80 and h > 80:
            try:
                frame_new = frame1[y:y+h,x:x+w]
                try:
                    k,out = drawARContour(frame_new)
                    tl,tr,br,bl,maxWidth,maxHeight,d= determinePoints(out)
                    H = findHomography(np.array([tl,tr,br,bl],np.float32), d)
                    pts4 = np.array([[0,0],[199,0],[199,199],[0,199]])
                    pts3 = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
                    H_cube = findHomography(pts4,pts3)
                    P,Rt,t = projectionMatrix(H_cube,K)
#                     P,Rt,r,t = projectionMatrix(H_cube,K)
                    x1,y1,z1 = np.matmul(P,[0,0,0,1])
                    x2,y2,z2 = np.matmul(P,[0,199,0,1])
                    x3,y3,z3 = np.matmul(P,[199,0,0,1])
                    x4,y4,z4 = np.matmul(P,[199,199,0,1])
                    x5,y5,z5 = np.matmul(P,[0,0,-199,1])
                    x6,y6,z6 = np.matmul(P,[0,199,-199,1])
                    x7,y7,z7 = np.matmul(P,[199,0,-199,1])
                    x8,y8,z8 = np.matmul(P,[199,199,-199,1])
                    
                    cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (255,0,0), 2)
                    cv2.line(frame,(int(x2/z2),int(y2/z2)),(int(x6/z6),int(y6/z6)), (255,0,0), 2)
                    cv2.line(frame,(int(x3/z3),int(y3/z3)),(int(x7/z7),int(y7/z7)), (255,0,0), 2)
                    cv2.line(frame,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (255,0,0), 2)

                    cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (0,255,0), 2)
                    cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x3/z3),int(y3/z3)), (0,255,0), 2)
                    cv2.line(frame,(int(x2/z2),int(y2/z2)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)
                    cv2.line(frame,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)

                    cv2.line(frame,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (0,0,255), 2)
                    cv2.line(frame,(int(x5/z5),int(y5/z5)),(int(x7/z7),int(y7/z7)), (0,0,255), 2)
                    cv2.line(frame,(int(x6/z6),int(y6/z6)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)
                    cv2.line(frame,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)
                    cv2.imshow("DISPLAY", frame)
                    frame_2 = cv2.resize(frame,(1280,720))
                    cam_out.write(frame_2)
                    eight_grid_threshed = thresholdAndDraw(warped)
                    eight_grid_threshed = cv2.cvtColor(eight_grid_threshed,cv2.COLOR_BGR2GRAY)
                    g = detectARTagStraightness(eight_grid_threshed)
                    if g == True:
                        tagid_returned = Tag_ID(eight_grid_threshed)  
                        cv2.putText(frame,'TAG ID: '+tagid_returned,(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,4,(0,255,0),3)
                except:
                    pass
            except:
                pass
    if cv2.waitKey(1)== 27:
        break
cap.release()
cam_out.release()
cv2.destroyAllWindows()


# In[ ]:




