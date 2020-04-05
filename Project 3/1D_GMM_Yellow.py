import cv2
import numpy as np
from imutils import contours


def gaussian_pdf(data, mean, covar):
    data_mean = np.matrix(data - mean)
    covar_inv = np.linalg.pinv(covar)
    pdf = (2.0 * np.pi) ** (-len(data[1]) / 2.0) * (1.0 / np.linalg.det(covar) ** 0.5) * \
          np.exp(-0.5 * np.sum(np.multiply(data_mean * covar_inv, data_mean), axis=1))
    return pdf


# In[6]:
updated_mean = np.load('yellow_means.npy')
updated_covar = np.load('yellow_covar.npy')
updated_weights = np.load('yellow_weights.npy')
K = 2
rad_yellow = 5
thresh_yellow = 1.2
#calculate the optimal cluster paramters
# updated_mean,updated_covar,updated_weights = train_GMM(data,K,yellow_mean,covar,weights)
print("updated parameters=",updated_mean,updated_covar,updated_weights)


# In[7]:


#read the video
cap = cv2.VideoCapture("detectbuoy.avi")
images = []
prev_cnts = 0
while (cap.isOpened()):
    #check if frame read is true
    ret, frame = cap.read()
    if ret == False:
        print("Exit!")
        cv2.destroyAllWindows()
        break    
    
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
    # cv2.imshow('Yellow masked', output)
    #blur = cv2.GaussianBlur(output,(5,5),0)
    #Do filtering and edge detection to detect the buoys
    blur = cv2.medianBlur(output,5)
    edged = cv2.Canny(blur,20,255 )
    # cv2.imshow("Edge detection- Canny",edged)
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
        cv2.imshow("Yellow detected",reshaped)
        images.append(reshaped)
    else:
        reshaped = cv2.resize(curr_image,(640,480),interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Yellow detected",reshaped)
        images.append(reshaped)
   
    if cv2.waitKey(10) == 27:
        cv2.destroyAllWindows()# exit if Escape is hit
        break

#Save the video file in .avi
out = cv2.VideoWriter('Yellow_1D.avi', cv2.VideoWriter_fourcc(*'XVID'), 5.0, (640, 480))
for image in images:
    out.write(image)
    cv2.waitKey(10)


out.release()
    
cap.release()
