import cv2
import os
count = 0
for filename in os.listdir("green_train"):
    image = cv2.imread(os.path.join("green_train",filename))
    resized = cv2.resize(image,(40,40),interpolation=cv2.INTER_LINEAR)
    image = resized[13:27,13:27]
    filename_green = "C://Users//govin//OneDrive - University of Maryland//UMD DOCS//Semester 2//ENPM673//ENPM673 Project 3//buoy_Green//Train//file_green_%d.png" % count
    cv2.imwrite(filename_green, image)
    count+=1