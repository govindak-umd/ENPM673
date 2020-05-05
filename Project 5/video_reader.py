# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:34:12 2020

@author: nsraj
"""

import cv2


img_array = []
for i in range(0,3801,50):
    print (i)
    print(str(i)+'.png')
    img = cv2.imread('good_output/'+str(i)+'.png')
    img_array.append(img)
    
    
#%%
count = 0
out = cv2.VideoWriter('user_function_vid.avi', cv2.VideoWriter_fourcc(*'XVID'), 2, (640,480))
for image in img_array[:]:
    out.write(image)
    count+=1
    print(count)
    cv2.waitKey(10)
out.release()

