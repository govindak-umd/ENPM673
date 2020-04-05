
Project 3 Submission - ENPM673 - Perception for Autonomous Robotics

Submission date : 
16 April - 2020
-------------------------------------------------------------
There are  10 codes attached without any sub-directories. 
------------------------------------------------------------------------------------------------------------------------------------
The codes in .py files are as follows:

## Codes are :
 
roipoly.py - > does the cropping using points given by the user
Training_Images.py - > Python script to train the 
Frame_Splitting.py - > Splits the frames for testing and training
green_crop.py - > just to crop the green image even further
Colour_Hist_Determine.py - > to get the RGB histogram as well as R,G and B histograms seperately
EM_1d_Gaussian.py - > Code for sample expectation maximization

NOTE: Please make sure the path is given properly for the below images,
NOTE: Also, in that case, make sure the data set is present as well
1D_GMM_Green.py - > 1D gaussian for the green buoy
1D_GMM_Orange.py - > 1D gaussian for the Orange buoy
1D_GMM_Yellow.py- > 1D gaussian for the Yellow buoy
combined_buoys.py -> Combined output for all the buoys together

NOTE : Running the last file should give all the outputs in one image itself

------------------------------------------------------------------------------------------------------------------------------------
## The Youtube links to the videos are as follows:

## Combined Video >>>>

https://youtu.be/k08j0B4qnFw

## Individual Videos >>>>>

https://youtu.be/RguKZUCVQdY
https://youtu.be/cgsLLHPmiag
https://youtu.be/19244eKtAhs
------------------------------------------------------------------------------------------------------------------------------------
## Github repo  : 

https://github.com/govindak-umd/ENPM673/tree/master/Project%203
------------------------------------------------------------------------------------------------------------------------------------
## All the media files have been linked above. Few images have been attached as well.
-----------------------------------------------------------------------------------------------------------------------------------
## Group members

Ashwin Prabhakar
Govind Ajith Kumar
Rajeshwar N S 
-----------------------------------------------------------------------------------------------------------------------------------
## Programme used:
Python 3.7
OpenCV 4.xx
------------------------------------------------------------------------------------------------------------------------------------
## Libraries used:

import cv2
import numpy as np
from scipy.stats import multivariate_normal
from imutils import contours
from matplotlib import pyplot as plt
import glob
import imageio
from roipoly import roipoly
import os

-----------------------------------------------------------------------------------------------------------------------------------

____________________________________________________________________________________________________________________________________
