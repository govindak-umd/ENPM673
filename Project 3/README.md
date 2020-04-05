								Project 3 Submission - ENPM673 - Perception for Autonomous Robotics
--------------------------------------------------------TEAM----------------------------------------------------------------------
## Group members :

Ashwin Prabhakar
Govind Ajith Kumar
Rajeshwar N S 

------------------
Submission date : 
6th April - 2020
-------------------------------------------------------------
						There are  13 codes attached without any sub-directories. 
---------------------------------------------- CODES------------------------------------------------------------------
The codes in .py files are as follows:

## Codes are :

NOTE: CODE BELOW WILL ONLY WORK IF THERE ARE A BUNCH OF IMAGES IN THE FOLDER TO CROP
roipoly.py - > does the cropping using points given by the user

NOTE: CODE BELOW WILL ONLY WORK IF THERE ARE A BUNCH OF IMAGES IN THE FOLDER
Training_Images.py - > Python script to train the images 

NOTE: CODE BELOW WILL ONLY WORK IF THERE IS A VIDEO IN THE LINK
Frame_Splitting.py - > Splits the frames for testing and training

NOTE: CODE BELOW WILL ONLY WORK IF THERE ARE A BUNCH OF IMAGES IN THE FOLDER TO CROP
green_crop.py - > just to crop the green image even further

NOTE: CODE BELOW WILL ONLY WORK IF THERE IS A FOLDER AS SHOWN IN THE SCRIPT WITH IMAGES IN IT
Colour_Hist_Determine.py - > to get the RGB histogram as well as R,G and B histograms seperately

EM_1d_Gaussian.py - > Code for sample expectation maximization

NOTE: THE THREE CODES BELOW WILL REQUIRE THE TRAINING IMAGES IN THE FOLDERS AS MENTIONED IN THE CODE. THEY HAVE ALREADY BEEN RUN,
AND THEY OUPUT THE .npy files. You can find the .NPY Files in the submissions.
yellow_dataset_training.py
green_dataset_training.py
orange_dataset_training.py

NOTE: Please make sure the path is given properly for the below images,
NOTE: Also, in that case, make sure the data set is present as well
1D_GMM_Green.py - > 1D gaussian for the green buoy
1D_GMM_Orange.py - > 1D gaussian for the Orange buoy
1D_GMM_Yellow.py- > 1D gaussian for the Yellow buoy

NOTE : Running the next file should give all the outputs in one image itself
combined_buoys.py -> Combined output for all the buoys together



------------------------------------------------------NOTE REGARDING THE .npy FILES----------------------------------------

The .npy files have been attached here as well.

They contain the means weights and covariances for green orange and yellow buoys, respectively.

-------------------------------------------------------VIDEOS---------------------------------------------------------------
## The Youtube links to the videos are as follows:

## Combined Video >>>>

https://youtu.be/k08j0B4qnFw

## Individual Videos >>>>>

https://youtu.be/RguKZUCVQdY
https://youtu.be/cgsLLHPmiag
https://youtu.be/19244eKtAhs
--------------------------------------------------------GITHUB---------------------------------------------------------------------
## Github repo  : 

https://github.com/govindak-umd/ENPM673/tree/master/Project%203
------------------------------------------------------------------------------------------------------------------------------------
## All the VIDEO files have been linked above. Few images have been attached in submission and report as well.

------------------------------------------------------PROGRAMME-----------------------------------------------------------------------
## Programme used:
Python 3.7
OpenCV 4.xx
-------------------------------------------------LIBRARIES IMPORTED-------------------------------------------------------------------
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
