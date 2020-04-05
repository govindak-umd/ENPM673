
Project 3 Submission - ENPM673 - Perception for Autonomous Robotics<br/>

Submission date : <br/>
16 April - 2020
-------------------------------------------------------------
There are  10 codes attached without any sub-directories. <br/>
------------------------------------------------------------------------------------------------------------------------------------
The codes in .py files are as follows:
<br/>
## Codes are :
 <br/>
roipoly.py - > does the cropping using points given by the user<br/>
Training_Images.py - > Python script to train the <br/>
Frame_Splitting.py - > Splits the frames for testing and training<br/>
green_crop.py - > just to crop the green image even further<br/>
Colour_Hist_Determine.py - > to get the RGB histogram as well as R,G and B histograms seperately<br/>
EM_1d_Gaussian.py - > Code for sample expectation maximization<br/>
<br/>
NOTE: Please make sure the path is given properly for the below images,<br/>
NOTE: Also, in that case, make sure the data set is present as well<br/>
1D_GMM_Green.py - > 1D gaussian for the green buoy<br/>
1D_GMM_Orange.py - > 1D gaussian for the Orange buoy<br/>
1D_GMM_Yellow.py- > 1D gaussian for the Yellow buoy<br/>
combined_buoys.py -> Combined output for all the buoys together<br/>
<br/>
NOTE : Running the last file should give all the outputs in one image itself<br/>
<br/>
------------------------------------------------------------------------------------------------------------------------------------
## The Youtube links to the videos are as follows:

## Combined Video >>>><br/>
<br/>
https://youtu.be/k08j0B4qnFw<br/>
<br/>

## Individual Videos >>>>><br/>
<br/>
https://youtu.be/RguKZUCVQdY<br/>
https://youtu.be/cgsLLHPmiag<br/>
https://youtu.be/19244eKtAhs<br/>
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

import cv2<br/>
import numpy as np<br/>
from scipy.stats import multivariate_normal<br/>
from imutils import contours<br/>
from matplotlib import pyplot as plt<br/>
import glob<br/>
import imageio<br/>
from roipoly import roipoly<br/>
import os<br/>
<br/>
-----------------------------------------------------------------------------------------------------------------------------------

____________________________________________________________________________________________________________________________________
