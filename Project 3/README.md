								Project 3 Submission - ENPM673 - Perception for Autonomous Robotics
--------------------------------------------------------TEAM----------------------------------------------------------------------
## Group members :

Ashwin Prabhakar<br />
Govind Ajith Kumar<br />
Rajeshwar N S <br />

------------------
Submission date : 
6th April - 2020<br />
-------------------------------------------------------------
						There are  13 codes attached without any sub-directories. 
---------------------------------------------- CODES------------------------------------------------------------------
The codes in .py files are as follows:<br />

## Codes are :

NOTE: CODE BELOW WILL ONLY WORK IF THERE ARE A BUNCH OF IMAGES IN THE FOLDER TO CROP<br />
roipoly.py - > does the cropping using points given by the user<br />

NOTE: CODE BELOW WILL ONLY WORK IF THERE ARE A BUNCH OF IMAGES IN THE FOLDER<br />
Training_Images.py - > Python script to train the images <br />

NOTE: CODE BELOW WILL ONLY WORK IF THERE IS A VIDEO IN THE LINK<br />
Frame_Splitting.py - > Splits the frames for testing and training<br />

NOTE: CODE BELOW WILL ONLY WORK IF THERE ARE A BUNCH OF IMAGES IN THE FOLDER TO CROP<br />
green_crop.py - > just to crop the green image even further<br />

NOTE: CODE BELOW WILL ONLY WORK IF THERE IS A FOLDER AS SHOWN IN THE SCRIPT WITH IMAGES IN IT<br />
Colour_Hist_Determine.py - > to get the RGB histogram as well as R,G and B histograms seperately<br />

EM_1d_Gaussian.py - > Code for sample expectation maximization<br />

NOTE: THE THREE CODES BELOW WILL REQUIRE THE TRAINING IMAGES IN THE FOLDERS AS MENTIONED IN THE CODE. THEY HAVE ALREADY BEEN RUN,<br />
AND THEY OUPUT THE .npy files. You can find the .NPY Files in the submissions.<br />
yellow_dataset_training.py<br />
green_dataset_training.py<br />
orange_dataset_training.py<br />

NOTE: Please make sure the path is given properly for the below images,<br />
NOTE: Also, in that case, make sure the data set is present as well<br />
1D_GMM_Green.py - > 1D gaussian for the green buoy<br />
1D_GMM_Orange.py - > 1D gaussian for the Orange buoy<br />
1D_GMM_Yellow.py- > 1D gaussian for the Yellow buoy<br />

NOTE : Running the next file should give all the outputs in one image itself
combined_buoys.py -> Combined output for all the buoys together



------------------------------------------------------NOTE REGARDING THE .npy FILES----------------------------------------

The .npy files have been attached here as well.<br />

They contain the means weights and covariances for green orange and yellow buoys, respectively.<br />

-------------------------------------------------------VIDEOS---------------------------------------------------------------
## The Youtube links to the videos are as follows:<br />
<br />
## Combined Video >>>><br />

https://youtu.be/k08j0B4qnFw<br />
<br />
## Individual Videos >>>>><br />

https://youtu.be/RguKZUCVQdY<br />
https://youtu.be/cgsLLHPmiag<br />
https://youtu.be/19244eKtAhs<br />
--------------------------------------------------------GITHUB---------------------------------------------------------------------
## Github repo  : <br />

https://github.com/govindak-umd/ENPM673/tree/master/Project%203<br />
------------------------------------------------------------------------------------------------------------------------------------
## All the VIDEO files have been linked above. Few images have been attached in submission and report as well.<br />

------------------------------------------------------PROGRAMME-----------------------------------------------------------------------
## Programme used:<br />
Python 3.7<br />
OpenCV 4.xx<br />
-------------------------------------------------LIBRARIES IMPORTED-------------------------------------------------------------------
## Libraries used:<br />

import cv2<br />
import numpy as np<br />
from scipy.stats import multivariate_normal<br />
from imutils import contours<br />
from matplotlib import pyplot as plt<br />
import glob<br />
import imageio<br />
from roipoly import roipoly<br />
import os<br />

-----------------------------------------------------------------------------------------------------------------------------------

____________________________________________________________________________________________________________________________________
