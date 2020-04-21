		Project 4 Submission - ENPM673 - Perception for Autonomous Robotics
		
TEAM
------
    Ashwin Prabhakar
    Govind Ajith Kumar
    Rajeshwar N S 

Submission date : 
------
    20th April - 2020
-------------------------------------------------------------
						There are  6 codes attached without any sub-directories. 
The codes in .py files are as follows:

    lucas-kanade_BABY.py (lucas kanade algorithm for detection)
    rectangle_coord_Baby.py (to draw the rectangular coordinates around the baby)

    lucas-kanade_CAR.py (lucas kanade algorithm for car detection WITHOUT accounting for illumination change)
    lucas-kanade_CAR_brightened.py (lucas kanade algorithm for car detection AFTER accounting for illumination change)
    rectangle_coord_Car.py (to draw the rectangular coordinates around the car

    lucas-kanade_BOLT.py (lucas kanade algorithm for detection)
    rectangle_coord_Bolt.py (to draw the rectangular coordinates around the bolt)

NOTE: Ensure that 'APART FROM THE DATASET', the following folders are in the same directory as the codes as well 
------
Directories are:

    all_new_imgs_BABY

    all_new_imgs_CAR

    all_new_imgs_BOLT

These can be empty folders that will get populated when the code is run.

YOUTUBE VIDEOS
------


    https://youtu.be/4Lh4ISospVQ
    https://youtu.be/i4TaXdF2sTY
    https://youtu.be/o-suy8SslAo
    https://youtu.be/Zbeu6yfYEiQ
   
GITHUB
------


    https://github.com/govindak-umd/ENPM673/tree/master/Project%204

Software used:
------

    Python 3.7
    OpenCV 4.xx

Libraries
------

    import cv2
    import numpy as np
    import glob
    from scipy.ndimage import affine_transform
    import math
-----------------------------------------------------------------------------------------------------------------------------------

____________________________________________________________________________________________________________________________________
