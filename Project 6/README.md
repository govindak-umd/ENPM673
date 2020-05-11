		Project 6 Submission - ENPM673 - Perception for Autonomous Robotics

Problem  and Dataset:
---------
    https://www.kaggle.com/c/dogs-vs-cats/data
    
A lot of prior installations have to be done before running the notebook.

    Install Tensorflow - GPU - Version 2.1.0
    If run on an NVidia Graphics card Laptop, install Cuda 10.2
    Install corresponding cuDNN version as well.
Follow this with the installation of the following libraries:

The libraries imported include: 

    import numpy as np 
    import pandas as pd 
    import sys
    from keras.preprocessing.image import ImageDataGenerator, load_img
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import random
    from IPython.display import display 
    from PIL import Image
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import os
    print(os.listdir("dogs-vs-cats"))
    import PIL
    from tensorflow.keras.callbacks import ModelCheckpoint
    import seaborn as sns


						There is just 1 notebook attached here


-------------------------------------------------------------
Directories: Make sure the data set is in the folder:

The directory structure looks like as follows :

    dogs-vs-cats
      test
        test
          <contains all the training images (25000 images)>
      train
        train
          <contains all the testing images (12500 images)>	
      
Software used:

    Python 3.7

-----------------------------------------------------------------------------------------------------------------------------------

____________________________________________________________________________________________________________________________________
