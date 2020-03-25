# NOTE: This file is just to be run once

# Motive :

# To run the video and then takes every third frame into the testing frame,
# thus splitting between Training and Testing in a desired 70-30 percent ratio
import cv2

# To learn the total number of frames in the video


cap = cv2.VideoCapture('detectbuoy.avi')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( 'Total frames in the video : ', length )

# Now splitting the video to testing and training data

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if count % 3 == 0:
            cv2.imwrite('Testing_Frames//frame{:d}.png'.format(count), frame)
        else:
            cv2.imwrite('Training_Frames//frame{:d}.png'.format(count), frame)
        count += 1  # i.e. at 7 fps, this advances one second
        cap.set(1, count)
    else:
        cap.release()
        break
