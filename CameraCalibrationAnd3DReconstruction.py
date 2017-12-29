################################# Camera Calibration and 3D Reconstruction  ###################################

import numpy as np
import cv2

########################## Camera Calibration #################################

""" dont care too much about this:   plus it's hella long and mathy
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
"""



############################ POSE ESTIMATION #####################################

### basically tells you how an object is situated in space
## this is a build up on the previous tutorial on Camera Calibration so Read that ! 
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html



############################ EPIPOLAR GEOMETRY #################################

## this is baasically disparity : needs 2 cameras which im not too interested in rn
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html



########################## Depth Map from StereoImages ########################

# this is basically disparity as well..... neeed 2 cameras, not too interested and it didnt work before so well lol
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html