########################### CONTOUR PROPERTIES ###########################################

import cv2
import numpy as np 

cap = cv2.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    _ , thresh = cv2.threshold(frame,127,255,cv2.THRESH_BINARY) 
    _ , thresh2 = cv2.threshold(frame,200,255,cv2.THRESH_BINARY)   
    th2 = cv2.adaptiveThreshold(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

    cv2.imshow('frame',frame)
    cv2.imshow('tresh', thresh)
    cv2.imshow("bilateralFilter",bilateralFilter)
    #cv2.imshow('tresh2', thresh2)
    cv2.imshow('adaptive thresh', th2)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF  ### 5ms wait time 
    if k == 27:
        break
cv2.destroyAllWindows()

################# CANNY EDGE DETECTION #################################

## Multiple stage algorithm to detect edges:
# Stage 1: Noise reduction.... removes noise with 5x5 Gaussian
# Stage 2: Finds Intensity Gradient of Image  ....  uses derivative to get gradient and direction
# Stage 3: Non-Maximum Supression.... only leaves local maximums to get left with thin lines
# Stage 4: Hysteresis Thresholding...  Tells the algorithm the intensity value at which a pixel is an edge
#              Also, if a value is between maxVal and minVal threshold values, it's part of edge if it's 
#                     attached to a 'sure edge' point... making if part of the edge
#Ex:
edges = cv2.Canny(img,100,200, L2gradient = False) # Where 100 = minValue   and 200 = maxValue for Threshold 
# Note: Set L2gradient = True for higher accuracy



############## CONTOURS ##############################

## PRE REQUISITE FOR FINDING CONTOURS
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

###### DRAW CONTOURS  #####

img = cv2.drawContours(img, contours, -1, (0,255,0), 3)    # draws green contours for all contours found
img = cv2.drawContours(img, contours, 3, (0,255,0), 3)     # draws green contours for contour #4
cnt = contours[4]
img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)        # draws green contours for contour #4
# Pass CHAIN_APPROX_NONE to store all points in the contours found, rather than jsut some boundary points
# USING CHAIN_APPROX_SIMPLE IS MUCH MORE MEMORY EFFICIENT AND FASTER!!!!!!!!

# Contour Area
area = cv2.contourArea(cnt)
#Contour perimeter
perimeter = cv2.arcLength(cnt,True)    # Second arg specifies if contour is closed or not... True = closed

## Approvimation
# uses epsilon value to 'round' a contour shape to another (less noisy shapy)  ... see docs
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

# Draw bounding rectangle
x,y,w,h = cv2.boundingRect(cnt)     
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# Draw bounding rectangle for tilted items ( MORE ACCURATE FOR TILTED ITEMS)
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
im = cv2.drawContours(im,[box],0,(0,0,255),2)


##################################### CONTOUR PROPERTIES + FUNCTIONS  ##########################

# aspect_ratio = width / height                ## definition of aspect ratio
x,y,w,h = cv2.boundingRect(cnt)
aspect_ratio = float(w)/h

# extent = object_area / bounding_rect_area    ## definition of extent
area = cv2.contourArea(cnt)
x,y,w,h = cv2.boundingRect(cnt)
rect_area = w*h
extent = float(area)/rect_area

# Solidity = contour_area / convex_hull_area
area = cv2.contourArea(cnt)
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area

# Equivalent_Diameter : diam of circle whos area is same as contour area
area = cv2.contourArea(cnt)
equi_diameter = np.sqrt(4*area/np.pi)

# Orientation: angle at which object is directed... gives major and minor axis lengths
(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)

# Mask and Pixel points:     gives all points which comprise the object
mask = np.zeros(imgray.shape,np.uint8)
cv2.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))  # equivalent to ... pixelpoints = cv2.findNonZero(mask)

# Gets max and min values of contour
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgray,mask = mask)

# Gets mean color or mean intensity (for greyscale) of object
mean_val = cv2.mean(im,mask = mask)

# Gets extreme points of object
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

# to find distance from contour line to object follow:
#         https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html

############ MATCH SHAPES
# this compares two shapes or two contnours and returns metric 
#          value of 0.00 = identical match, value of 1.0 = completely different
ret, thresh = cv2.threshold(img1, 127, 255,0)
ret, thresh2 = cv2.threshold(img2, 127, 255,0)
contours,hierarchy = cv2.findContours(thresh,2,1)
cnt1 = contours[0]
contours,hierarchy = cv2.findContours(thresh2,2,1)
cnt2 = contours[0]

ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
print ret



######## CONTOURS HIERARCHY  #########################

# In the line below, we identify what the hierarchy parameter actually does !!!!!!!!!!!!!!!!!!!!!
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#  BASICALLY, if we find a bunch of contours in an image , any contours NESTED inside other contours,
#      are considered to be child contours and obtain the child hierarchy level...
#               SO it's used to find nested contours !!!!!!! Hella useful

# Hierarchy returns : [Next, Previous, First_Child, Parent]
# NExt:        shows next contour on same level, -1 if none
# Previous:    shows previous contour on same level, -1 if none
# First_child: shows first child of the contour   , -1 if none
# Parent:      shows its parent       , -1 if none
##### see docs for better example

# Explaining the RETR_TREE and other RETR_FOO Flags

#### RETR_LIST
# Returns all contours with no child-parent relation.... aka NO NESTING

#### RETR_EXTERNAL
# Only returns outter contours

#### RETR_CCOMP
#  arranges contours in a wierd ass hierarchy... see docs

#### RETR_TREE
# Organizes it pretty nicely into a proper hierarchy tree... see docs for better example
