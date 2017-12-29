########################################## Image Processing Training  ###################################
 
import cv2
import numpy as np 

######################Changing Colorspaces  #################################


# Color conversion: use cv2.cvtColor( Input_Image , flag) where flag = type of conversion

### See all the possible color conversion flags:###
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print (flags)

"""HSV: hue saturation value
hue range is 0, 179    
saturation range is 0, 255
value range is 0 , 255"""

## REALLLLYYYYYYYY COOL AND POWERFUL COLOR FILTERING EXAMPLE  (using webcam )
cap = cv2.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
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


################### HOW TO FIND HSV VALUES TO TRACK???????? #############
# great example:
green = np.uint8([[[0,255,0 ]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print (hsv_green)   # print out :::[[[ 60 255 255]]]
###### SO for example you can set lower value with : 60 - 10,  and upper value with 60 + 10 !!!!!! 


####### IMAGE THRESHOLDING ######################
### Params:   Greyscale Image, 
##         threshold value used to classify pixel values,
##         maxVal (value given is pixel value is more than threshold value)...usually 255,
##         threshold style (Thresh_Binary, Thresh_Tozero, etc..)
### Returns: retval and thresholded image
#Ex:
ret, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY) ## if 127 was set to 200, all dark pixels would be turned to black

######## ADAPTIVE THRESHOLD   ##############   used if different lighting is expected
## basically averages out / uses weighted sums to get better results when lighting is inconssitent
### Params: Greyscale Image, 
##         maxVal (value given is pixel value is more than threshold value)...usually 255,
##         Adaptive threshold type ( EITHER cv2.ADAPTIVE_THRESH_MEAN_C or cv2.ADAPTIVE_TRESH_GAUSSIAN_C)
##         threshold style (THRESH_BINARY, Thresh_Tozero, etc..)
##         Block Size (# of pixels that each block to be calculated will have ## neighbours of central pixel)
##         Constant C .. Basically, high C value = less detail / less background noise
### Returns : threshold image
#Ex: 									see docs as well
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
### NOTE THAT BIGGER BLOCK SIZES = LESS DETAILED   ..... 

############## OTSU'S BINARIZATION #################### uses the retVal output from normal threshold
### ONLY USED FOR BIMODAL IMAGES 
## AUtomatically calculates threshold value , note that threshold val = 0, and we +cv2.THRESH_OTSU
## Run normal thresh, and then call the OTSU which will optimize threshold value
#Ex:                  See docs for images
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#NOTE: Gaussian blurring the image before thresholding can give better results



######## GEOMETRIC TRANSFORMATIONS OF IMAGES   #################

## Resize:
height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC) # resizes the image by x2...see docs

## Translation:
rows,cols = img.shape
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img,M,(cols,rows))   #shifts object 100 pixels right, 50 pixels down,
##  params: 0, 1 in Matrix need to be kept to keep the matrix an identity matrix ish
##  params: cols, rows is the size of output image... so here, we are keeping it same size as original

## Rotation
rows,cols = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)    ## Rotates image 90 degrees...returns a Matrix
dst = cv2.warpAffine(img,M,(cols,rows))     ## WE MUST use this to output the rotated image via warpAffine()

## Affine Transformation
# Select 3 points, and then output image so that the perspective changes.... USEFUL FOR DISPARTY ALGORITHMS
#EX:
rows,cols,ch = img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])    # Sets 3 points for input 
pts2 = np.float32([[10,100],[200,50],[100,250]])  # Sets 3 points for output
M = cv2.getAffineTransform(pts1,pts2)             # Gets Affine Matrix 
dst = cv2.warpAffine(img,M,(cols,rows))           # Creates output image of same size with new perspective

## Perspective Transformation
# Kind of flattens out a *curved* image... see docs example
#Ex:
rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]]) # Sets 4 input points (3 can't be collinear)
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])     # Sets 4 outputs points
M = cv2.getPerspectiveTransform(pts1,pts2)               # Gets Matrix
dst = cv2.warpPerspective(img,M,(300,300))               # Uses warpPerspective to make warped image 


################ SMOOTHING IMAGES ################

## 2D Convolution
# Basically it's just convolving an area, and getting its sumValue/num (like what gaussian blur uses)
kernel = np.ones((5,5),np.float32)/25   # Creates convolution filter of 5x5
dst = cv2.filter2D(img,-1,kernel)       # Creates convolved image , where -1 = the entire image

## Image Blurring: Averaging
# same idea as above, except you're using a normalized averageing method
blur = cv2.blur(img,(5,5), normalize = True)    # basically blurs it with a 5x5 filter across entire image

## Image Blurring: Gaussian Filtering
#  Blurs image using Gaussian Function... has more params: std deviation X, Y  and variance X, Y 
blur = cv2.GaussianBlur(img,(5,5),0)  # where 0 is standard deviation of X, can add more params, see docs

## Image Blurring: Median Filtering
# Excellent for reducing noise, since you filter out all the unexpected out-of-bounds pixels
median = cv2.medianBlur(img,5)   ## makes filter of size 5x5

## Image Blurring: Bilateral Filtering  
# Uses Gaussian to blur images, while leaving edges!!!!!!!!!!!!!    ####VERY USEFUL WOOOOOOOOOOOOOOOOOOOW
# Basically works by comparing intensity differences, so edges will have very different intesities, so 
# they won't be filtered, whereas other similair points will
blur = cv2.bilateralFilter(img,9,75,75)   # 9 = size of block, 75 = param that doesn't do much.. leave at 75



############# MORPHOLOGICAL TRANSFORMATIONS ##################

## Erosion
## Used with BINARY PICS ONLY
# A filter goes through the image, and if all the pixels != 1, then the entire block = 0.....#eroded
kernel = np.ones((5,5),np.uint8)   
erosion = cv2.erode(img,kernel,iterations = 1)# Can iterate multiple times to keep eroding image more and more


## Dilation 
## Opposite of erosion... if at least one pixel is '1', changes entire regoin to all '1's. 
### USE: USE DILATION AFTER EROSION TO BRING IMAGE BACK TO NORMAL SIZE, also joins broken parts together
dilation = cv2.dilate(img,kernel,iterations = 1)

## Opening
# Erodes, then dilates  ....good for removing noise
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

## Closing
# Dilates, then Erodes.... good for filling holes in broken image (negative noise)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

## Morphological Gradient
# Difference between erosion and dilation of an image .... basically only leaves outline of image
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

## STRUCTURING ELEMENT:
# makes the numpy array for you... VERYYYYYYYYYYYYYYYY USEFUL IF YOU GET LOST
cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
""" output:
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]], dtype=uint8) """