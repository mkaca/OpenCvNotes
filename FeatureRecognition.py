########################################## Feature Recognition Training  ###################################

import cv2
from matplotlib import pyplot as plt
import numpy as np 
import sys
#print(sys.path)
#print ('\n'.join(sys.path))
print("OpenCV Version: {}". format(cv2. __version__))

cap = cv2.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    _ , thresh = cv2.threshold(frame,127,255,cv2.THRESH_BINARY) 
    _ , thresh2 = cv2.threshold(frame,200,255,cv2.THRESH_BINARY)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create(threshold=25)
    # find and draw the keypoints
    kp = fast.detect(frame,None)
    img2 = cv2.drawKeypoints(frame, kp, None,color=(255,0,0))

    orb = cv2.ORB_create(WTA_K = 2)  # wta =2 is the default 
    kp = orb.detect(frame,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(frame, kp)
    img3 = cv2.drawKeypoints(frame,kp,None,color=(255,0,0))

    th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    imageHarrison = np.float32(gray)
    dst = cv2.cornerHarris(imageHarrison,2,3,0.04)
    #dst = cv2.dilate(dst,None)
    #frame[dst>0.01*dst.max()]=[0,0,255]
    corners = cv2.goodFeaturesToTrack(gray,25,0.1,10)
    corners = np.int0(corners)
    for i in corners:
    	x,y = i.ravel()
    	cv2.circle(frame,(x,y),3,255,-1)
    #plt.imshow(frame),plt.show()
    cv2.imshow('frame',frame)
    #cv2.imshow('tresh', thresh)
    #cv2.imshow('adaptive thresh', th2)
    cv2.imshow('img2', img2)
    cv2.imshow('img3', img3)
    #cv2.imshow('Harrison2', dst2)
    k = cv2.waitKey(5) & 0xFF  ### 5ms wait time 
    if k == 27:
        break
cv2.destroyAllWindows()


################# HARRIS CORNER DETECTION #################################

# uses color intensity differences and sums of the results to get hte eigenvales which determine if an area is an edge, corner, or flat
# Basically just finds corners  
#NOTE:         NOTE SCALE INVARIANT!!!!!!!!!!.... so SCALE will mess up results
# Params:     
#          img (must be grayscale float 32)
#          blockSize .... this is the size of neighbourhood considered       .... LARGE blocksize recognizes less lines
#          ksize .... this is the aperture param for the Sobel derivative used... LARGE ksize outputs thicker lines 
#          k ... this is the Harris detector free parameter in the equation   ... Wierd value... magic value seems to be around 0.04
# Ex:
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
### Which is basically the same as :
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)

# Can find more accurate corners with sub-pixel accuracy centroid method:
####   https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html ####



##################  SHI-TOMASI CORNER DETECTOR ###################

### almsot same thing as HARRIS CORNER DETECTION except slightly different
# Basically just finds corners
# Params:
#			img (must be greyscale)
# 			number of corners you want to find   ...finds highest quality corners I think
#           quality level (between 0 and 1)... which is basically a threshold for quality of corners found to be accepted
#           minimum distance between corners detected
# Ex:
img = cv2.imread('simple.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)
plt.imshow(img),plt.show()     ### wouldnt recommend using plt.imshow() 
# OR to be more dynamic:
cv2.imshow('frame',frame)



########################## Scale-Invariant Feature Transform (SIFT) ##########################
# 								CANT USE CUZ ITS PATENTED

"""                         1. Scale-space Extrema Detection   : finds keypoints
Uses Laplacian of Gaussian..... one parameter (sigma) determines what size of corners we find
         big sigma finds big corners, small sigma finds small corners
  uses x,y,sigma   ....so a potential keypoint at x,y at sigma scale
Since this is hella costly, it uses differnece of Gaussians (DoG) instead.. and then finds local extrema
SO: 1. Finds DoG blurring of an image with two diff sigmas   ....done repeatedly at diff scales (block sizes)
   2. Finds local extrema over scale and space
      for example:   one pixel is compared with its 8 neighbours, it's also compared with the 9 pixels in0
        next scale, and also compared with 9 pixels in previous scale... local extrema = potential keypoint
octaves = 4, number of scale levels = 5, initial sigma=1.6, second sigma=\sqrt{2} etc as optimal values.

                           2. Keypoint localization   : eliminates some keypoints
 - Refines previously found keypoints by using Taylor Series expansion of scale space to improve accuracy, if 
    the intensity at this extrema is less than a threshold value (0.03) then it's rejected #contrastThreshold 
 - Then we want to remove edges so it uses Hessian Matrix(H) to get curvature... 
       if (one eigen value > the other eigen) then it is an edge
       if ratio is greater than a threshold #edgeThreshold then the keypoint is discarded

                          3. Orientation Assignment   : stabilizes matching
 - neighbourhood aroudn keypoint is taken and its gradient magntidue and driection are found
 - creates a histogram with 36 bins covering 360 degrees...any peak above 80% of keypoint value is used
 - creates keypoints with same location and scale, but different orientation

 					      4. Key point Descriptor   : describes keypoint iwth a vector
 - 16x16 neighbourhood is taken around keypoint  ...divided into 16 subblocks of 4x4
 - each subblock has 8 bin orientation histogram made... so total: 128 bin values...represented as vector

                          5. Keypoint matching: matches keypoints and filters bad ones
 - ratio of second-closest to closest keypoint match is taken, if it's greater than 0.8 then they're rejected
 - eliminates 90% of bad matches and 5% of good ones

It's basically a heavy duty corner detection
"""
#Ex:
img = cv2.imread('home.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img3 = cv2.drawKeypoints(frame,kp,None,color=(255,0,0))
cv2.imwrite('sift_keypoints.jpg',img)



########################## Speeded-Up Robust Features (SURF) ##########################

# basically a faster SIFT ..... pretty useful, setting the threshold helps a lot!!! 
#    Higher treshold displays less points                  ...still patented though   :( 

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 4000
surf = cv2.xfeatures2d.SURF_create(4000)
# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(frame,None)
img3 = cv2.drawKeypoints(frame,kp,None,color=(255,0,0))
#                                                          I like this one ...has potential



#####################  FAST FEATURE DETECTOR in OPENCV

# Detects corners FASTTTTTT AF
""" USED IN SLAM!!!!!!!!!!!!!!!!!!! WOAH SUPER USEFUL
1. Selects a pixel (random) with intensity Ip 
2. Selects threshold value T
3. Consider a circle of 16 pixels aroudn the pixel under test.... see docs for better visualization
4. a corner exists if n continious pixels in the circle (of 16 pixels) are all either brighter (Ip + t) 
       or darker (Ip - t) than the selected pixel.
5. Performs a high speed test: where it first checks if the top, bottom , left, and right most pixels in
  in the circle fill the criteria ..at least 3 have to pass for the algorithim to proceed... this saves time!

Ex:"""
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create(threshold=25)
# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, None,color=(255,0,0))

# Disable nonmaxSuppression                              #### This basically detects more points
fast.setBool('nonmaxSuppression',0)
fast.setNonmaxSuppression(0)
img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

# Print all default params
print ("Threshold: ", fast.getInt('threshold'))
print ("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
print ("neighborhood: ", fast.getInt('type'))
print ("Total Keypoints with nonmaxSuppression: ", len(kp))



################################## BRIEF (Binary Robust Independent Elementary Features)   #################

#  Fast way of obtaining a descriptor!!  It doesn't actually find anything!!!!!!!!!!!!!!
# How it works: gets a set (n) of location pairs (x,y) that are uniquely selected, location pair = p, q
#      Then it checks if I(p) < I(q) ....if so then result = 1, else result = 0
#       You do this for all pairs and get a resultant bitstring of all 0s and 1s. 
# In the example below we use the CenSurE detector aka the STAR detector with BRIEF:
# Initiate STAR detector
star = cv2.xfeatures2d.StarDetector_create(25)
# Initiate BRIEF extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# find the keypoints with STAR
kp = star.detect(frame,None)
# compute the descriptors with BRIEF
kp, des = brief.compute(frame, kp)
img3 = cv2.drawKeypoints(frame, kp, None,color=(255,0,0))
#             Basically it's a faster method... but i still prefer FastFeatureDetector_create



########################## ORB (oriented FAST and Rotated BRIEF) ################################

""" NOT PATENTED WOW! 
# Basically a combination of BRIEF and FAST
 1. Uses FAST to find keypoints
 2. Then applies Harris corners to find top N points among them
 3. Uses pyramid to produce multiscale features
 4. Gets orientation using standard centroid vector method with a circular region of radius r... see docs
 5. Uses BRIEF with some extra matrice stuff (see docs) to get orientation to the nearest 12 degrees
 6. Constructs a lookup table of precomputed BRIEF patterns so it's faster! 
 7. For best results ORB looks for all possible binary tests and finds the one that has a high variance and
      a mean close to 0.5...... as well as being uncorrelated... this result is called:
      				rBRIEF

   Apparently ORB has a better descriptor than SURF and SIFT and is obviously faster so LITTTTTTTTTt

  Additional Params:
  		nFeatures: max numb of features to be retained (default =500)
  		scoreType: tells to use Harris or Fast score (default = Harris)
  		WTA_K: number of points that produce each element of the oriented BRIEF descripter (default = 2)
  		       aka is selects 2 points at a time
EX:
"""
#Init ORB
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# Create image based on our kp values 
img3 = cv2.drawKeypoints(frame,kp,None,color=(255,0,0))

#### SO far this is my fav..... gives lots of detail. yay 




##################### BRUTE FORCE FEATURE MATCHING ##########################

# basically matches a feature with other features in an image
#Example with ORB and sift detecter
img1 = cv2.imread('box.png',0)          # queryImage
img2 = cv2.imread('box_in_scene.png',0) # trainImage
# Initiate SIFT detector
orb = cv2.ORB()
# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)   ### use crossCheck for better results, 
#           											use norm_hamming for ORB ALWAYS
# Match descriptors.
matches = bf.match(des1,des2)     ## see docs about more details on this
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)




#################### FLANN FEATURE MATCHING ###################

"""# FLANN = Fast library for approximate nearest neighbours
 Faster than BFMatcher with large datassets
Takes two dictionary parameters!! index and search:
	 # used for SIFT and SURF
     	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)   
     # Used for ORB:      ... commented values are recommended values by docs
	     index_params= dict(algorithm = FLANN_INDEX_LSH,
	                   table_number = 6, # 12
	                   key_size = 12,     # 20
	                   multi_probe_level = 1) #2
	 Search param specifies number of times the tree in the index should be recursievly traversed
	               highe values = better precision but take longer 
	     search_params = dict(checks =100)
Ex:
"""
img1 = cv2.imread('box.png',0)          # queryImage
img2 = cv2.imread('box_in_scene.png',0) # trainImage
# Initiate SIFT detector
sift = cv2.SIFT()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)




### another method of object matching (backup) is with Homography
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html