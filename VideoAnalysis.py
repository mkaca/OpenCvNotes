########################################## Video Analysis Training  ###################################

import numpy as np
import cv2

################# MEAN SHIFT #################################

#basically finds the most concentrated point on the image (like brighest or whatever)
# Basically in applications, it can track a moving object
### PRETTTTTTTTTY COOOOL
# However, the window size doesnt change which sucks, we fix this with camshift


############      CAMSHIFT #####################################

# Gets meanshift, and once it converges, it updates size of window with a best fitting elipse. 
# Then it meanshifts again, and repeats process until required accuracy is met

### see combined example below
###    the code below basically gets an intital image with its thresholds, and tries to find the 
###       highest concentrated area (usually brightness / intensity)

cap = cv2.VideoCapture(0)
# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)
# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))  ## avoids low light values
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        """
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        x,y,w,h = track_window               # Draw it on image
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2) """

        # apply camshift which we are gonna use
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        cv2.imshow('img2',img2)
        cv2.imshow("hsv_roi",hsv_roi)
        cv2.imshow("mask",mask)
        cv2.imshow("hsv",hsv)
        cv2.imshow("dst",dst)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)
    else:
        break

cv2.destroyAllWindows()
cap.release()




############################## OPTICAL FLOW ###########################################

# Gets an Intesntiy at a point , I(x,y,t)  ... and rearranges it to get fx*u + fy*v +ft = 0
#     We don't have u or v , so we use Lucas-Kanade method to find them.
"""  Lucas Kanade: all neighbours pixels will ave similair motion so take a 3x3 patch around a point, and find
       fy, fy, and ft for these 9 poiints. So now we ahve 9 equations with 2 unknowns....easily solvable
          Focuses on harris corners as points to track to get better results
            It also scales well since it uses the pyramid thingy
   BASICALLY: it shows you the path of a moving object
    THIS IS ACUALLY REALLL COOOOOL... can find an image and then track its path and tell you if it
        is getting closer or furhter for example 
Ex:
"""
cap = cv2.VideoCapture(0)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]    ## st ==1 is a param returned as 1 from opticalFlowPyrLK if next point is found
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    cv2.imshow('mask',mask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()


########### 

#DENSE OPTICAL FLOW EXAMPLE:
## this basically shows all the moving objects in the image with hue values

cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

cap.release()
cv2.destroyAllWindows()




############################### BACKGROUND SUBTRACTION ###################################

### THIS IS VERY IMPORTANT in MANY VISION APPLICATIONS
#   basically subtracts static background and slightly-moving ones like shadows from frame


############ BackgroundSubtractorMOG ####################
# Ex:
cap = cv2.VideoCapture(0)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()   ### init frame for subtraction

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


############ BackgroundSubtractorMOG2 ####################

# basically same as MOG except it's better since it adapts to illumination changes
# it pretty much is accurate enough to show shadows which is awesome!!!
#Ex:
cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


################ BackgroundSubtractorGMG  ##########################

# uses first few frames (120 by default) to model background, then uses opening and closing filters
#   to remove background noise ..... 
#       first few frames are always black!!!!!
# WOAAAOOAOAOAOA this is precise as fuckkkkkkkkkkkkkkkkkkkkkkkkk
# Ex:
cap = cv2.VideoCapture(0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()




############################# Face Detection using Haar Cascades ################################

""" basically takes the fact that eyes are darker than its surroundings to find them in an image using
     a summation of points within a given rectangular area ..... 
       since this is so computationally heavy, algorithm focuses on discarding areas that are most likely
         not faces first... aka CASCADE OF CLASSIFIERS  ...aka if a window passes first stage, it goes to
            the second stage, then the 3rd stage and so on, until it classifies a face or fails at a stage

  OpenCv comes with a trainer and a detector .... you can train your own classifier for any object like car
    planes, etc... but you have to create it .... see existing pre-trained classifiers for face, eyes, smile,
     etc, in opencv/data/haarcascades    ....

  Below is an example of creating a face and eye detector in opencv:"""

# load cascade classifiers: 
face_cascade = cv2.CascadeClassifier('C:\\Users\\dabes\\Downloads\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\dabes\\Downloads\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while(1):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # if faces are found, returns x,y,w,h  
    #   draw the faces and then search for eyes within the faces and draw them!!!
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            if ew*3 < w and ey*3 < h:
              cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()