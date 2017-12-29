#OpenCV CHeat Sheet WIHH A BUNCH OF COMMANDS!!

import cv2 
import numpy as numpy
img = cv2.imread("picturename.jpg")


#### REMEMBER IT'S BGR FORMAT
pixel = img[100,100]
print (pixel)    ### would output for example, "[157 153 200]"

#can modify specific pixels
img [100,100] = [255,255,255]
print (img[100,100])   ### [255,255,255]  .... basically it would turn it white

# modifying RED value
img.itemset((10,10,2),100)    # Changes intensity of pixel's RED value to 100
print (img.item(10,10,2))    # 100

print (img.shape)   ## Returns size of image... example (354 , 548, 3)

print (img.size)   ## returns number of pixels ...example (562248)

print (img.dtype) ## returns DATATYPE .... #### IMPORTANT FOR DEBUGGING######## example : uint8 

### Copying fragment of picture 
ball = img[280:340, 330:390]
### ANDDD pasting that fragment of a picture onto this
img[273:333, 100:160] = ball

#### Set all red pixels to zero #### works as an imaging filter
img[:,:,2] = 0


#### Make border around image
cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REPLICATE) 
 ### Params: image, border top width, border bottom width,border left width, border right width, borderType
 ##### Border types: cv2.BORDER_CONSTANT , BORDER_WRAP , etc.... see docs)


########## ARITHMETIC OPERATIONS ON IMAGES ####################

"""### Addition : opencv addition is saturated, numpy addition is modulo
     ### example : 
x = np.uint8([250])
y = np.uint8([10])

print cv2.add(x,y) # 250+10 = 260 => 255
print x+y          # 250+10 = 260 % 256 = 4   """


### Image blending   ####     can use this to make a nice transparent looking image... see docs
### EQUATION: dst = a*img1 + B*img2 + c
#ex:
img1 = cv2.imread('ml.png')
img2 = cv2.imread('opencv_logo.jpg')

dst = cv2.addWeighted(img1,0.7,img2,0.3,0)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

#######   BITWISE OPERATIONS: AMAZINGGGGGGGGGGGGGGGGGGGGGGG FEATURE: See : for image example: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html
#Code example:           this merges two non square images, kinda complex tbh
# Load two images
img1 = cv2.imread('messi5.jpg')
img2 = cv2.imread('opencv_logo.png')
# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]
# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()


##### Gaussian blur: averages groups of pixels to make one blurry image
#Ex:
blur = cv2.GaussianBlur(img,(5,5),0)

