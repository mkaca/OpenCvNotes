########################################## Machine Learning Training  ###################################

import numpy as np
import cv2
from matplotlib import pyplot as plt


##################  K- Nearest Neighbour (kNN) ############################

# This is used for supervised learning
""" 
	Literally classifies an object based on which family/class he is physically closest to....
	 However, it needs k nearest family members!! So , if k = 3, it looks for the 3 closest members
	     to the object, and majority wins..... 
	        k must be an ODD number !!!!
	    So, items close to the object get higher weights than those that are far.... simple as that, 
	       and higher summation wins
Basic Example: """

# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)
# Labels each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0,2,(25,1)).astype(np.float32)
# init newcomer to be analyzed
newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)        ### getting a funky error here
ret, results, neighbours ,dist = knn.findNearest(newcomer, k=3)   
"""
result:  [[ 1.]]                  # classifies the newcomes to class 1
neighbours:  [[ 1.  1.  1.]]      # closest 3 member belong to class 1
distance:  [[ 53.  58.  61.]]
"""


##################### OCR of HAND-WRITTEN DATA USING KNN ###############################

# Read digits using KNN
""" Gonna use opencv/samples/python2/data to get 10 digits (500 each) of size 20x20 .
      then gonna flatten them to 400 pixels, use first 250 samples to train, and second 250 samples to test"""


img = cv2.imread('C:\\Users\\dabes\\Downloads\\opencv\\sources\\samples\\data\\digits.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare train_data and test_data.
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print (accuracy)



######## you should save the trained data to save time for next time you run it as such :

# save the data
np.savez('knn_data.npz',train=train, train_labels=train_labels)

# Now load the data
with np.load('knn_data.npz') as data:
    print (data.files)
    train = data['train']
    train_labels = data['train_labels']



# Read letters using KNN:
# Load the data, converters convert the letter to a number
path = 'C:\\Users\\dabes\\Downloads\\opencv\\sources\\samples\\data\\letter-recognition.data'
data= np.loadtxt(path, dtype= 'float32', delimiter = ',',
                    converters= {0: lambda ch: ord(ch)-ord('A')})

# split the data to two, 10000 each for train and test
train, test = np.vsplit(data,2)

# split trainData and testData to features and responses
responses, trainData = np.hsplit(train,[1])
labels, testData = np.hsplit(test,[1])

# Initiate the kNN, classify, measure accuracy.
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
ret, result, neighbours, dist = knn.findNearest(testData, k=5)      ### use this to evaluate

correct = np.count_nonzero(result == labels)
accuracy = correct*100.0/10000
print (accuracy)




###### SVM (see docs) #############





#################################### K-MEANS CLUSTERING ###############################

""" basically divides data into groups ....
 For example, grouping t-shirt sizes into small, medium, and large based on people's height vs weight
 How algorithm works:
    step 1. Algorithm randomly chooses two centroids  (c1 and c2)
    step 2. calculates distance from each point to both centroids... if test data point is closer to c1,
             then it is labeled as 0, if it is closer to c2, it's labeled as 1..... if more centroids
                are used, then they're labeled as 2,3, etc....
    step 3. Then we seperately calculate the average of all grouped points (0,1,2,3,etc.), and set
     our new centroids based on the average
    step 4. Repeat steps 2 and 3 until centroids are converged or until a minimum centroid to point 
              distance is reached (c1-point1 < maxDistance)
    Input Params:
    	1. Samples ....np.float32  , each feature should be in a single column
    	2. nclusters(K)...number of clusters required at end
    	3. criteria....iteration termination criteria, when it's satisfied, algorithm stops
    		Has 3 params:  
    			a. type, type of iteration: cv2.TERM_CRITERIA_EPS...stops when accuracy(epsilon) is reached
		 	            cv2.TERM_CRITERIA_MAX_ITER ....stops when num of iterations is reached
		 	            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER....stops when any condition above 
		 	            	is reached
		 	    b. max_iteration..... max number of iterations (int)
		 	    c. epsilon ..... required accuracy
		4. attempts .... flag to specify the # of times the algo is executed using diff initial labelings
		5. Flags... specifies how init centers are taken: cv2.KMEANS_PP_CENTERS or cv2.KMEANS_RANDOM_CENTERS
	Ouput Params:
		1. Compactness... sum of squared distance from each point to their corresponding centers
		2. Labels ... this is the label array (so which class it belongs to, like above 0,1,2,3,etc..)
		3. Centers .... array of centers of clusters """

#Single Feature Example:

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#let z be a randomly generated 1D feature
z = np.random.randint(175,255,25)
z = np.float32(z.reshape((25,1)))
# Apply KMeans
compactness,labels,centers = cv2.kmeans(z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# split clusters based on their labels
A = z[labels == 0]
B = z[labels == 1]


# Multiple Feature Example:

X = np.random.randint(25,50,(25,2))
Y = np.random.randint(60,85,(25,2))
Z = np.vstack((X,Y))

# convert to np.float32
Z = np.float32(Z)

# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now separate the data, Note the flatten()
A = Z[label.ravel()==0]
B = Z[label.ravel()==1]

# Plot the data
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()



############## COLOR QUANTIZATION ###############

# reduces number of colors in an image...reduces memory used
# Idea is to reshape  R,G,B (3 features) to an array of M x 3 size where M = number of pixels in image
# Ex:

img = cv2.imread('TestImg.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8                                                           ## this is how many colors we will see
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()