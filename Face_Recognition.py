from datetime import datetime
startTime = datetime.now()
print("Starting time of the process :", startTime)

import dlib
import cv2
from cv2 import imread
import numpy as np 
import os
from sklearn.svm import LinearSVC
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
# from scipy.optimize import minimize
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold

folder = '/Users/lenin.s/Desktop/SOLVERMINDS/PROJECTS/EDGE/FACE_RECOGNITION/python_edge_face/face_project'
root='/Users/lenin.s/Desktop/SOLVERMINDS/PROJECTS/EDGE/FACE_RECOGNITION/python_edge_face/face_project/SVM_FACES_NEW'
unkroot = '/Users/lenin.s/Desktop/SOLVERMINDS/PROJECTS/EDGE/FACE_RECOGNITION/python_edge_face/face_project/unk.npy'
unkroots = '/Users/lenin.s/Desktop/SOLVERMINDS/PROJECTS/EDGE/FACE_RECOGNITION/python_edge_face/face_project/unk2.npy'
smodel = '/Users/lenin.s/Desktop/SOLVERMINDS/PROJECTS/EDGE/FACE_RECOGNITION/python_edge_face/face_project/shape_predictor_68_face_landmarks.dat'
model = '/Users/lenin.s/Desktop/SOLVERMINDS/PROJECTS/EDGE/FACE_RECOGNITION/python_edge_face/face_project/dlib_face_recognition_resnet_model_v1.dat'




image_size = 128
num_channels = 3
bar = 0.44          #percentage similiraity for detection 
train_nos = 2
#bar = (100-bar)/100

"""
Initialize Models 
"""
facerec = dlib.face_recognition_model_v1(model)  #feature vector resnet model
shaper = dlib.shape_predictor(smodel)            #coordinate helper for splicing face from image
detector = dlib.get_frontal_face_detector()      #face box detector

"""
Get labels from the given folder with paths for all the labels and number of labels
this label data wil return classnames directory for each classes and number of classes
"""
def labelData(path):
  classes = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)) and item!="unk"]
  return classes, [root+'/'+str(x) for x in classes],len(classes)

"""
Helper function to convert image at path into feature space with side variable 
denoting the number of face detected
"""
###def getVec(path):
#  img  = imread(path,1)
#  output = np.zeros(129)
#  dets,scores,idx = detector.run(img,1)
#  print("Number of faces detected: {}".format(len(dets)))
#  output[0] = len(dets)
#  if (output[0] > 0):
#    output[1:] = np.array(facerec.compute_face_descriptor(img,shaper(img,dets[int(np.argmax(scores))])))
#  else:
#    output[1:] = np.zeros((128))
#    #print(path)
###  return output

def getVec(img):
  if type(img) == str and type(img) != np.ndarray : img = imread(img,1)
  output = np.zeros(129)
  try:
    dets,scores,idx = detector.run(img,1)
    print("Number of faces detected: {}".format(len(dets)))
    output[0] = np.sign(len(dets))
    if (output[0] > 0):
      output[1:] = np.array(facerec.compute_face_descriptor(img,shaper(img,dets[int(np.argmax(scores))])))
    else:
      output[1:] = np.zeros((128))
    
  except:
    print("Error in image, skipping")
    
  return output


def getVecI(img):
  output = np.zeros(129)
  dets,scores,idx = detector.run(img,1)
  output[0] = np.sign(len(dets))
  if (output[0] > 0):
    output[1:] = np.array(facerec.compute_face_descriptor(img,shaper(img,dets[int(np.argmax(scores))])))
  else:
    print("There is NO faces")  
    output[1:] = np.zeros((128))
  return output

"""
imPaths = Getting paths from the label using its index in classpath
collectValid = get feature vecs that are valid i.e, single face only images (only used for training and testing)
train = train the classfier and return the model
"""
def imPaths(i):
  imagepaths = [classpath[i]+'/'+item for item in os.listdir(classpath[i])]
  return imagepaths

def collectValid(vecs):
  V = np.delete(vecs[vecs[:,1]>=1,:],1, 1)
  return V[:,1:], V[:,0]

def train(V_train,y_train):
  reg = LinearSVC()
  reg.fit(V_train,y_train)
  return reg

classes, classpath,num_classes = labelData(root)
# num_classes = 100
# a = np.array(classes)
# np.save('/Users/lenin.s/Desktop/SOLVERMINDS/PROJECTS/EDGE/FACE_RECOGNITION/classes.npy',a)

"""
Find the number of images in the available dataset
Intitalize training and testing feature vectors accordingly
Iterate over each label and save the feature vectors of each image with
corresponding index of the label
Get the valid feature vetors from the total dataset
Train the model and output the prediction score on the test dataset
"""
total = 0
for pathi in range(num_classes):
  imagepaths = imPaths(pathi)
  l = len(imagepaths)
  total += l
#vecs_full = np.zeros((total,130))
vecs_train = np.zeros((num_classes*train_nos,130))
vecs_test = np.zeros((total - num_classes*train_nos ,130))
jtr = 0
jts = 0
jtf = 0
for pathi in range(num_classes):
  imagepaths = imPaths(pathi)  #imagepaths will have a list of images path for a particular class iteration will go on
  l = min(10,len(imagepaths))
  for i in range(train_nos):
    vecs_train[jtr,0] = pathi
    vecs_train[jtr,1:] = np.array(getVec(imagepaths[i])).reshape((129))
    jtr+=1
  for i in range(train_nos,l):
    vecs_test[jts,0] = pathi
    vecs_test[jts,1:] = np.array(getVec(imagepaths[i])).reshape((129))
    jts+=1
  print("Extracted Class "+str(pathi+1)+" : "+ classes[pathi]) 

np.save('/Users/lenin.s/Desktop/SOLVERMINDS/PROJECTS/EDGE/FACE_RECOGNITION/training_model.npy', vecs_train)
np.save('/Users/lenin.s/Desktop/SOLVERMINDS/PROJECTS/EDGE/FACE_RECOGNITION/testing_model.npy', vecs_test)

print("Ending time of the process:",datetime.now())

EndTime = datetime.now() - startTime
hours, minutes =EndTime.seconds // 3600, EndTime.seconds % 3600 / 60.0
print("Time taken for complete training is "+str(hours)+" hours,"+str(minutes)+" mins")
#print("Time taken for complete training :", EndTime)



vecs_train = np.load('/Users/lenin.s/Desktop/SOLVERMINDS/PROJECTS/EDGE/FACE_RECOGNITION/training_model.npy')
vecs_test = np.load('/Users/lenin.s/Desktop/SOLVERMINDS/PROJECTS/EDGE/FACE_RECOGNITION/testing_model.npy')

V_train, y_train = collectValid(vecs_train)
V_test, y_test = collectValid(vecs_test)

#V_train = np.delete(V_train,[4,5],axis =0)
#y_train = np.delete( y_train,[4,5])

bar = 0.44
barrier = [0 for i in range(num_classes)]
for k in range(num_classes):
    dums = V_train[y_train == k]
    slen = 0
    for i in range(len(dums)):
      for j in range(len(dums)):
         if i!= j : 
             slen = max(slen,np.linalg.norm(dums[i]-dums[j]))
    
    dumd = V_train[y_train != k]

    dlen = 1
    for i in range(len(dums)):
      for j in range(len(dums)):
         dlen = min(dlen,np.linalg.norm(dums[i]-dumd[j]))
    
    barrier[k] = max(0.4,slen + (dlen-slen)/3)
    
    
# barrier = [min(i, bar) for i in barrier]
barrier = [0.44 for i in barrier]      
    
    

reg = train(V_train,y_train)
print(reg.score(V_test,y_test))


"""
getVecs = return all the image path and feature vectors of the corresponding
images from the given folder. Removes invalid images internally.
"""
"""
Get feature vectors for unknown images
Get the minimum distance for analysis
"""

V_unk = np.load('/Users/lenin.s/Desktop/SOLVERMINDS/PROJECTS/EDGE/FACE_RECOGNITION/python_edge_face/face_project/unk.npy')
V_unk2 = np.load('/Users/lenin.s/Desktop/SOLVERMINDS/PROJECTS/EDGE/FACE_RECOGNITION/python_edge_face/face_project/unk2.npy')


Eucdist = np.zeros((V_unk.shape[0]+V_test.shape[0]+V_unk2.shape[0],V_train.shape[0]))
for i in range(V_train.shape[0]):
  for j in range(V_unk.shape[0]):
    Eucdist[j,i] = np.linalg.norm(V_train[i,:]-V_unk[j,:])
for i in range(V_train.shape[0]):
  for j in range(V_unk2.shape[0]):
    Eucdist[j+V_unk.shape[0],i] = np.linalg.norm(V_train[i,:]-V_unk2[j,:])
for i in range(V_train.shape[0]):
  for j in range(V_test.shape[0]):
    Eucdist[j+V_unk.shape[0]+V_unk2.shape[0],i] = np.linalg.norm(V_train[i,:]-V_test[j,:])
Euarg = np.argmin(Eucdist,axis =1)  #will have the index of min value from each row
Eu = np.c_[np.min(Eucdist,axis =1),np.argmin(Eucdist,axis =1)]   #have the min values and the index of the value

cosdist = np.zeros((V_unk.shape[0]+V_test.shape[0]+V_unk2.shape[0],V_train.shape[0]))
for i in range(V_train.shape[0]):
  for j in range(V_unk.shape[0]):
    cosdist[j,i] = cosine(V_train[i,:],V_unk[j,:])
for i in range(V_train.shape[0]):
  for j in range(V_unk2.shape[0]):
    cosdist[j+V_unk.shape[0],i] = cosine(V_train[i,:],V_unk2[j,:])
for i in range(V_train.shape[0]):
  for j in range(V_test.shape[0]):
    cosdist[j+V_unk.shape[0]+V_unk2.shape[0],i] = cosine(V_train[i,:],V_test[j,:])
cosarg = np.argmin(cosdist,axis =1)
cos = np.c_[np.min(cosdist,axis =1),np.argmin(cosdist,axis =1)]

core = np.zeros((V_unk.shape[0]+V_test.shape[0]+V_unk2.shape[0],V_train.shape[0]))
for i in range(V_train.shape[0]):
  for j in range(V_unk.shape[0]):
    core[j,i] = np.corrcoef(V_train[i,:],V_unk[j,:])[0,1]
for i in range(V_train.shape[0]):
  for j in range(V_unk2.shape[0]):
    core[j+V_unk.shape[0],i] = np.corrcoef(V_train[i,:],V_unk2[j,:])[0,1]
for i in range(V_train.shape[0]):
  for j in range(V_test.shape[0]):
    core[j+V_unk.shape[0]+V_unk2.shape[0],i] = np.corrcoef(V_train[i,:],V_test[j,:])[0,1]
Eucore = np.argmax(core,axis =1)
Eucore = np.c_[np.max(core,axis =1),np.argmax(core,axis =1)]


"""
predictClass = predicts the class of the image at given path, other cases are 
also covered
predictFolder = predicts classes for all the images in a folder
"""
#pred = predictFolder('E:/Acads/9th sem/SVM/training_data/103')
#print(pred)

def distribution(data, bins = 10):
  tests = np.histogram(data,bins)
  plt.bar(tests[1][:-1], tests[0], width = tests[1][5]/100)
  
a = V_unk.shape[0]
b = V_unk2.shape[0]
#distribution(cos[:a,0])
#distribution(cos[a:a+b,0])
#distribution(cos[a+b:,0])
#distribution(Eu[:a,0])
distribution(Eu[a:a+b,0])
distribution(Eu[a+b:,0])

#def separation_error(m,Eu,a,w1=1,w2=1):
#  n1 = 1 - (sum(Eu[:a,0]>m)/a)
#  n2 = 1 - (sum(Eu[a:,0]<m)/(len(Eu)-a))
#  return ((n1**(1/w1))*a + (n2**(1/w2))*(len(Eu)-a))/len(Eu)
#
#def raw_error(m,Eu,a):
#  n1 = 1 - (sum(Eu[:a,0]>m)/a)
#  n2 = 1 - (sum(Eu[a:,0]<m)/(len(Eu)-a))
#  return n1,n2
#erlist = [separation_error(0.3+(i/250),Eu,b,1,1) for i in range(100)]
#erlist2 = [separation_error(0.3+(i/250),Eu,b,3,0.5) for i in range(100)]
#
#plt.plot(erlist)
#plt.plot(erlist2)
#
#res = minimize(separation_error,0.6,(Eu,b,1,0.4),method='Nelder-Mead', tol=1e-6)
#bar = res.x
#raw = raw_error(bar,Eu,b)
#accuracy = [100*(1-i) for i in raw]
#print(raw)
#print(accuracy)

bar = 0.44

def predictImage(img,reg=reg,V_train= V_train,bar = bar):
  #print(img)
  #print(img.shape)
  vec = np.array(getVecI(img)).reshape((129))
  #print(vec)
  #print(vec.shape)
  faces = vec[0]
  if(faces==0):
    return 'Face Not detected'
  elif(faces>1):
    return str(faces-1) + 'extra faces detected' 
  vec = vec[1:]
  Eucdist = np.zeros((1,V_train.shape[0]))
  for i in range(V_train.shape[0]):
    Eucdist[0,i] = np.linalg.norm(V_train[i,:]-vec)
  Eucore = np.zeros((1,V_train.shape[0]))
  for i in range(V_train.shape[0]):
    Eucore[0,i] = np.corrcoef(V_train[i,:],vec)[0,1]
  print(np.min(Eucdist,axis =1))
  print(np.min(Eucore,axis =1))
  beer = int(reg.predict(vec.reshape(1, -1))[0])
  print(beer)
  
  if(np.min(Eucdist,axis =1)>=barrier[beer]  or np.argmax(Eucore,axis =1)!=np.argmin(Eucdist,axis =1)):
    return 'Unknown'
  else:
    return classes[beer]
#path = "/Users/lenin.s/Desktop/SOLVERMINDS/PROJECTS/EDGE/FACE_RECOGNITION/python_edge_face/face_project/svm_faces/Mr.Mohit/IMG_0058.JPG"
#img = imread(path,1)
        
#print(predictImage(img[:,:,:]))    

cap = cv2.VideoCapture(0)
while(True):
  ret, img = cap.read()
  cv2.imshow('frame2',img)
  cv2.waitKey(30)
  print(predictImage(img[:,:,:]))
  
cap.release()
  



#seconds = 0.1
#arnold = cv2.VideoCapture(folder+'Arnold.mp4')
#for j in range(10):
#  for i in range(int(29*seconds)): ret, img = arnold.read()
#  cv2.imshow('img', img)
#  print(predictImage(img[:,:800,:]))

#harrison= cv2.VideoCapture(folder+'harrison.mp4')
#for j in range(10):
#  for i in range(int(29*seconds)): ret, img = harrison.read()
#  cv2.imshow('img', img)
#  print(predictImage(img[:,:,:]))

#arnold = cv2.VideoCapture('C:/Users/Varun/Pictures/Camera Roll/WIN_20181116_163230.MP4')
#for j in range(1000):
#  for i in range(int(29*seconds)): ret, img = arnold.read()
#  cv2.imshow('img', img)
#  cv2.waitKey(30)
#  print(predictImage(img[:,:,:]))





