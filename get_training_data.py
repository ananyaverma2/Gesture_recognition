import cv2
import time
import pickle
import numpy as np
import json


class mpHands:
    import mediapipe as mp
    # def __init__(self,maxHands=2,tol1=.5,tol2=.5):
    #     self.hands=self.mp.solutions.hands.Hands(False,maxHands,tol1,tol2)
    def __init__(self, mode=False, maxHands=2,modelC=1, tol1=0.5, tol2=0.5):
        
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = tol1
        self.trackCon = tol2
        self.hands=self.mp.solutions.hands.Hands(self.mode,self.maxHands, self.modelC ,self.detectionCon,self.trackCon)
        
    def Marks(self,frame):
        myHands=[]
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=self.hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
            for handLandMarks in results.multi_hand_landmarks:
                myHand=[]
                for landMark in handLandMarks.landmark:
                    myHand.append((int(landMark.x*width),int(landMark.y*height)))
                myHands.append(myHand)
        return myHands

def findDistances(handData):
    distMatrix=np.zeros([len(handData),len(handData)],dtype='float')
    palmSize=((handData[0][0]-handData[9][0])**2+(handData[0][1]-handData[9][1])**2)**(1./2.)
    for row in range(0,len(handData)):
        for column in range(0,len(handData)):
            distMatrix[row][column]=(((handData[row][0]-handData[column][0])**2+(handData[row][1]-handData[column][1])**2)**(1./2.))/palmSize
    return distMatrix

def findError(gestureMatrix,unknownMatrix,keyPoints):
    error=0
    for row in keyPoints:
        for column in keyPoints:
            error=error+abs(gestureMatrix[row][column]-unknownMatrix[row][column])
    print(error)
    return error
    
def findGesture(unknownGesture,knownGestures,keyPoints,gestNames,tol):
    errorArray=[]
    for i in range(0,len(gestNames),1):
        error=findError(knownGestures[i],unknownGesture,keyPoints)
        errorArray.append(error)
    errorMin=errorArray[0]
    minIndex=0
    for i in range(0,len(errorArray),1):
        if errorArray[i]<errorMin:
            errorMin=errorArray[i]
            minIndex=i
    if errorMin<tol:
        gesture=gestNames[minIndex]
    if errorMin>=tol:
        gesture='Unknown'
    return gesture


width=1280
height=720

gestNames = ['Nodding', 'Stop sign',
'Thumbs down',
'Waving',
'Pointing',
'Calling someone',
'Thumbs up',
'Wave someone away',
'Shaking head']

knownGestures = []
ges_names_received = []



#get labels
# Opening JSON file
f = open('data/training_labels.json')
# returns JSON object as 
# a dictionary
data = json.load(f)
  
# Iterating through the json

# data = {"video0000.mp4": [2], "video0001.mp4": [7], "video0002.mp4": [5], "video0003.mp4": [6], "video0004.mp4": [3], "video0005.mp4": [6], "video0006.mp4": [999], "video0007.mp4": [7], "video0008.mp4": [4], "video0009.mp4": [5], "video0010.mp4": [8],}
# data = {"video0000.mp4": [2], "video0001.mp4": [7], "video0002.mp4": [5], "video0003.mp4": [6], "video0004.mp4": [3],}

for key in data:
    cap = cv2.VideoCapture('data/training_set/{}'.format(key))
    print("reading video {}".format(key))

    findHands=mpHands(1)
    time.sleep(5)
    keyPoints=[0,4,5,9,13,17,8,12,16,20]

    s, frame = cap.read()

    if s==True:
        frame=cv2.resize(frame,(width,height))
        handData=findHands.Marks(frame)

        if handData!=[]:
            knownGesture=findDistances(handData[0])
            knownGestures.append(knownGesture)
            ges_name = (data[key])[0]-1
            print(ges_name, gestNames[ges_name])
            ges_names_received.append(gestNames[ges_name])
            # print("known gestures.......................", knownGestures)
            print("gesture names........................", len(ges_names_received))
            # print("gest names................................", gestNames)
    else:
        print("the video is empty")

finalGestures = []

for arr in knownGestures:
    finalGestures.append(arr.tolist())
print(knownGestures, finalGestures)


with open('gesture_names.json', 'w') as f:
    json.dump(ges_names_received, f)

with open('gestures.json', 'w') as f:
    json.dump(finalGestures, f)


# with open('test.pkl','wb') as f:
#     pickle.dump(ges_names_received,f)
#     pickle.dump(knownGestures,f)

# # Closing file
# f.close()