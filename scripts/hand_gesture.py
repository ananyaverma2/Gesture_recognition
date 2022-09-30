import time
import cv2
import numpy as np
import pickle

import rospy
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge, CvBridgeError
from metrics_refbox_msgs.msg import Command, GestureRecognitionResult



class Gesture_recognition():
    import mediapipe as mp
    def __init__(self) -> None:
        rospy.loginfo("Gesture recognition node is ready...")
        self.cv_bridge = CvBridge()
        self.image_queue = None
        self.clip_size = 300 # manual number
        self.stop_sub_flag = False
        self.cnt = 0
        self.mode = False
        self.maxHands = 2
        self.modelC = 1
        self.detectionCon = 0.5
        self.trackCon = 0.5
        self.hands=self.mp.solutions.hands.Hands(self.mode,self.maxHands, self.modelC ,self.detectionCon,self.trackCon)
        self.width=1280
        self.height=720

        # # HSR pan motion publisher
        # self.hsr_pan_pub = rospy.Publisher(
        #     '/hsrb/head_trajectory_controller/command',
        #     trajectory_msgs.msg.JointTrajectory, queue_size=10)
        # self.move_right_flag = False
        # self.move_left_flag = True

        # set of the HSR camera to get front straight view
        self.move_front_flag = False
        # self._hsr_head_controller('front')

        # subscriber
        # self.requested_object = None
        self.referee_command_sub = rospy.Subscriber(
            "/metrics_refbox_client/command", Command, self._referee_command_cb)

        # publisher
        self.output_bb_pub = rospy.Publisher(
            "/metrics_refbox_client/gesture_recognition_result", GestureRecognitionResult, queue_size=10)



    def _referee_command_cb(self, msg):

        # Referee comaand message (example)
        '''
        task: 4
        command: 1
        task_config: "{}"
        uid: "0888bd42-a3dc-4495-9247-69a804a64bee"
        '''

        # START command from referee
        if msg.task == 4 and msg.command == 1:

            print("\nStart command received from refree box")

            # # set of the HSR camera to get front straight view
            # if not self.move_front_flag:
            #     self._hsr_head_controller('front')

            # start subscriber for image topic
            self.image_sub = rospy.Subscriber("/camera/color/image_raw",
                                              Image,
                                              self._input_image_cb)


        # STOP command from referee
        if msg.command == 2:

            self.image_sub.unregister()
            self.stop_sub_flag = False
            rospy.loginfo("Received stopped command from referee")
            rospy.loginfo("Subscriber stopped")


    def _input_image_cb(self, msg):
        """
        :msg: sensor_msgs.Image
        :returns: None
        """

        try:
            if not self.stop_sub_flag:

                # convert ros image to opencv image
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                if self.image_queue is None:
                    self.image_queue = []

                self.image_queue.append(cv_image)
                # print("Counter: ", len(self.image_queue))

                if len(self.image_queue) > self.clip_size:
                    # Clip size reached
                    # print("Clip size reached...")
                    rospy.loginfo("Image received..")

                    self.stop_sub_flag = True

                    # pop the first element
                    self.image_queue.pop(0)

                    # deregister subscriber
                    self.image_sub.unregister()

                    # call object inference method
                    print("converted to ros image")
                    head_gesture_show = self.hand_gesture_recognition()

        except CvBridgeError as e:
            rospy.logerr(
                "Could not convert ros sensor msgs Image to opencv Image.")
            rospy.logerr(str(e))
            return



    def Marks(self,frame):
        myHands=[]
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=self.hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
            for handLandMarks in results.multi_hand_landmarks:
                myHand=[]
                for landMark in handLandMarks.landmark:
                    myHand.append((int(landMark.x*self.width),int(landMark.y*self.height)))
                myHands.append(myHand)
        return myHands


    def findDistances(self, handData):
        distMatrix=np.zeros([len(handData),len(handData)],dtype='float')
        palmSize=((handData[0][0]-handData[9][0])**2+(handData[0][1]-handData[9][1])**2)**(1./2.)
        for row in range(0,len(handData)):
            for column in range(0,len(handData)):
                distMatrix[row][column]=(((handData[row][0]-handData[column][0])**2+(handData[row][1]-handData[column][1])**2)**(1./2.))/palmSize
        return distMatrix

    def findError(self, gestureMatrix,unknownMatrix,keyPoints):
        error=0
        for row in keyPoints:
            for column in keyPoints:
                error=error+abs(gestureMatrix[row][column]-unknownMatrix[row][column])
        # print(error)
        return error

    def findGesture(self, unknownGesture,knownGestures,keyPoints,gestNames,tol):
        errorArray=[]
        for i in range(0,len(gestNames),1):
            error=self.findError(knownGestures[i],unknownGesture,keyPoints)
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
        return gesture, gestNames[minIndex]

    def hand_gesture_recognition(self):
        # cam=cv2.VideoCapture(0)
        findHands=Gesture_recognition()
        time.sleep(5)
        keyPoints=[0,4,5,9,13,17,8,12,16,20]
        hand_gesture = []
        gesture_detection_msg = GestureRecognitionResult()
        gesture_detection_msg.message_type = GestureRecognitionResult.RESULT
        

        with open('default.pkl','rb') as f:
            gestNames=pickle.load(f)
            knownGestures=pickle.load(f)
        
        tol=20

        for number in range(1,len(self.image_queue)):   
            frame = self.image_queue[number]
            frame=cv2.resize(frame,(self.width,self.height))
            handData=findHands.Marks(frame)

            if handData!=[]:
                if len(hand_gesture) < 10:
                    unknownGesture=self.findDistances(handData[0])
                    myGesture, gesture_true=self.findGesture(unknownGesture,knownGestures,keyPoints,gestNames,tol)
                    print("with error the gesture is : ", myGesture, "without error : ", gesture_true)
                    hand_gesture.append(gesture_true)
                    #error=findError(knownGesture,unknownGesture,keyPoints)
                    # cv2.putText(frame,myGesture,(100,175),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),8)
                    # print("the detected gesture is : ", myGesture)
                else:
                    break
            # else:
            #     print("no hand gesture detected")
            # for hand in handData:
            #     for ind in keyPoints:
            #         cv2.circle(frame,hand[ind],25,(255,0,255),3)
            # cv2.imshow('my WEBcam', frame)
            # cv2.moveWindow('my WEBcam',0,0)
            # if cv2.waitKey(1) & 0xff ==ord('q'):
            #     break
        # cam.release()

        print("the gestures are ", hand_gesture)
        if len(hand_gesture) > 0:
            print(" the final gesture is ", hand_gesture[0])
            # gesture_detection_msg.gestures = gesture
            gesture_detection_msg.gestures = hand_gesture
        self.output_bb_pub.publish(gesture_detection_msg)
        print("gesture published")
        

if __name__ == "__main__":
    rospy.init_node("gesture_recognition_node")
    gesture_recognition_obj = Gesture_recognition()

    rospy.spin()


