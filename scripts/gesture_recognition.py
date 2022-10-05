#!/usr/bin/env python3

import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge, CvBridgeError
from metrics_refbox_msgs.msg import Command, GestureRecognitionResult

from std_msgs.msg import String

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('nodcontrol.avi',fourcc, 20.0, (640,480))


class Gesture_recognition():
    def __init__(self) -> None:
        rospy.loginfo("Head gesture recognition node is ready...")
        self.cv_bridge = CvBridge()
        self.image_queue = None
        print("setting image queue to zero 3")
        self.clip_size = 100  # manual number
        self.stop_sub_flag = False
        self.cnt = 0
        self.gesture_result = None
        # self.image_sub = None

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

        # self.gesture_recognition_ack_pub = rospy.Publisher(
        #     "/gesture_ack", String, queue_size=10)

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

            self.gesture_result = None

            print(
                "\n[head gesture recognition] Start command received from refree box for ")

            # # set of the HSR camera to get front straight view
            # if not self.move_front_flag:
            #     self._hsr_head_controller('front')

            # start subscriber for image topic
            self.image_sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw",
                                              Image,
                                              self._input_image_cb)

        # STOP command from referee
        if msg.command == 2:

            self.image_sub.unregister()
            self.stop_sub_flag = False
            self.gesture_result = None
            rospy.loginfo(
                "Received stopped command from referee head gesture recognition")
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
                    print("setting image queue to zero 2")

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
                    print("converted to ros image for head gesture recognition")
                    head_gesture_show = self.head_gesture_recognition()

        except CvBridgeError as e:
            rospy.logerr(
                "Could not convert ros sensor msgs Image to opencv Image.")
            rospy.logerr(str(e))
            return

    # dinstance function
    def distance(self, x, y):
        import math
        return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

    # function to get coordinates
    def get_coords(self, p1):
        try:
            return int(p1[0][0][0]), int(p1[0][0][1])
        except:
            return int(p1[0][0]), int(p1[0][1])

    def head_gesture_recognition(self):

        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # path to face cascde
        face_cascade = cv2.CascadeClassifier(
            f"{cv2.data.haarcascades}haarcascade_frontalface_alt.xml")

        # define font and text color
        font = cv2.FONT_HERSHEY_SIMPLEX

        # define movement threshodls
        max_head_movement = 20
        movement_threshold = 50
        gesture_threshold = 120

        # find the face in the image
        face_found = False
        frame_num = 0
        if not face_found and frame_num < 5:
            # print(" stuck here ")
            # Take first frame and find corners in it
            # # #capture source video
            # cap = cv2.VideoCapture("data/video0096.mp4")
            # ret, frame = cap.read()
            frame = self.image_queue[frame_num]

            # if ret==True:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_found = True
                print("[head gesture recognition] Face found: ", face_found)
                # cv2.imshow('image',frame)
                # out.write(frame)
                # cv2.waitKey(1)
            frame_num += 1

        if face_found:
            face_center = x+w/2, y+h/3
            p0 = np.array([[face_center]], np.float32)
            print("the face center is ", face_center)

            gesture = []
            x_movement = 0
            y_movement = 0
            gesture_show = 120  # number of frames a gesture is shown
            frame_counter = 0
            # print("length of the image queue is ", len(self.image_queue))

            for number in range(1, len(self.image_queue)):
                # print(" the frame number is ", number)
                if len(gesture) < 10:
                    # print("gesture")
                    frame = self.image_queue[number]
                    frame_counter += 1
                    # print(frame)
                    # #If the last frame is reached, reset the capture and the frame_counter
                    # if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    #     frame_counter = 0 #Or whatever as long as it is the same as next line
                    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    old_gray = frame_gray.copy()
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(
                        old_gray, frame_gray, p0, None, **lk_params)
                    cv2.circle(frame, self.get_coords(p1), 4, (0, 0, 255), -1)
                    cv2.circle(frame, self.get_coords(p0), 4, (255, 0, 0))

                    # get the xy coordinates for points p0 and p1
                    a, b = self.get_coords(p0), self.get_coords(p1)
                    x_movement += abs(a[0]-b[0])
                    y_movement += abs(a[1]-b[1])

                    # text = 'x_movement: ' + str(x_movement)
                    # if not gesture: cv2.putText(frame,text,(50,50), font, 0.8,(0,0,255),2)
                    # text = 'y_movement: ' + str(y_movement)
                    # if not gesture: cv2.putText(frame,text,(50,100), font, 0.8,(0,0,255),2)

                    gesture_detection_msg = GestureRecognitionResult()
                    gesture_detection_msg.message_type = GestureRecognitionResult.RESULT

                    if x_movement > gesture_threshold:
                        gesture.append("Shaking head")
                        print("the gesture is : Shaking head")
                        # gesture_detection_msg.gestures = 'Shaking head'
                    if y_movement > gesture_threshold:
                        gesture.append("Nodding")
                        print("the gesture is : Nodding")
                        # gesture_detection_msg.gestures = 'Nodding'
                    if gesture_show == 0:
                        gesture = False
                        x_movement = 0
                        y_movement = 0
                        gesture_show = 120  # number of frames a gesture is shown
                    # if x_movement < gesture_threshold and y_movement < gesture_threshold:
                    #     print("no head gesture detected")

                    # print("the gesture is ", gesture)
                    # publish message
                    # rospy.loginfo("Publishing result to referee...")
                    # if gesture == False:
                    #     hand_gesture_show = self.hand_gesture_recognition()

                    # print("the gesture is : ", gesture)
                    p0 = p1

                    # cv2.imshow('image',frame)
                    # # out.write(frame)
                    # cv2.waitKey(1)

                    # cv2.destroyAllWindows()
                    # cap.release()
                    # print(" gesture detected ", gesture)

                else:
                    # print("the gesture for head gesture recognition", gesture)
                    break

            if len(gesture) > 0:
                print(" the final head gesture is... ", gesture)
                # print(" the final gesture for head gesture recognition is ", gesture[0])
                # gesture_detection_msg.gestures = gesture
                gesture_detection_msg.gestures = gesture
                self.output_bb_pub.publish(gesture_detection_msg)
                print("gesture published")
                self.gesture_result = True

                # ack_msg = String()
                # ack_msg.data = "success"
                # self.gesture_recognition_ack_pub.publish(ack_msg)

                self.stop_sub_flag = False
                self.image_queue = []
                print("setting image queue to zero")
                print("[gesture recognition] gesture result is ",
                      self.gesture_result)
            else:
                self.gesture_result = False
                # ack_msg = String()
                # ack_msg.data = "fail"
                # self.gesture_recognition_ack_pub.publish(ack_msg)
                print("no head gesture detected")
                print("[gesture recognition] gesture result is ",
                      self.gesture_result)
                gesture_detection_msg.gestures = ["None"]
                self.output_bb_pub.publish(gesture_detection_msg)
                # self.stop_sub_flag = False
                # self.image_queue = []

        else:
            print("no face detected")
            self.gesture_result = False
            # ack_msg = String()
            # ack_msg.data = "fail"
            # self.gesture_recognition_ack_pub.publish(ack_msg)
            gesture_detection_msg = GestureRecognitionResult()
            gesture_detection_msg.message_type = GestureRecognitionResult.RESULT
            gesture_detection_msg.gestures = ["None"]
            self.output_bb_pub.publish(gesture_detection_msg)
            print("[gesture recognition] gesture result is ",
                  self.gesture_result)


if __name__ == "__main__":
    rospy.init_node("gesture_recognition_node")
    gesture_recognition_obj = Gesture_recognition()

    rospy.spin()
