#!/usr/bin/env python3

import cv2
import numpy as np
from std_msgs.msg import String
from math import dist
import time
import random

import rospy
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge, CvBridgeError
from metrics_refbox_msgs.msg import Command, GestureRecognitionResult


class Gesture_recognition():
    def __init__(self) -> None:
        import mediapipe as mp
        rospy.loginfo("Gesture recognition node is ready...")
        self.cv_bridge = CvBridge()
        self.image_queue = None
        self.clip_size = 100  # manual number
        self.stop_sub_flag = False
        self.gesture_result = None
        self.width = 1280
        self.height = 720
        self.image_sub = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.gestures = []
        self.x_coordinate_0 = []
        self.y_coordinate_0 = []
        self.z_coordinate_0 = []
        self.x_coordinate_9 = []
        self.y_coordinate_9 = []
        self.z_coordinate_9 = []
        self.gesture_detection_msg = GestureRecognitionResult()
        self.gesture_detection_msg.message_type = GestureRecognitionResult.RESULT

        # subscriber
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

            self.gesture_result = None

            print( "\n[Gesture recognition] Start command received from refree box")

            # start subscriber for image topic for intel real sense camera
            self.image_sub = rospy.Subscriber("/camera/color/image_raw",
                                              Image,
                                              self._input_image_cb)
            self.image_queue = []

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
        print("\n[Gesture recognition] cv_bridge function called")
        head_gesture_show, hand_gesture_show = [],[]
        start_time = time.time()
        get_frames = 0
        yes_count = 0
        try:
            if not self.stop_sub_flag :
                # convert ros image to opencv image
                if (time.time() - start_time) < 0.2 or 4 < (time.time() - start_time) < 3.2:
                    print(time.time() - start_time)
                    cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                    self.image_queue.append(cv_image)
                
                if self.image_queue is None:
                    self.image_queue = []
            
                if len(self.image_queue) > 100:
                    for image in self.image_queue:
                        cv2.imwrite("/home/ananya/Documents/B-it-bots/gesture_benchmark/gesture_reco_ws/frames/frame%d.jpg" % yes_count, image)
                        yes_count += 1
                    print("Clip size reached...")
                    rospy.loginfo("\n[Gesture recognition] Image received...")

                    self.stop_sub_flag = True
                    self.image_queue.pop(0)

                    # deregister subscriber
                    self.image_sub.unregister()

                    # call object inference method
                    print("\n[Gesture recognition] converted to ros image")
                    head_gesture_show = self.head_gesture_recognition()

                    if head_gesture_show == []:
                        # print(" going for hand gesture", head_gesture_show)
                        hand_gesture_show = self.hand_gesture_recognition()

                    if head_gesture_show == [] and hand_gesture_show == []:
                        gesture_detection_msg = GestureRecognitionResult()
                        gesture_detection_msg.message_type = GestureRecognitionResult.RESULT
                        gesture_detection_msg.gestures = ["None"]
                        self.output_bb_pub.publish(gesture_detection_msg)
                        print("\n****************************")
                        print("\n[Gesture recognition] Nothing detected gesture published")
                        print("\n****************************\n")
                        self.stop_sub_flag = False
                        self.image_queue = [] 

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
        self.gestures = []
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # path to face cascde
        face_cascade = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalface_alt.xml")
        # define movement threshodls
        gesture_threshold = 120

        # find the face in the image
        face_found = False
        frame_num = 0
        #trying to find face for 5 frames
        if not face_found and frame_num < 5:
            frame = self.image_queue[frame_num]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_found = True
            frame_num += 1

        if face_found:
            face_center = x+w/2, y+h/3
            p0 = np.array([[face_center]], np.float32)
            print("\n****************************")
            print("\n[Head Gesture recognition] Face detected, the face center is ", face_center)
            print("\n****************************\n")

            gesture = []
            x_movement = 0
            y_movement = 0
            gesture_show = 120  # number of frames a gesture is shown
            frame_counter = 0

            for number in range(1, len(self.image_queue)):
                if len(gesture) < 10:
                    frame = self.image_queue[number]
                    frame_counter += 1
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

                    gesture_detection_msg = GestureRecognitionResult()
                    gesture_detection_msg.message_type = GestureRecognitionResult.RESULT

                    if x_movement > gesture_threshold:
                        gesture.append("Shaking head")
                    if y_movement > gesture_threshold:
                        gesture.append("Nodding")
                    if gesture_show == 0:
                        gesture = False
                        x_movement = 0
                        y_movement = 0
                        gesture_show = 120  # number of frames a gesture is shown

                    p0 = p1

                else:
                    break

            if len(gesture) > 0:
                print("\n****************************")
                print("\n[Head Gesture recognition] the final head gesture is... ", gesture)
                print("\n****************************\n")
                gesture_detection_msg.gestures = gesture
                self.output_bb_pub.publish(gesture_detection_msg)
                print("\n[Head Gesture recognition] gesture published")
                self.gesture_result = True
                self.stop_sub_flag = False
                self.image_queue = []
                self.gestures = []
                return gesture[0]

            else:
                print("\n****************************")
                print("\n[Head Gesture recognition] no head gesture detected", gesture)
                print("\n****************************\n")
                self.stop_sub_flag = False
                return self.gestures

        else:
            print("\n****************************")
            print("\n[Head Gesture recognition] no face detected")
            print("\n****************************\n")
            self.stop_sub_flag = False
            return self.gestures

    
    def check_movement(self):
        counter = 0
        gesture_detected = None
        my_rounded_list = []
        my_rounded_list = [ round(elem, 1) for elem in self.x_coordinate_0 ]

        for i in range(0,len(my_rounded_list)-1):
            if abs(my_rounded_list[i] - my_rounded_list[i+1]) >= 2:
                counter += 1
        if counter > 70:
            gesture_detected = "waving"
        else:
            gesture_detected = "stop"
        return gesture_detected

    def stop_sign(self):
        self.gestures = []
        self.x_coordinate_0 = []
        self.y_coordinate_0 = []
        self.z_coordinate_0 = []
        self.x_coordinate_9 = []
        self.y_coordinate_9 = []
        self.z_coordinate_9 = []

        with self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

            for i in range(0,101):

                # create a folder with name frames and give its path here
                frame = cv2.imread("/home/ananya/Documents/B-it-bots/gesture_benchmark/gesture_reco_ws/frames/frame{}.jpg".format(i))
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame)

                if results.multi_hand_landmarks:
                    landmarks_0 = results.multi_hand_landmarks[0].landmark[0]
                    landmarks_9 = results.multi_hand_landmarks[0].landmark[9]
                    landmarks_8 = results.multi_hand_landmarks[0].landmark[8]
                    coordinate_landmark_0 = [landmarks_0.x * self.width,
                                            landmarks_0.y * self.height, landmarks_0.z]
                    coordinate_landmark_9 = [landmarks_9.x * self.width,
                                            landmarks_9.y * self.height, landmarks_9.z]
                    coordinate_landmark_8 = [landmarks_8.x * self.width,
                                            landmarks_8.y * self.height, landmarks_8.z]

                    x0 = coordinate_landmark_0[0]
                    y0 = coordinate_landmark_0[1]
                    z0 = coordinate_landmark_0[2]
                    x9 = coordinate_landmark_9[0]
                    y9 = coordinate_landmark_9[1]
                    z9 = coordinate_landmark_9[2]
                    x8 = coordinate_landmark_8[0]
                    y8 = coordinate_landmark_8[1]
                    z8 = coordinate_landmark_8[2]
                    self.x_coordinate_0.append(x0)
                    self.y_coordinate_0.append(y0)
                    self.z_coordinate_0.append(z0)
                    self.x_coordinate_9.append(x9)
                    self.y_coordinate_9.append(y9)
                    self.z_coordinate_9.append(z9)

                else:
                    continue

        if len(self.x_coordinate_0) >0:
            if abs(self.x_coordinate_9[0] - self.x_coordinate_0[0]) < 0.05:  # since tan(0) --> ∞
                m = 1000000000
            else:
                m = abs((self.y_coordinate_9[0] - self.y_coordinate_0[0])/(self.x_coordinate_9[0] - self.x_coordinate_0[0]))

            if m > 1:
                if self.y_coordinate_9[0] < self.y_coordinate_0[0]:  # since, y decreases upwards
                    gesture_detected = self.check_movement(self.x_coordinate_0, self.y_coordinate_0)

                    if gesture_detected != None:
                        self.gestures.append(gesture_detected)
                        self.gesture_detection_msg.gestures = self.gestures
                        self.output_bb_pub.publish(
                            self.gesture_detection_msg)

                        print("\n****************************")
                        print("[hand gesture recognition] the final hand gesture is ", gesture_detected)
                        print("\n****************************\n")
                        print("gesture published")
                        self.stop_sub_flag = False
                        return self.gestures
                    else:
                        print(
                            "[hand gesture recognition] received None from check movement")
                else:
                    rospy.loginfo_once(
                        "[hand gesture recognition] no hand detected")
                    return self.gestures
            else:
                return self.gestures

        else:
            print("no hand detected")
            return self.gestures

    def thumbs_sign(self):
        self.gestures = []
        if self.hand_landmarks is not None:
            try:
                # coordinates of landmark 0
                p0x = self.hand_landmarks.landmark[0].x
                p0y = self.hand_landmarks.landmark[0].y
                p7x = self.hand_landmarks.landmark[7].x
                p7y = self.hand_landmarks.landmark[7].y
                d07 = dist([p0x, p0y], [p7x, p7y])
                # coordinates of mid index
                p8x = self.hand_landmarks.landmark[8].x
                p8y = self.hand_landmarks.landmark[8].y
                d08 = dist([p0x, p0y], [p8x, p8y])
                # coordinates of tip middlefinger
                p11x = self.hand_landmarks.landmark[11].x
                p11y = self.hand_landmarks.landmark[11].y
                d011 = dist([p0x, p0y], [p11x, p11y])
                # coordinates of mid index
                p12x = self.hand_landmarks.landmark[12].x
                p12y = self.hand_landmarks.landmark[12].y
                d012 = dist([p0x, p0y], [p12x, p12y])
                # coordinates of mid index
                p15x = self.hand_landmarks.landmark[15].x
                p15y = self.hand_landmarks.landmark[15].y
                d015 = dist([p0x, p0y], [p15x, p15y])
                # coordinates of tip middlefinger
                p16x = self.hand_landmarks.landmark[16].x
                p16y = self.hand_landmarks.landmark[16].y
                d016 = dist([p0x, p0y], [p16x, p16y])
                # coordinates of mid index
                p19x = self.hand_landmarks.landmark[19].x
                p19y = self.hand_landmarks.landmark[19].y
                d019 = dist([p0x, p0y], [p19x, p19y])
                # coordinates of mid index
                p20x = self.hand_landmarks.landmark[20].x
                p20y = self.hand_landmarks.landmark[20].y
                d020 = dist([p0x, p0y], [p20x, p20y])

                if d07 > d08 and d011 > d012 and d015 > d016 and d019 > d020:
                    if self.hand_landmarks.landmark[4].y < self.hand_landmarks.landmark[5].y:
                        self.gestures.append("thumbs up")
                if self.hand_landmarks.landmark[4].y > self.hand_landmarks.landmark[5].y:
                    if d07 > d08 and d011 > d012 and d015 > d016 and d019 > d020:
                        self.gestures.append("thumbs down")

                if len(self.gestures) > 0:
                    self.gesture_detection_msg.gestures = self.gestures
                    self.output_bb_pub.publish(self.gesture_detection_msg)
                    print("\n****************************")
                    print("[hand gesture recognition] the thumb gesture is : ", self.gestures)
                    print("\n****************************\n")
                    self.stop_sub_flag = False
                    return self.gestures

                else:
                    self.gestures = []
                    return self.gestures

            except:
                pass

    def hand_gesture_recognition(self):
        with self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

            for number in range(1, len(self.image_queue)):
                print("working on hand gesture")
                while len(self.image_queue) > 0:
                    frame = self.image_queue[number]
                    frame.flags.writeable = False
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame)

                    # Draw the hand annotations on the image.
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.height = frame.shape[0]
                    self.width = frame.shape[1]
                    if results.multi_hand_landmarks:
                        for self.hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(frame, self.hand_landmarks, self.mp_hands.HAND_CONNECTIONS, self.mp_drawing_styles.get_default_hand_landmarks_style(
                            ), self.mp_drawing_styles.get_default_hand_connections_style())
                            rospy.loginfo_once(
                                " lets detect the hand gestures")
                            self.gestures = self.thumbs_sign()

                            if self.gestures == []:
                                self.gestures = self.stop_sign()

                            if self.gestures == []:
                                rospy.loginfo_once(
                                    "[hand gesture recognition] no hand gesture detected")
                                self.nothing_detected = True

                            # I detected something
                            else:
                                print("\n****************************")
                                print("\n[hand gesture recognition] The hand gesture is ",
                                      self.gestures)
                                print("\n****************************\n")
                                self.image_queue = []
                                got_gesture = self.gestures
                                self.gestures = []
                                self.stop_sub_flag = False
                                return got_gesture
                    else:
                        print("\n****************************")
                        print("\n[[hand gesture recognition] No Hand Detected")
                        print("\n****************************\n")

                        self.image_queue = []
                        self.gestures = []
                        self.stop_sub_flag = False
                        return self.gestures

if __name__ == "__main__":
    rospy.init_node("gesture_recognition_node")
    gesture_recognition_obj = Gesture_recognition()

    rospy.spin()
