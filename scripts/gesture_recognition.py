#!/usr/bin/env python3

import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge, CvBridgeError
from metrics_refbox_msgs.msg import Command, GestureRecognitionResult

from std_msgs.msg import String
from math import dist
import rospy

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('nodcontrol.avi',fourcc, 20.0, (640,480))


class Gesture_recognition():
    def __init__(self) -> None:
        import mediapipe as mp
        rospy.loginfo("Gesture recognition node is ready...")
        self.cv_bridge = CvBridge()
        self.image_queue = None
        print("setting image queue to zero 3")
        self.clip_size = 100  # manual number
        self.stop_sub_flag = False
        self.cnt = 0
        self.gesture_result = None
        self.width = 1280
        self.height = 720
        self.image_sub = None
        self.move_front_flag = False
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.gestures = []
        self.x_coordinate = []
        self.y_coordinate = []
        self.z_coordinate = []
        self.gesture_detection_msg = GestureRecognitionResult()
        self.gesture_detection_msg.message_type = GestureRecognitionResult.RESULT
        self.function_start = None
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
                "\n[Gesture recognition] Start command received from refree box")

            # # set of the HSR camera to get front straight view
            # if not self.move_front_flag:
            #     self._hsr_head_controller('front')

            # start subscriber for image topic
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
        print(
                "\n[Gesture recognition] cv_bridge function called")
        head_gesture_show, hand_gesture_show = [],[]
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
                    rospy.loginfo("\n[Gesture recognition] Image received...")

                    self.stop_sub_flag = True

                    # pop the first element
                    self.image_queue.pop(0)
                    # length = len(self.image_queue)
                    # print(" original length of the queue is ", length)
                    # self.image_queue = self.image_queue[0:length: 3]

                    # deregister subscriber
                    self.image_sub.unregister()

                    # call object inference method
                    print("\n[Gesture recognition] converted to ros image")
                    print("\n[Gesture recognition]  the length of the image_queue before head is : ", len(self.image_queue))
                    head_gesture_show = self.head_gesture_recognition()

                    print("\n[Gesture recognition]  the length of the image_queue before hand is : ", len(self.image_queue))
                    if head_gesture_show == []:
                        print(" going for hand gesture", head_gesture_show)
                        hand_gesture_show = self.hand_gesture_recognition()
                    # print(" the values of hand and head gestures ", head_gesture_show, hand_gesture_show)

                    if head_gesture_show == [] and hand_gesture_show == []:
                        gesture_detection_msg = GestureRecognitionResult()
                        gesture_detection_msg.message_type = GestureRecognitionResult.RESULT
                        gesture_detection_msg.gestures = ["None"]
                        self.output_bb_pub.publish(gesture_detection_msg)
                        print("\n[Gesture recognition] Nothing detected gesture published")
                        self.stop_sub_flag = False
                        self.image_queue = [] 
                        # self.gesture_result = True



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
        print("\n[Head Gesture recognition]  function started")
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
                # print("[head gesture recognition] Face found: ", face_found)
                # cv2.imshow('image',frame)
                # out.write(frame)
                # cv2.waitKey(1)
            frame_num += 1

        if face_found:
            face_center = x+w/2, y+h/3
            p0 = np.array([[face_center]], np.float32)
            print("\n[Head Gesture recognition] Face detected, the face center is ", face_center)

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
                        print("\n[Head Gesture recognition] the gesture is : Shaking head")
                        # gesture_detection_msg.gestures = 'Shaking head'
                    if y_movement > gesture_threshold:
                        gesture.append("Nodding")
                        print("\n[Head Gesture recognition] the gesture is : Nodding")
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
                print("\n[Head Gesture recognition] the final head gesture is... ", gesture)
                # print(" the final gesture for head gesture recognition is ", gesture[0])
                # gesture_detection_msg.gestures = gesture
                gesture_detection_msg.gestures = gesture
                self.output_bb_pub.publish(gesture_detection_msg)
                print("\n[Head Gesture recognition] gesture published")
                self.gesture_result = True

                # ack_msg = String()
                # ack_msg.data = "success"
                # self.gesture_recognition_ack_pub.publish(ack_msg)

                self.stop_sub_flag = False
                self.image_queue = []
                print("setting image queue to zero")
                print("[gesture recognition] gesture result is ",
                      self.gesture_result)
                self.gestures = []
                return gesture[0]

            else:
                # self.gesture_result = False
                # ack_msg = String()
                # ack_msg.data = "fail"
                # self.gesture_recognition_ack_pub.publish(ack_msg)
                # print("no head gesture detected")
                # print("[gesture recognition] gesture result is ",
                #       self.gesture_result)
                # gesture_detection_msg.gestures = ["None"]
                # self.output_bb_pub.publish(gesture_detection_msg)
                print("\n[Head Gesture recognition] no head gesture detected", gesture)
                self.stop_sub_flag = False
                # self.image_queue = []
                return self.gestures

        else:
            print("\n[Head Gesture recognition] no face detected")
            # self.gesture_result = False
            # ack_msg = String()
            # ack_msg.data = "fail"
            # self.gesture_recognition_ack_pub.publish(ack_msg)
            # gesture_detection_msg = GestureRecognitionResult()
            # gesture_detection_msg.message_type = GestureRecognitionResult.RESULT
            # gesture_detection_msg.gestures = ["None"]
            # self.output_bb_pub.publish(gesture_detection_msg)
            # print("[gesture recognition] gesture result is ",
            #       self.gesture_result)
            self.stop_sub_flag = False
            # self.image_queue = []
            return self.gestures

    
    def check_movement(self, x_coordinate, y_coordinate):
        counter = 0
        gesture_detected = None
        leng = len(x_coordinate)

        if abs(x_coordinate[0] - x_coordinate[99]) >= 0.01:
            print(" the coordinates are", x_coordinate[0], x_coordinate[leng-1])
            # TODO: break the loop after 10 counter

            counter = 1
        # print(" the coordinates are", x_coordinate[0], x_coordinate[100])
        print(" the coordinates are", x_coordinate[0], x_coordinate[leng-1])
        print(" the coordinates are ", x_coordinate)
        print(" counter value ", counter)
        if counter == 0:
            gesture_detected = "stop"
        else:
            gesture_detected = "waving"
        return gesture_detected

    def stop_sign(self):
        self.gestures = []
        # print(" [hand gesture] try to detect stop sign")
        # flag_gesture_detected = False
        # while flag_gesture_detected == False:
        landmarks_0 = self.hand_landmarks.landmark[0]
        landmarks_9 = self.hand_landmarks.landmark[9]
        landmarks_8 = self.hand_landmarks.landmark[8]
        coordinate_landmark_0 = [landmarks_0.x * self.width,
                                 landmarks_0.y * self.height, landmarks_0.z]
        coordinate_landmark_9 = [landmarks_9.x * self.width,
                                 landmarks_9.y * self.height, landmarks_9.z]
        coordinate_landmark_8 = [landmarks_8.x * self.width,
                                 landmarks_8.y * self.height, landmarks_8.z]
        x0 = coordinate_landmark_0[0]
        y0 = coordinate_landmark_0[1]
        x9 = coordinate_landmark_9[0]
        y9 = coordinate_landmark_9[1]
        x8 = coordinate_landmark_8[0]
        y8 = coordinate_landmark_8[1]
        z8 = coordinate_landmark_8[2]
        x_not_true_0 = landmarks_0.x
        y_not_true_0 = landmarks_0.y


        if abs(x9 - x0) < 0.05:  # since tan(0) --> ∞
            m = 1000000000
        else:
            m = abs((y9 - y0)/(x9 - x0))

        if m > 1:
            if y9 < y0:  # since, y decreases upwards
                self.x_coordinate.append(x_not_true_0)
                self.y_coordinate.append(y_not_true_0)
                self.z_coordinate.append(z8)
                # print("x bcdevn", self.x_coordinate)
                # print("y bcdevn", self.y_coordinate)
                # print("z bcdevn", self.z_coordinate)
                # print(" the length of ", len(self.x_coordinate))
                if len(self.x_coordinate) >= 100:
                    gesture_detected = self.check_movement(
                        self.x_coordinate, self.y_coordinate)
                    if gesture_detected != None:
                        # print(" the final hand gesture is ... ", gesture_detected)
                        # flag_gesture_detected = True
                        self.gestures.append(gesture_detected)
                        self.gesture_detection_msg.gestures = self.gestures
                        self.output_bb_pub.publish(
                            self.gesture_detection_msg)
                        print("[hand gesture recognition] the final gesture for hand gesture recognition is ", gesture_detected)
                        print("gesture published")
                        self.stop_sub_flag = False
                        # self.image_queue = []
                        # self.gestures = []
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
            return self.gestures

    def thumbs_sign(self):
        self.gestures = []
        # print(" try to detect thumbs sign")
    # is z="finger, it retuens which finger is closed. If z="true coordinate", it returns the true coordinates
        if self.hand_landmarks is not None:
            try:
                # coordinates of landmark 0
                p0x = self.hand_landmarks.landmark[0].x
                p0y = self.hand_landmarks.landmark[0].y
                # coordinates of tip index
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

                # print(self.hand_landmarks.landmark[4].y, self.hand_landmarks.landmark[5].y)
                # print(" trying to detect thumbs")
                # to make it more robust turn the and conditions to or
                if self.hand_landmarks.landmark[4].y < self.hand_landmarks.landmark[5].y:
                    # print(" first loop for thumbs up")
                    if d07 > d08 and d011 > d012 and d015 > d016 and d019 > d020:
                        # print(" second loop for thumbs up")
                        # gesture = "thumbs up"
                        self.gestures.append("thumbs up")
                if self.hand_landmarks.landmark[4].y > self.hand_landmarks.landmark[5].y:
                    # print(" first loop for thumbs down")
                    if d07 > d08 and d011 > d012 and d015 > d016 and d019 > d020:
                        # print(" second loop for thumbs down")
                        # gesture = "thumbs down"
                        self.gestures.append("thumbs down")

                if len(self.gestures) > 0:
                    # print("the final hand gesture detected is...", self.gestures)
                    self.gesture_detection_msg.gestures = self.gestures
                    self.output_bb_pub.publish(self.gesture_detection_msg)
                    # print(" the final gesture for hand gesture recognition is ", hand_gesture[0])
                    print("[hand gesture recognition] thumb gesture published")
                    self.stop_sub_flag = False
                    # self.image_queue = []
                    # self.gestures = []
                    return self.gestures

                else:
                    self.gestures = []
                    rospy.loginfo_once(
                        "[hand gesture recognition] no thumb gesture found 1")
                    return self.gestures

            except:
                pass

    def hand_gesture_recognition(self):
        hand_gesture = []

        with self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

            for number in range(1, len(self.image_queue)):
                print("working on hand gesture")
                while len(self.image_queue) > 0:
                    # print(" the image queue is ", number, len(self.image_queue))
                    # print(len(self.image_queue))
                    frame = self.image_queue[number]

                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    frame.flags.writeable = False
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame)

                    # Draw the hand annotations on the image.
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.height = frame.shape[0]
                    self.width = frame.shape[1]
                    if results.multi_hand_landmarks:
                        #   print(" the landmarks are : ", results.multi_hand_landmarks)
                        for self.hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(frame, self.hand_landmarks, self.mp_hands.HAND_CONNECTIONS, self.mp_drawing_styles.get_default_hand_landmarks_style(
                            ), self.mp_drawing_styles.get_default_hand_connections_style())
                            # pdb.set_trace()
                            # print(" individual land mark is ", hand_landmarks.landmark[0])

                            # print(self.stop_sign(hand_landmarks), self.thumbs_sign(hand_landmarks))
                            rospy.loginfo_once(
                                " lets detect the hand gestures")
                            # print(self.hand_landmarks)

                            print(" trying to detect thumb gesture")
                            self.gestures = self.thumbs_sign()

                            if self.gestures == []:
                                print(" trying to detect stop gesture")
                                # print("[hand gesture] no thumb gesture found 2")
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
                                print("setting image queue to zero 6")
                                got_gesture = self.gestures
                                self.gestures = []
                                self.stop_sub_flag = False
                                return got_gesture
                    else:
                        print("\n****************************")
                        print("\n[[hand gesture recognition] No Hand Detected")
                        print("\n****************************\n")

                        # gesture_detection_msg = GestureRecognitionResult()
                        # gesture_detection_msg.message_type = GestureRecognitionResult.RESULT
                        # gesture_detection_msg.gestures = ["None"]
                        # self.output_bb_pub.publish(gesture_detection_msg)

                        self.image_queue = []
                        print("setting image queue to zero 7")
                        self.gestures = []
                        self.stop_sub_flag = False
                        return self.gestures




if __name__ == "__main__":
    rospy.init_node("gesture_recognition_node")
    gesture_recognition_obj = Gesture_recognition()

    rospy.spin()
