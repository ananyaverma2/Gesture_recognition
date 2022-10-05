#!/usr/bin/env python3

import time
from tokenize import String
import cv2
import numpy as np
import pickle
from math import dist

import rospy
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge, CvBridgeError
from metrics_refbox_msgs.msg import Command, GestureRecognitionResult
import pdb


class Hand_Gesture_recognition():
    def __init__(self) -> None:
        import mediapipe as mp
        rospy.loginfo("Hand gesture recognition node is ready...")
        self.cv_bridge = CvBridge()
        self.image_queue = None
        print("setting image queue to zero 4")
        self.clip_size = 100  # manual number
        self.stop_sub_flag = False
        self.cnt = 0
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
        self.gesture_detection_msg = GestureRecognitionResult()
        self.gesture_detection_msg.message_type = GestureRecognitionResult.RESULT
        self.function_start = None

        # subscriber
        self.referee_command_sub = rospy.Subscriber(
            "/metrics_refbox_client/command", Command, self._referee_command_cb)

        # pdb.set_trace()
        # rospy.sleep(5)
        # subscriber
        # rospy.wait_for_message("/gesture_ack", String)
        # self.gesture_recognition_sub = rospy.Subscriber(
        #     "/gesture_ack", String, self._hand_gesture_callback)

        # publisher
        self.output_bb_hand_pub = rospy.Publisher(
            "/metrics_refbox_client/gesture_recognition_result", GestureRecognitionResult, queue_size=10)

    def _hand_gesture_callback(self, msg):
        print("hand gesture callback called")
        print(msg.data)

        if msg.data == "success":
            self.function_start = True
        elif msg.data == "fail":
            self.function_start = False
        else:
            rospy.logerr("Invalid message received")

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

            print(
                "\n [hand gesture recognition] start command received from refree box")

            # start subscriber for image topic
            self.image_sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw",
                                              Image,
                                              self._input_image_cb)

        # STOP command from referee
        if msg.command == 2:

            self.image_sub.unregister()
            self.stop_sub_flag = False
            rospy.loginfo(
                "Received stopped command from referee for hand gesture recognition")
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
                    print("setting image queue to zero 5")

                self.image_queue.append(cv_image)

                if len(self.image_queue) > self.clip_size:
                    rospy.loginfo(
                        "Image received for hand gesture recognition ..")

                    self.stop_sub_flag = True

                    # pop the first element
                    # self.image_queue.pop(0)

                    # deregister subscriber
                    self.image_sub.unregister()

                    # call object inference method
                    print("converted to ros image")
                    print(" lenth of image queue recorded",
                          len(self.image_queue))

                    # msg = rospy.wait_for_message("/gesture_ack", String)
                    # if msg.data == "success":
                    hand_gesture_show = self.hand_gesture_recognition()
                    # else:
                    #     self.image_queue = None
                    #     print("setting image queue to zero 7")
                    #     self.gestures = []
                    #     self.stop_sub_flag = False

        except CvBridgeError as e:
            rospy.logerr(
                "Could not convert ros sensor msgs Image to opencv Image.")
            rospy.logerr(str(e))
            return

    def check_movement(self, x_coordinate, y_coordinate):
        counter = 0
        gesture_detected = None
        for i in range(0, len(x_coordinate)-3):
            # print(i, i+3)
            if abs(x_coordinate[i] - x_coordinate[i+3]) <= 1 or abs(y_coordinate[i] - y_coordinate[i+3]) <= 1:
                # TODO: break the loop after 10 counter
                counter += 1

        if counter > 10:
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

        if abs(x9 - x0) < 0.05:  # since tan(0) --> ∞
            m = 1000000000
        else:
            m = abs((y9 - y0)/(x9 - x0))

        if m > 1:
            if y9 < y0:  # since, y decreases upwards
                self.x_coordinate.append(x8)
                self.y_coordinate.append(y8)
                # print(" the length of ", len(self.x_coordinate))
                if len(self.x_coordinate) >= 100:
                    gesture_detected = self.check_movement(
                        self.x_coordinate, self.y_coordinate)
                    if gesture_detected != None:
                        # print(" the final hand gesture is ... ", gesture_detected)
                        # flag_gesture_detected = True
                        self.gestures.append(gesture_detected)
                        self.gesture_detection_msg.gestures = self.gestures
                        self.output_bb_hand_pub.publish(
                            self.gesture_detection_msg)
                        # print(" the final gesture for hand gesture recognition is ", hand_gesture[0])
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
                    self.output_bb_hand_pub.publish(self.gesture_detection_msg)
                    # print(" the final gesture for hand gesture recognition is ", hand_gesture[0])
                    print("[hand gesture] thumb gesture published")
                    self.stop_sub_flag = False
                    # self.image_queue = []
                    # self.gestures = []
                    return self.gestures

                else:
                    self.gestures = []
                    rospy.loginfo_once(
                        "[hand gesture] no thumb gesture found 1")
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

                            final_hand_gesture = self.thumbs_sign()

                            if final_hand_gesture == []:
                                # print("[hand gesture] no thumb gesture found 2")
                                final_hand_gesture = self.stop_sign()

                            if final_hand_gesture == []:
                                rospy.loginfo_once(
                                    "[hand gesture] no hand gesture detected")
                                self.nothing_detected = True

                            # I detected something
                            else:
                                print("\n****************************")
                                print("\n[hand gesture] The hand gesture is ",
                                      final_hand_gesture)
                                print("\n****************************\n")
                                self.image_queue = None
                                print("setting image queue to zero 6")
                                self.gestures = []
                                self.stop_sub_flag = False
                                return None
                    else:
                        print("\n****************************")
                        print("\n[hand gesture] No Hand Detected")
                        print("\n****************************\n")

                        gesture_detection_msg = GestureRecognitionResult()
                        gesture_detection_msg.message_type = GestureRecognitionResult.RESULT
                        gesture_detection_msg.gestures = ["None"]
                        self.output_bb_hand_pub.publish(gesture_detection_msg)

                        self.image_queue = None
                        print("setting image queue to zero 7")
                        self.gestures = []
                        self.stop_sub_flag = False
                        return None

            if self.nothing_detected:
                gesture_detection_msg = GestureRecognitionResult()
                gesture_detection_msg.message_type = GestureRecognitionResult.RESULT
                gesture_detection_msg.gestures = ["None"]
                self.output_bb_hand_pub.publish(gesture_detection_msg)
            else:
                pass


if __name__ == "__main__":
    rospy.init_node("hand_gesture_recognition_node")
    hand_gesture_recognition_obj = Hand_Gesture_recognition()

    rospy.spin()
