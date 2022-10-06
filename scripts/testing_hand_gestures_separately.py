import cv2
import mediapipe as mp
import pdb
from math import dist

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands



def stop_sign(hand_landmarks, c): 
    landmarks_0 = hand_landmarks.landmark[0]
    landmarks_9 = hand_landmarks.landmark[9]
    coordinate_landmark_0 = [landmarks_0.x * width, landmarks_0.y * height, landmarks_0.z]
    coordinate_landmark_9 = [landmarks_9.x * width, landmarks_9.y * height, landmarks_9.z]
    x0 = coordinate_landmark_0[0]
    y0 = coordinate_landmark_0[1]
    z0 = coordinate_landmark_0[2]
    x9 = coordinate_landmark_9[0]
    y9 = coordinate_landmark_9[1]
    while c <= 300:
        print(x0,y0,z0*1000000000)
        c+=1
    
    # if abs(x9 - x0) < 0.05:      #since tan(0) --> ∞
    #     m = 1000000000
    # else:
    #     m = abs((y9 - y0)/(x9 - x0))       
        
    # # if m>=0 and m<=1:
    # #     if x9 > x0:
    # #         return "Right"
    # #     else:
    # #         return "Left"
    # if m>1:
    #     if y9 < y0:       #since, y decreases upwards
    #         return "stop"
    #     # else:
    #     #     return "Down"



def thumbs_sign(hand_landmarks):   
#is z="finger, it retuens which finger is closed. If z="true coordinate", it returns the true coordinates
    if hand_landmarks is not None:
        try:
            p0x = hand_landmarks.landmark[0].x #coordinates of landmark 0
            p0y = hand_landmarks.landmark[0].y
            p7x = hand_landmarks.landmark[7].x #coordinates of tip index
            p7y = hand_landmarks.landmark[7].y
            d07 = dist([p0x, p0y], [p7x, p7y])
          
            p8x = hand_landmarks.landmark[8].x #coordinates of mid index
            p8y = hand_landmarks.landmark[8].y
            d08 = dist([p0x, p0y], [p8x, p8y])
            p11x = hand_landmarks.landmark[11].x #coordinates of tip middlefinger
            p11y = hand_landmarks.landmark[11].y
            d011 = dist([p0x, p0y], [p11x, p11y])
            p12x = hand_landmarks.landmark[12].x #coordinates of mid index
            p12y = hand_landmarks.landmark[12].y                   
            d012 = dist([p0x, p0y], [p12x, p12y])
            p15x = hand_landmarks.landmark[15].x #coordinates of mid index
            p15y = hand_landmarks.landmark[15].y                  
            d015 = dist([p0x, p0y], [p15x, p15y])
            p16x = hand_landmarks.landmark[16].x #coordinates of tip middlefinger
            p16y = hand_landmarks.landmark[16].y
            d016 = dist([p0x, p0y], [p16x, p16y])
            p19x = hand_landmarks.landmark[19].x #coordinates of mid index
            p19y = hand_landmarks.landmark[19].y                   
            d019 = dist([p0x, p0y], [p19x, p19y])
            p20x = hand_landmarks.landmark[20].x #coordinates of mid index
            p20y = hand_landmarks.landmark[20].y               
            d020 = dist([p0x, p0y], [p20x, p20y])
  
            print(hand_landmarks.landmark[4].y, hand_landmarks.landmark[5].y)
            # to make it more robust turn the and conditions to or
            if hand_landmarks.landmark[4].y < hand_landmarks.landmark[5].y:
                # if orientation(hand_landmarks):
                if d07>d08 or d011>d012 or d015>d016 or d019>d020:
                    return "thumbs up"
            if hand_landmarks.landmark[4].y > hand_landmarks.landmark[5].y:
                # if orientation(hand_landmarks):
                if d07>d08 or d011>d012 or d015>d016 or d019>d020:
                    return "thumbs down"
                        
        except:
           pass


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    c = 0

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height = image.shape[0]
    width = image.shape[1]
    if results.multi_hand_landmarks:
    #   print(" the landmarks are : ", results.multi_hand_landmarks)
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
        # pdb.set_trace() 
        # print(" individual land mark is ", hand_landmarks.landmark[0])  

        print(stop_sign(hand_landmarks, c), thumbs_sign(hand_landmarks))

        # print(thumbs_up(hand_landmarks))

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xff ==ord('q'):
      break
cap.release()