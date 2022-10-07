# Gesture Recognition Benchmark

> This task was performed in [METRICS HEART-MET Competition](https://metricsproject.eu/healthcare/heart-met-competition/) at [Cobot Maker Space](https://cobotmakerspace.org/), University of Nottingham, United Kingdom.
>> This functionality benchmark assesses the robot’s capability of recognizing gestures performed by a human. The robot is placed in front of a human who performs a gesture. The robot needs to recognize the gesture being performed by the human.

## Project Description:

The gestures (dependent variable) will be chosen from a list consisting of:

1. **Head gestures:**
    - [x] Nodding
    - [x] Shaking head

2. **Hand gestures:**
    - [x] Stop sign
    - [ ] Pointing  
    - [x] Thumb down
    - [x] Thumbs up
    - [ ] Pulling hand in
    - [ ] Pushing hand out
    - [x] Waving

Note: In this repo you can find the code of the gestures ticked above

## Pre-requisites:

1. OpenCV `pip install opencv-python`
2. Mediapipe `pip install mediapipe`
3. Ubuntu (20.04
4. ROS Noetic
5. Camera (Intel RealSense camera) `sudo apt-get install ros-$ROS_DISTRO-realsense2-camera`

## How to run the code:

Run the following commands on different terminals:

Purpose | Command
| :---: | :---: 
communicate with the camera  | `roslaunch realsense2_camera rs_camera.launch`
visualize the camera topic `/camera/color/image_raw` | `rviz`
run the code | `roslaunch gesture_recognition_benchmark gesture_recognition_benchmark.launch`
run referee box | `roslaunch metrics_refbox_client metrics_refbox_client.launch`

## How to Use the Project:

1. Run the referee box and select **Gesture Recognition** task from the given set of tasks.
2. press the start command and do the gesture infront of the camera.
3. the result can be viewed on the referee box.
