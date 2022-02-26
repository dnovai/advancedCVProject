import cv2
import numpy as np
import time
import advancedcv.pose_estimation as pdm


cap = cv2.VideoCapture(0)

detector = pdm.PoseDetector()

while True:

    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    img = detector.find_pose(img, False)
    lm_list = detector.get_position(img, False)

    if len(lm_list) != 0:

        detector.get_angle(img, 12, 14, 16, True)

    cv2.imshow("Image", img)
    cv2.waitKey(1)