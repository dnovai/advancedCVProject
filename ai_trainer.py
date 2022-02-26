import cv2
import numpy as np
import time
import advancedcv.pose_estimation as pdm


cap = cv2.VideoCapture(0)
detector = pdm.PoseDetector()

p_time = 0
count = 0
direction = 0

while True:

    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    img = detector.find_pose(img, False)
    lm_list = detector.get_position(img, False)

    if len(lm_list) != 0:

        # # Right arm landmarks:
        # detector.get_angle(img, 12, 14, 16)

        # Left arm landmarks
        angle = detector.get_angle(img, 11, 13, 15)
        percentage = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (210, 310), (650, 100))

        # Check for the dumbbell curls
        color = (255, 0, 255)
        if percentage == 100:
            color = (0, 255, 0)
            if direction == 0:

                count += 0.5
                direction = 1

        if percentage == 0:
            color = (0, 255, 0)
            if direction == 1:
                count += 0.5
                direction = 0

        # Draw bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(percentage)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        # Draw curl counter
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 255), 25)

    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time

    cv2.putText(img, f'FPS:{str(int(fps))}', (45, 45), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
