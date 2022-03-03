import cv2
import advancedcv.hand_tracking as htm
import time
import math
import numpy as np
import osascript


volume = 0
volume_bar = 400
p_time = 0

cap = cv2.VideoCapture(0)

detector = htm.HandDetector()
key_hand_landmarks = [4, 8]

osascript.osascript("set volume output volume "+str(volume))
min_volume, max_volume = 0, 100

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.get_position(img, draw=False)

    if len(lm_list) != 0:
        x1, y1 = lm_list[key_hand_landmarks[0]][1], lm_list[key_hand_landmarks[0]][2]
        x2, y2 = lm_list[key_hand_landmarks[1]][1], lm_list[key_hand_landmarks[1]][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y1-y2)

        # Hand range 30 - 300
        # Volume range 0 - 100
        volume = np.interp(length, [30, 250], [min_volume, max_volume])
        volume_bar = np.interp(length, [30, 250], [400, 150])
        osascript.osascript("set volume output volume "+str(volume))

        if length < 30:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volume_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{str(round(volume))}%', (50, 130), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time

    cv2.putText(img, str(round(fps)), (40, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
