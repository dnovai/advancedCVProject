import cv2
import advancedcv.hand_tracking as htm
import time

w_cam, h_cam = 640, 480

p_time = 0

cap = cv2.VideoCapture(0)
# cap.set(3, w_cam)
# cap.set(4, h_cam)

detector = htm.HandDetector()
hand_landmarks = [4, 8]
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.get_position(img, hand_landmarks=hand_landmarks)

    if len(lm_list) != 0:
        print(lm_list)

    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time

    cv2.putText(img, str(round(fps)), (40, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
