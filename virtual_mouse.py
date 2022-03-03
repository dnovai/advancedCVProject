import cv2
import numpy as np
import advancedcv.hand_tracking as htm
import time
import autopy

w_cam, h_cam = 640, 480
frame_reduction = 100
smoothening = 10

p_location_x, p_location_y = 0, 0
c_location_x, c_location_y = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.HandDetector(max_hands=1)

p_time = 0

w_screen, h_screen = autopy.screen.size()
print(w_screen, h_screen)

while True:
    # Find hand landmarks (left hand)
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    lm_list, bbox = detector.get_position(img)
    
    # Get the tip of index of middle fingers
    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
    
        # Check which fingers are up
        fingers = detector.fingers_up()
        cv2.rectangle(img, (frame_reduction, frame_reduction),
                      (w_cam - frame_reduction, h_cam - frame_reduction), (255, 0, 255), 2)

        # Only index finger: Moving mode
        if fingers[1] == 1 and fingers[2] == 0:

            # Convert coordinates
            x3 = np.interp(x1, (frame_reduction, w_cam - frame_reduction), (0, w_screen))
            y3 = np.interp(y1, (frame_reduction, h_cam - frame_reduction), (0, h_screen))
    
            # Smoothen Values
            c_location_x = p_location_x + (x3 - p_location_x) / smoothening
            c_location_y = p_location_y + (y3 - p_location_y) / smoothening
    
            # Move Mouse
            autopy.mouse.move(x3, y3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

            p_location_x, p_location_y = c_location_x, c_location_y
    
        # Both index and middle fingers are up: clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between fingers
            length, img, d_info = detector.get_distance(8, 12, img, radius=10)
            print(length)

            # Click mouse if distance short
            if length < 40:

                cv2.circle(img, (d_info[4], d_info[5]), 10, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
    
    # Frame rate
    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    
    cv2.putText(img, f'FPS: {str(int(fps))}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)