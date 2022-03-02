import cv2
import numpy as np
import advancedcv.hand_tracking as htm
import time
import autopy

w_cam, h_cam = 640, 480

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
    img = detector.find_hands(img)
    lm_list = detector.get_position(img, draw=False)
    
    # Get the tip of index of middle fingers
    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
    
        # Check which fingers are up
        fingers = detector.fingers_up()

        # Only index finger: Moving mode
        if fingers[1] == 1 and fingers[2] == 0:

            # Convert coordinates
            x3 = np.interp(x1, (0, w_cam), (0, w_screen))
            y3 = np.interp(y1, (0, h_cam), (0, h_screen))
    
    # Smoothen Values
    
            # Move Mouse
            #autopy.alert.alert("Hello, world")
            autopy.mouse.move(x3, y3)
    
    # Both index and middle fingers are up: clicking mode
    
    # Find distance between fingers
    
    # Click mouse if distance short
    
    # Frame rate
    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    
    cv2.putText(img, f'FPS: {str(int(fps))}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)