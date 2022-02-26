import cv2
import numpy as np
import time
import os
import advancedcv.hand_tracking as htm


brush_thickness = 15
eraser_thickness = 100

folder_path = 'headers'
my_list = os.listdir(folder_path)
my_list.pop(0)
my_list.sort()

print(my_list)
overlay_list = []

for img_path in my_list:
    image = cv2.imread(f'{folder_path}/{img_path}')
    overlay_list.append(image)

header = overlay_list[0]
draw_color = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 540)

img_canvas = np.zeros((540, 960, 3), np.uint8)

detector = htm.HandDetector(detection_confidence=0.85)
xp, yp = 0, 0

while True:
    # Import the image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    print(np.shape(img), np.shape(img_canvas))

    # Find hand landmarks
    img = detector.find_hands(img)
    lm_list = detector.get_position(img, draw=False)

    if len(lm_list) != 0:

        # tip of index and middle fingers
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]

        # Check which fingers are up
        fingers = detector.fingers_up()

        # If selection mode - two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            if y1 < 175:
                if 200 < x1 < 250:
                    header = overlay_list[0]
                    draw_color = (255, 0, 255)
                elif 450 < x1 < 500:
                    header = overlay_list[1]
                    draw_color = (255, 0, 0)
                elif 650 < x1 < 700:
                    header = overlay_list[2]
                    draw_color = (0, 255, 0)
                elif 800 < x1 < 850:
                    header = overlay_list[3]
                    draw_color = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)

        # If drawing mode - index finger is up
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if draw_color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
            
            xp, yp = x1, y1
            
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inverse = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inverse = cv2.cvtColor(img_inverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inverse)
    img = cv2.bitwise_or(img, img_canvas)

    # Setting the header image
    img[0:110, 0:912] = header
    img = cv2.addWeighted(img, 0.5, img_canvas, 0.5, 0.8)
    cv2.imshow('Image', img)
   #cv2.imshow('Canvas', img_canvas)
    cv2.waitKey(1)


