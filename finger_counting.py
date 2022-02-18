import cv2
import os
import time
import advancedcv.hand_tracking as htm

p_time = 0
cap = cv2.VideoCapture(0)

folder_path = "finger_images"
my_list = os.listdir(folder_path)
print(my_list.sort())
print(my_list)
overlay_list = []

detector = htm.HandDetector()

for im_path in my_list:
    image = cv2.imread(f'{folder_path}/{im_path}')
    print(f'{folder_path}/{im_path}')
    overlay_list.append(image)

key_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.get_position(img, hand_number=0, draw=False)

    if len(lm_list) != 0:
        fingers = []

        # Thumb
        if lm_list[key_ids[0]][1] > lm_list[key_ids[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for id in range(1, len(key_ids)):
            if lm_list[key_ids[id]][2] < lm_list[key_ids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)

        print(total_fingers, overlay_list[total_fingers+1])
        h, w, c = overlay_list[total_fingers+1].shape
        img[0:h, 0:w] = overlay_list[total_fingers+1]
    
    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time
    
    cv2.putText(img, f'FPS: {str(round(fps))}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    