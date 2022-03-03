import cv2
import mediapipe as mp
import math


class HandDetector(object):

    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.model_complexity = model_complexity
        self.track_confidence = track_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity,
                                         self.detection_confidence, self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils

        self.lm_list = []
        self.results = None
        self.key_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def get_position(self, img, hand_number=0, draw=True):

        x_list = []
        y_list = []
        bbox = []
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]

            for id, lm in enumerate(my_hand.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                x_list.append(cx)
                y_list.append(cy)
                self.lm_list.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 15, cv2.FILLED)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            if draw:
                cv2.rectangle(img, (x_min - 5, y_min - 5), (x_max - 5, y_max - 5), (0, 255, 0), 2)

        return self.lm_list, bbox

    def fingers_up(self):
        fingers = []

        # Thumb
        if self.lm_list[self.key_ids[0]][1] < self.lm_list[self.key_ids[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for idx in range(1, len(self.key_ids)):
            if self.lm_list[self.key_ids[idx]][2] < self.lm_list[self.key_ids[idx]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def get_distance(self, p1, p2, img, draw=True, radius=15, thickness=3):

        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), thickness)
            cv2.circle(img, (x1, y1), radius, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), radius, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), radius, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
