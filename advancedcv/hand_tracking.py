import cv2
import mediapipe as mp


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
        self.results = None

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def get_position(self, img, hand_number=0, draw=True, hand_landmarks=None):

        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]

            for id, lm in enumerate(my_hand.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)

                if draw:
                    if hand_landmarks is None:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                        lm_list.append([id, cx, cy])
                    else:
                        if id in hand_landmarks:
                            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                            lm_list.append([id, cx, cy])
        return lm_list
