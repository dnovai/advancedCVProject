import cv2
import mediapipe as mp


class FaceDetector(object):

    def __init__(self, min_detection_confidence=0.5, model_selection=0):

        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.faces = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.faces.FaceDetection(self.min_detection_confidence, self.model_selection)
        self.results = None

    def find_faces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        self.results = self.face_detection.process(img_rgb)
        bbox_list = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bounding_box = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bounding_box.xmin * iw), int(bounding_box.ymin * ih), \
                       int(bounding_box.width * iw), int(bounding_box.height * ih)
                bbox_list.append([id, bbox, detection.score])

                if draw:
                    img = self.fancy_draw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return img, bbox_list

    @staticmethod
    def fancy_draw(img, bbox, l=30, t=5, rt=1):
        x0, y0, w, h = bbox
        x1, y1 = x0+w, y0+h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left x, y
        cv2.line(img, (x0, y0), (x0+l, y0), (255, 0, 255), t)
        cv2.line(img, (x0, y0), (x0, y0+l), (255, 0, 255), t)

        # Top Right x1, y
        cv2.line(img, (x1, y0), (x1-l, y0), (255, 0, 255), t)
        cv2.line(img, (x1, y0), (x1, y0+l), (255, 0, 255), t)

        # Bottom Left x, y1
        cv2.line(img, (x0, y1), (x0+l, y1), (255, 0, 255), t)
        cv2.line(img, (x0, y1), (x0, y1-l), (255, 0, 255), t)

        # Bottom Right x1, y1
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 255), t)

        return img
