from logging import exception
import cv2
import mediapipe as mp
import time
import numpy as np
from keras_facenet import FaceNet
import os
import sys

cap = cv2.VideoCapture(0)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_recognition = FaceNet()
dataset = dict()
label_name = ["Jarunyawat"]
roi_box = list()
face_img = list()

for name in label_name:
  data = list()
  for _ in os.listdir(os.path.join("python", "face_recognition", "image", name)):
    data.append(np.load(os.path.join("python", "face_recognition", "image", name, _)))
  dataset[name] = np.array(data)

with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        start = time.time()
        success, image = cap.read()
        roi_box = list()
        face_img = list()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        flip_img = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(flip_img, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = face_detection.process(rgb_image)

        if results.detections:
            for detection in results.detections:
              #mp_drawing.draw_detection(image, detection)
              bBox = detection.location_data.relative_bounding_box
              label_id = detection.label_id
              h, w, c = image.shape
              boundBox = [int(bBox.xmin * w), int(bBox.ymin * h),
                          int(bBox.width * w), int(bBox.height * h)]
              crop_img = rgb_image[boundBox[1]:boundBox[1] +
                                  boundBox[3], boundBox[0]:boundBox[0]+boundBox[2]]
              try:
                crop_img = cv2.resize(crop_img, (160, 160))
                expand_crop_img = np.expand_dims(crop_img, axis=0)
                roi_box.append(boundBox)
                face_img.append(expand_crop_img)
              except:
                pass
              cv2.rectangle(flip_img,(boundBox[0], boundBox[1]), (boundBox[0]+boundBox[3], boundBox[1]+boundBox[2]), (0, 0, 255), 1)
            for _ in zip(roi_box,face_img):
              embeddings = face_recognition.embeddings(_[1])
              distant = np.average(np.linalg.norm(dataset["Jarunyawat"] - embeddings))
              if distant < 5.0:
                cv2.putText(flip_img, "jarunyawat", (
                    _[0][0], _[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
              else:
                cv2.putText(flip_img, "unknow", (_[0][0], _[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
              #print(distant)
              #print(_)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', flip_img)
        end = time.time()
        total = end - start
        fps = 1 / total
        #print(f"{fps} FPS")
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
sys.exit()