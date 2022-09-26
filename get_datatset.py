import cv2
import mediapipe as mp
import time
import numpy as np
import os
from keras_facenet import FaceNet

cap = cv2.VideoCapture(0)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
frame_count = 0
label_name = "Jarunyawat"
face_recognition = FaceNet()
data = dict()
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    face_appear_current = list()
    start = time.time()
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        #mp_drawing.draw_detection(image, detection)
        bBox = detection.location_data.relative_bounding_box
        label_id = detection.label_id
        h ,w ,c = image.shape
        boundBox = [int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)]
        rgb_img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        crop_img = rgb_img[boundBox[1]:boundBox[1]+boundBox[3],boundBox[0]:boundBox[0]+boundBox[2]]
        crop_img = cv2.resize(crop_img,(160,160))
        expand_crop_img = np.expand_dims(crop_img,axis=0)
        cv2.rectangle(image,(boundBox[0], boundBox[1]), (boundBox[0]+boundBox[3], boundBox[1]+boundBox[2]), (0, 0, 255), 1)
        embeddings = face_recognition.embeddings(expand_crop_img)
        # Writing to sample.json
        np.save(os.path.join("python","face_recognition","image","Jarunyawat",f"{frame_count}"),embeddings)
        frame_count += 1
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image,1))
    end = time.time()
    total = end - start
    fps = 1 / total
    print(f"{fps} FPS")
    if frame_count >= 50:
        break
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()