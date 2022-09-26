from keras_facenet import FaceNet
import cv2
import numpy as np
import os
import mediapipe as mp

class FaceRecog(object):
    def __init__(self):
        self.embedder = FaceNet()

    def extract_face(image,required_size = (160,160)):
        #detection = embedder.extract(rgb_img, threshold=0.8)
        #roi = detection[0]["box"]
        rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        crop_img = rgb_img[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
        crop_img = cv2.resize(crop_img,required_size)
        expand_crop_img = np.expand_dims(crop_img,axis=0)
        return (expand_crop_img,roi)

    def load_dataset(self,directory):
        X, y = list(), list()
        # enumerate folders, on per class
        for subdir in os.listdir(directory):
            # path
            path = os.path.join(directory,subdir)
            # skip any files that might be in the dir
            if not os.isdir(path):
                continue
            # load all faces in the subdirectory
            faces = self.load_faces(path)
            # create labels
            labels = [subdir for _ in range(len(faces))]
            # summarize progress
            print(f">loaded {len(faces)} examples for class: {subdir}")
            # store
            X.extend(faces)
            y.extend(labels)
        return np.asarray(X), np.asarray(y)
    
    def load_faces(self,directory):
        faces = list()
        # enumerate files
        for filename in os.listdir(directory):
            # path
            path = os.path.join(directory,filename)
            # get face
            face = self.extract_face(path)
            # store
            faces.append(face)
        return faces

img = cv2.imread(r"./image/komodo_dragon.jpg")
face_recognition = FaceRecog()
[face, roi] = face_recognition.extract_face(img)
roi_img = img.copy()
roi_img = cv2.rectangle(roi_img, (roi[0],roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (0,0,255), 3)
embeddings = face_recognition.embedder.embeddings(face)
cv2.imshow("kaiw",roi_img)
print(embeddings)
cv2.waitKey(0)
cv2.destroyAllWindows()