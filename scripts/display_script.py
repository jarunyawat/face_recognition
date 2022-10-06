#!/usr/bin/python3

from sentinel_description.dummy_module import dummy_function, dummy_var
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import mediapipe as mp
import time
import numpy as np
from keras_facenet import FaceNet
import os

class Display(Node):
    def __init__(self):
        super().__init__('img_subscriber')
        self.img_sub = self.create_subscription(Image,"image_pub",self.listener_callback,10)
        self.br = CvBridge()
        #img processing initialization
        mp_face_detection = mp.solutions.face_detection
        self.face_recognition = FaceNet()
        self.dataset = dict()
        label_name = ["Jarunyawat"]
        self.roi_box = list()
        self.face_img = list()
        self.face_detection = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5)
        #load dataset
        for name in label_name:
            data = list()
            for _ in os.listdir(os.path.join("image", name)):
                data.append(np.load(os.path.join("image", name, _)))
        self.dataset[name] = np.array(data)
    
    def listener_callback(self, data):
        current_frame = self.br.imgmsg_to_cv2(data)
        flip_img = cv2.flip(current_frame, 1)
        rgb_image = cv2.cvtColor(flip_img, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = self.face_detection.process(rgb_image)
        rgb_image.flags.writeable = False
        results = self.face_detection.process(rgb_image)
        if results.detections:
            for detection in results.detections:
                #mp_drawing.draw_detection(image, detection)
                bBox = detection.location_data.relative_bounding_box
                label_id = detection.label_id
                h, w, c = current_frame.shape
                boundBox = [int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)]
                crop_img = rgb_image[boundBox[1]:boundBox[1] + boundBox[3], boundBox[0]:boundBox[0]+boundBox[2]]
                try:
                    crop_img = cv2.resize(crop_img, (160, 160))
                    expand_crop_img = np.expand_dims(crop_img, axis=0)
                    self.roi_box.append(boundBox)
                    self.face_img.append(expand_crop_img)
                except:
                    pass
                cv2.rectangle(flip_img,(boundBox[0], boundBox[1]), (boundBox[0]+boundBox[3], boundBox[1]+boundBox[2]), (0, 0, 255), 1)
            for _ in zip(self.roi_box, self.face_img):
                embeddings = self.face_recognition.embeddings(_[1])
                distant = np.average(np.linalg.norm(self.dataset["Jarunyawat"] - embeddings))
                if distant < 5.0:
                    cv2.putText(flip_img, "jarunyawat", (_[0][0], _[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(flip_img, "unknow", (_[0][0], _[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("camera", flip_img)
        cv2.waitKey(1)
        
def main(args=None):
    rclpy.init(args=args)
    node = Display()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()