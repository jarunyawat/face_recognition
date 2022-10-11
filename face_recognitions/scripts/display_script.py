#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
import cv2
import mediapipe as mp
import numpy as np
from keras_facenet import FaceNet
from ament_index_python.packages import get_package_share_directory
from face_recognitions_interface.action import Recognition
from cv_bridge import CvBridge
import os
from face_recognitions_interface.msg import Roibox

class FaceRecognitionBackend(Node):
    def __init__(self):
        super().__init__("face_recognition_backend")
        self.rate = self.create_rate(10)
        self._action_server = ActionServer(self, Recognition, 'recognition', self.recognition_execute_callback)
        self.br = CvBridge()
        #img processing initialization
        mp_face_detection = mp.solutions.face_detection
        self.face_recognition = FaceNet()
        self.dataset = dict()
        label_name = ["Jarunyawat"]
        self.face_detection = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5)
        #load dataset
        for name in label_name:
            data = list()
            for _ in os.listdir(os.path.join(get_package_share_directory("face_recognitions"),"image", name)):
                data.append(np.load(os.path.join(get_package_share_directory("face_recognitions"),"image", name, _)))
        self.dataset[name] = np.array(data)
        self.current_frame = None
    
    def recognition_execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        result = Recognition.Result()
        result.header.stamp = self.get_clock().now().to_msg()
        img = self.br.imgmsg_to_cv2(goal_handle.request.img)
        [result.roi_box, result.name] = self.detect(img)
        goal_handle.succeed()
        return result

    def detect(self, frame):
        name_list = list()
        current_frame = frame
        flip_img = cv2.flip(current_frame, 1)
        rgb_image = cv2.cvtColor(flip_img, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = self.face_detection.process(rgb_image)
        rgb_image.flags.writeable = False
        roi_box = []
        face_img = []
        if results.detections:
            for detection in results.detections:
                #mp_drawing.draw_detection(image, detection)
                bBox = detection.location_data.relative_bounding_box
                h, w, c = current_frame.shape
                boundBox = [int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)]
                crop_img = rgb_image[boundBox[1]:boundBox[1] + boundBox[3], boundBox[0]:boundBox[0]+boundBox[2]]
                try:
                    crop_img = cv2.resize(crop_img, (160, 160))
                    expand_crop_img = np.expand_dims(crop_img, axis=0)
                    boundBox_msg = Roibox()
                    boundBox_msg.xmin = boundBox[0]
                    boundBox_msg.ymin = boundBox[1]
                    boundBox_msg.width = boundBox[2]
                    boundBox_msg.height = boundBox[3]
                    roi_box.append(boundBox_msg)
                    face_img.append(expand_crop_img)
                except:
                    pass
            for _ in zip(roi_box, face_img):
                embeddings = self.face_recognition.embeddings(_[1])
                distant = np.average(np.linalg.norm(self.dataset["Jarunyawat"] - embeddings))
                if distant < 5.0:
                    name_list.append("Jarunyawat")
                else:
                    name_list.append("Unknown")
        self.get_logger().info(f"{name_list}")
        return [roi_box, name_list]
        
def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionBackend()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()