#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import mediapipe as mp
import time
import numpy as np
from keras_facenet import FaceNet
import os
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from face_recognitions_interface.srv import SetEnb
from face_recognitions_interface.action import Recognition

class FaceRecognitionInterface(Node):
    def __init__(self, action_node):
        super().__init__('img_subscriber')
        self.enb_srv = self.create_service(SetEnb,"set_enb",self.enb_callback)
        self.img_sub = self.create_subscription(Image,"image_pub",self.listener_callback,10)
        self.timer = self.create_timer(1/10,self.img_update)
        self.action_node = action_node
        self.br = CvBridge()
        self.current_frame = None
    
    def listener_callback(self, data):
        current_frame = self.br.imgmsg_to_cv2(data)
        self.current_frame = current_frame
        flip_img = cv2.flip(current_frame, 1)
        cv2.imshow("camera", flip_img)
        cv2.waitKey(1)
    
    def enb_callback(self,request,response):
        self.action_node.isEnb = False
        return response
    
    def img_update(self):
        self.action_node.current_frame = self.current_frame


class FaceRecognition(Node):
    def __init__(self):
        super().__init__("face_recognition_backend")
        self.rate = self.create_rate(10)
        self._action_server = ActionServer(
            self,
            Recognition,
            'recognition',
            self.execute_callback)
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
            for _ in os.listdir(os.path.join(get_package_share_directory("face_recognitions"),"image", name)):
                data.append(np.load(os.path.join(get_package_share_directory("face_recognitions"),"image", name, _)))
        self.dataset[name] = np.array(data)
        self.current_frame = None
        self.isEnb = False
    
    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        feedback_msg = Recognition.Feedback()
        while self.isEnb:
            feedback_msg.header.stamp = self.get_clock().now().to_msg()
            feedback_msg.name = ["hello"]
            goal_handle.publish_feedback(feedback_msg)
            self.rate.sleep()
        goal_handle.succeed()
        result = Recognition.Result()
        return result

    def detect(self):
        name_list = []
        if self.current_frame is None:
            return
        current_frame = self.current_frame
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
                    name_list.append(str("Jarunyawat"))
        return name_list
        
def main(args=None):
    rclpy.init(args=args)
    try:
        action_server = FaceRecognition()
        interface = FaceRecognitionInterface(action_server)
        # create executor
        executor = MultiThreadedExecutor(num_threads=4)
        # add nodes to the executor
        executor.add_node(action_server)
        executor.add_node(interface)
        try:
            # spin both nodes in the executor
            executor.spin()
        finally:
            executor.shutdown()
            action_server.destroy_node()
            interface.destroy_node()
    finally:
        rclpy.shutdown()

if __name__=='__main__':
    main()