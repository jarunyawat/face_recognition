#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import numpy as np
from face_recognitions_interface.srv import SetEnb
from face_recognitions_interface.action import Recognition
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Int8


class FaceRecognitionInterface(Node):
    def __init__(self):
        super().__init__('face_recognition')
        self._action_client = ActionClient(self, Recognition, 'recognition')
        self.img_sub = self.create_subscription(Image,"image_pub",self.listener_callback,10)
        self.status_pub = self.create_publisher(Int8,"/people_detection/status",10)
        self.enb_srv = self.create_service(SetEnb,"/people_detection/enable", self.enb_srv_callback)
        self.br = CvBridge()
        self.current_frame = None
        self.isEnb = False
        self.ready = True
        self.boundBox = []
        self.name = []

    def recognition_send_goal(self, img):
        goal_msg = Recognition.Goal()
        goal_msg.img = self.br.cv2_to_imgmsg(img)

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(goal_msg)

        self._send_goal_future.add_done_callback(self.recognition_goal_response_callback)

    def recognition_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.recognition_get_result_callback)

    def recognition_get_result_callback(self, future):
        self.ready = True
        result = future.result().result
        self.get_logger().info(f"{result.name}")
        self.boundBox = result.roi_box
        self.name = result.name
        self.get_logger().info("Result recieve")
    
    def listener_callback(self, data):
        self.current_frame = self.br.imgmsg_to_cv2(data)
        if self.current_frame is not None and self.isEnb and self.ready:
            self.ready = False
            self.recognition_send_goal(self.current_frame)
        flip_img = cv2.flip(self.current_frame, 1)
        if self.isEnb:
            for _ in zip(self.boundBox,self.name):
                cv2.putText(flip_img, _[1], (_[0].xmin, _[0].ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                cv2.rectangle(flip_img,(_[0].xmin, _[0].ymin), (_[0].xmin+_[0].width, _[0].ymin+_[0].height), (0, 0, 255), 1)
        cv2.imshow("camera", flip_img)
        cv2.waitKey(1)
        status_msg = Int8()
        #running
        status_msg.data = 1
        self.status_pub.publish(status_msg)
    
    def enb_srv_callback(self, request, response):
        self.isEnb = not self.isEnb
        self.boundBox = []
        self.name = []
        return response

        
        
def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionInterface()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()