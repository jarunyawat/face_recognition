#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import numpy as np
from ament_index_python.packages import get_package_share_directory
from face_recognitions_interface.srv import SetEnb
from face_recognitions_interface.action import Recognition
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class FaceRecognitionInterface(Node):
    def __init__(self):
        super().__init__('face_recognition')
        self._action_client = ActionClient(self, Recognition, 'recognition')
        self.img_sub = self.create_subscription(Image,"image_pub",self.listener_callback,10)
        self.enb_srv = self.create_service(SetEnb,"start_recognition", self.enb_srv_callback)
        self.br = CvBridge()
        self.current_frame = None
        self.isEnb = False
        self.ready = True

    def send_goal(self, img):
        goal_msg = Recognition.Goal()
        goal_msg.img = self.br.cv2_to_imgmsg(img)

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(goal_msg)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.ready = True
        result = future.result().result
        self.get_logger().info(f"{result.name}")
        self.get_logger().info("Result recieve")
    
    def listener_callback(self, data):
        self.current_frame = self.br.imgmsg_to_cv2(data)
        if self.current_frame is not None and self.isEnb and self.ready:
            self.ready = False
            self.send_goal(self.current_frame)
        flip_img = cv2.flip(self.current_frame, 1)
        cv2.imshow("camera", flip_img)
        cv2.waitKey(1)
    
    def enb_srv_callback(self, request, response):
        self.isEnb = not self.isEnb
        return response

        
        
def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionInterface()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()