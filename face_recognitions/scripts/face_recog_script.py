#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import numpy as np
from ament_index_python.packages import get_package_share_directory
from face_recognitions_interface.srv import SetEnb
from face_recognitions_interface.action import Recognition

class FaceRecognitionInterface(Node):
    def __init__(self):
        super().__init__('face_recognition')
        self._action_client = ActionClient(self, Recognition, 'recognition')

    def send_goal(self):
        goal_msg = Recognition.Goal()

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

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
        self.get_logger().info("Face regnition deactivated")

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"{feedback}")
        
        
def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionInterface()
    node.send_goal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()