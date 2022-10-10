#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2

class Camera(Node):
    def __init__(self):
        #ros initialization
        super().__init__('camera_node')
        self.fps = 100.0
        self.cap = cv2.VideoCapture(0)
        self.img_pub = self.create_publisher(Image,"image_pub",10)
        self.timer = self.create_timer(1/self.fps, self.timer_callback)
        self.br = CvBridge()
    
    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret == True:
            self.img_pub.publish(self.br.cv2_to_imgmsg(frame))
        
def main(args=None):
    rclpy.init(args=args)
    node = Camera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()