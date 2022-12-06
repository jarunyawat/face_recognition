#!/usr/bin/python3
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Int8

class BaseController(Node):
    def __init__(self,name):
        super().__init__(name) #define node name "controller"

        self.K = 0.5

        self.isEnable = False
        self.command_publisher = self.create_publisher(Twist,'/cmd_vel',10)
        self.goal = np.array([0.0,0.0])
        timer_period = 0.1
        self.timer = self.create_timer(timer_period,self.timer_callback) #one timer in class
        self.pose_subscription = self.create_subscription(PoseStamped,'/goal_update',self.goal_callback,10)
        self.status_subscription = self.create_subscription(Int8,'/people_detection/status',self.status_callback,10)
        self.status = 0
    
    def status_callback(self,msg:Int8):
        self.status = msg.data

    def goal_callback(self,msg:PoseStamped):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])

    def timer_callback(self):
        if self.status == 1:
            self.get_logger().info(f"{np.linalg.norm(self.goal)}")
            msg = self.control()
            self.command_publisher.publish(msg)
        else:
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            self.command_publisher.publish(stop_msg)
        
    def control(self):
        msg = Twist()
        dp = self.goal
        e = np.arctan2(dp[1], dp[0])
        # K = 3.0
        w = (self.K)*np.arctan2(np.sin(e),np.cos(e))

        if np.linalg.norm(dp) > 1.0:
            v = 1.0
        elif np.linalg.norm(dp) < 0.7:
            v = -1.0
        else:
            self.isEnable = False
            v = 0.0
        msg.linear.x = v
        msg.angular.z = w
        return msg

def main(args=None):
    rclpy.init(args=args)
    base_controllers = BaseController("base_controller")
    rclpy.spin(base_controllers)
    base_controllers.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()