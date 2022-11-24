#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int8
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class RealSenseListener(Node):
    def __init__(self):
        super().__init__('people_position_node')
        self.goal_updater = self.create_publisher(PoseStamped,'goal_update',10)
        self.status_sub = self.create_subscription(Int8,'/people_detection/status',self.status_callback,10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.on_timer)
        self.status = Int8()
        self.status.data = 0
    
    def status_callback(self, data:Int8):
        self.status.data = data.data

    def on_timer(self):
        # Store frame names in variables that will be used to
        # compute transformations
        from_frame_rel = 'odom'
        to_frame_rel = 'user'

        if self.status.data == 1:
            try:
                t = self.tf_buffer.lookup_transform(
                    to_frame_rel,
                    from_frame_rel,
                    rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
                return
            self.get_logger().info(f"global coordinate X:{ t.transform.translation.x} Y:{ t.transform.translation.y} Z:{ t.transform.translation.z}")
        self.status_pub.publish(self.status)

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
