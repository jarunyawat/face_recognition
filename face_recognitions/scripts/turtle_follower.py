from face_recognitions.base_controller import * 

class TurtleFollower(BaseController):
    def __init__(self):
        super().__init__('turtle_follower')
        self.goal_subscription = self.create_subscription(Pose,'/goal',self.goal_callback,10)
        self.isEnable = True

    def goal_callback(self,msg):
        self.goal = np.array([msg.x,msg.y])

    def arrival_callback(self):
        self.isEnable = False
        self.get_logger().info('Arrive!!')

    def departure_callback(self):
        self.isEnable = True
        self.get_logger().info('Start follow!!')

def main1(args=None):
    rclpy.init(args=args)
    turtle_follower = TurtleFollower()
    rclpy.spin(turtle_follower)
    turtle_follower.destroy_node()
    rclpy.shutdown()