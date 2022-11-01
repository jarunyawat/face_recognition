#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import pyrealsense2 as rs2
import cv2
import mediapipe as mp

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class RealSenseListener(Node):
    def __init__(self):
        super().__init__('coordinate_transform')
        self.bridge = CvBridge()
        self.img_sub = self.create_subscription(Image, '/camera/color/image_raw', self.imageCallback,10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.imageDepthCallback,10)
        self.camerainfo_sub = self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.imageDepthInfoCallback,10)
        self.intrinsics = None
        self.br = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        #mediapipe
        self.mp_pose = mp.solutions.pose
        self.lmPose  = self.mp_pose.PoseLandmark
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.point_x = 0
        self.point_y = 0
        self.timer = self.create_timer(1.0, self.on_timer)
        self.detect_people = False

    def imageCallback(self, data):
        cv_image = self.br.imgmsg_to_cv2(data)
        h, w = cv_image.shape[:2]
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        cv_image.flags.writeable = False
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(cv_image)

        # Draw the pose annotation on the image.
        cv_image.flags.writeable = True
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        # Flip the image horizontally for a selfie-view display.
        if results.pose_landmarks!=None:
            mid_x_shoulder = (results.pose_landmarks.landmark[self.lmPose.LEFT_SHOULDER].x + results.pose_landmarks.landmark[self.lmPose.RIGHT_SHOULDER].x)/2 * w
            mid_y_shoulder = (results.pose_landmarks.landmark[self.lmPose.LEFT_SHOULDER].y + results.pose_landmarks.landmark[self.lmPose.RIGHT_SHOULDER].y)/2 * h
            mid_x_hip = (results.pose_landmarks.landmark[self.lmPose.LEFT_HIP].x + results.pose_landmarks.landmark[self.lmPose.RIGHT_HIP].x)/2 * w
            mid_y_hip = (results.pose_landmarks.landmark[self.lmPose.LEFT_HIP].y + results.pose_landmarks.landmark[self.lmPose.RIGHT_HIP].y)/2 * h
            self.point_x = int((mid_x_shoulder + mid_x_hip)/2)
            if self.point_x > w-1:
                self.point_x = w-1
            elif self.point_x < 0:
                self.point_x = 0
            self.point_y = int((mid_y_shoulder + mid_y_hip)/2)
            if self.point_y > h-1:
                self.point_y = h-1
            elif self.point_y < 0:
                self.point_y = 0
            cv2.circle(cv_image, (self.point_x,self.point_y), 0, (0,0,255), 20)
            self.detect_people = True
        cv2.imshow('MediaPipe Pose', cv2.flip(cv_image, 1))
        cv2.waitKey(1)

    def imageDepthCallback(self, data):
        try:
            #get image from msg
            depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #mediapipe pose
            if self.intrinsics:
                if self.detect_people:
                    depth = depth_image[self.point_y, self.point_x]
                    XYZ = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [self.point_x, self.point_y], depth)
                    t = TransformStamped()
                    # Read message content and assign it to
                    # corresponding tf variables
                    t.header.stamp = self.get_clock().now().to_msg()
                    t.header.frame_id = 'camera_link'
                    t.child_frame_id = 'user'

                    # Turtle only exists in 2D, thus we get x and y translation
                    # coordinates from the message and set the z coordinate to 0
                    t.transform.translation.x = XYZ[2]
                    t.transform.translation.y = -XYZ[0]
                    t.transform.translation.z = 0.0

                    # Send the transformation
                    self.tf_broadcaster.sendTransform(t)

        except CvBridgeError as e:
            print(e)
            return

    def imageDepthInfoCallback(self, cameraInfo):
        try:
            # import pdb; pdb.set_trace()
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.k[2]
            self.intrinsics.ppy = cameraInfo.k[5]
            self.intrinsics.fx = cameraInfo.k[0]
            self.intrinsics.fy = cameraInfo.k[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.d]
        except CvBridgeError as e:
            print(e)
            return
    
    def on_timer(self):
        # Store frame names in variables that will be used to
        # compute transformations
        from_frame_rel = 'odom'
        to_frame_rel = 'user'

        if self.detect_people:
            try:
                t = self.tf_buffer.lookup_transform(
                    to_frame_rel,
                    from_frame_rel,
                    rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
                return
            self.get_logger().info(f"global coordinate X:{ t.transform.translation.x} Y:{ t.transform.translation.y} Z:{ t.transform.translation.z}")

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
