#!/usr/bin/python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rs2_camera_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('realsense2_camera'),
                    'launch',
                    'rs_launch.py'
                ])
            ]),
            launch_arguments={
                'device_type': 'd455'
            }.items()
    )

    people_detection_interface = Node(
        package="face_recognitions",
        executable="people_detection_scripts.py",
        remappings=[
            ("depth_camera/image_raw", "camera/color/image_raw"),
            ("depth_camera/depth/image_raw","camera/depth/image_rect_raw"),
            ("depth_camera/depth/camera_info","camera/depth/camera_info"),
        ]
    )
    controller = Node(
        package="base_controller.py",
        executable="people_detection_scripts.py",
    )
    entity_to_run = [rs2_camera_node,people_detection_interface,controller]
    return LaunchDescription(entity_to_run)