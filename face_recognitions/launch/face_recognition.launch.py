#!/usr/bin/python3
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    camera_node = Node(
        package="face_recognitions",
        executable="camera_script.py"
    )
    face_recog_interface = Node(
        package="face_recognitions",
        executable="face_recog_script.py"
    )
    face_recog_backend = Node(
        package="face_recognitions",
        executable="display_script.py"
    )
    entity_to_run = [camera_node, face_recog_backend, face_recog_interface]
    return LaunchDescription(entity_to_run)