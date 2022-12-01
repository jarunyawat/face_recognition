# face_recognition
Package for face recognition and people perception for Cacao mobile robot paticipate in ROBOCUP
## People following
run this command in console
'''bash
ros2 launch face_recognitions people_detection.launch.py
'''
## Topic
### Publish
**/goal_update**
Publish people position in xy coordinate relative to map to bt naigator for following dynamic object<br/>
**/people_detection/status**
Publish status of the node (0=wait for command,1=running,2=succeed,-1=fail)<br/>
### Subscribe
**/depth_camera/image_raw**
Subscribe RGB image from D455<br/>
**/depth_camera/depth/image_raw**
Subscribe depth image from D455<br/>
**/depth_camera/depth/camera_info**
Subscribe camera info from D455<br/>
## Service
**/people_detection/enable**
Call when you want to enable publish tf tranfrom people position to rviz2<br/>
**/people_detection/arrival**
Call when you want to tell the robot that you arrive at destination and stop tracking<br/>
