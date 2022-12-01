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
Publish people position in xy coordinate relative to map to bt naigator for following dynamic object
**/people_detection/status**
Publish status of the node (0=wait for command,1=running,2=succeed,-1=fail)
### Subscribe
## Service
