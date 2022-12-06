cmake_minimum_required(VERSION 3.5)
project(face_recognitions)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find 
find_package(ament_cmake REQUIRED)		# cpp (optional)
find_package(rclcpp REQUIRED)			# cpp (optional)
find_package(ament_cmake_python REQUIRED)   	# python (optional)
find_package(rclpy REQUIRED)                	# python (optional)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs)
find_package(std_msgs)
find_package(sensor_msgs)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/camera_script.py
  scripts/face_recognition_backend.py
  scripts/face_recognition_script.py
  scripts/people_detection_behavior.py
  scripts/people_detection_scripts.py
  scripts/people_detect_rootfile.py
  scripts/people_pos_rel_odom_scripts.py
  scripts/base_controller.py
  scripts/inverse_kinematics.py
  DESTINATION lib/${PROJECT_NAME}
)

################ INSTALL LAUNCH, ETC #################
install(DIRECTORY
  launch
  image
  face_recognitions
  # add directories here
  DESTINATION share/${PROJECT_NAME})

ament_package()