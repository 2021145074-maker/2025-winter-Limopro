cmake_minimum_required(VERSION 3.0.2)
project(limo_mission)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  dynamic_reconfigure
)

# Dynamic reconfigure
generate_dynamic_reconfigure_options(
  cfg/LaneDetect.cfg
  cfg/TrafficLight.cfg
  cfg/Roundabout.cfg
  cfg/ObstacleAvoid.cfg
  cfg/Pedestrian.cfg
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs cv_bridge dynamic_reconfigure
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/lane_detect_node.py
  scripts/traffic_light_node.py
  scripts/roundabout_node.py
  scripts/obstacle_avoid_node.py
  scripts/pedestrian_node.py
  scripts/examples/01_color_filter.py
  scripts/examples/02_roi.py
  scripts/examples/03_lidar_distance.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
