# Risk-Aware Planning

## Introduction
Risk-Aware Planning is a planning tool developed by Lee Sanghun, a graduate student at UNIST HMC LAB. This tool focuses on risk-aware planning for autonomous systems like drones. It supports Ground_truth mode and Slam mode and has expanded its openness through integration with PX4 Autopilot.

## Features
- Supports GPS mode and Slam mode
- Integration with PX4 Autopilot
- Compatibility with AirSim environment in Unreal Engine
- Sensor settings customization through settings.json file

## Prerequisites
- Operating System: Ubuntu 20.04
- ROS version: Noetic
- Unreal Engine's AirSim environment
- Necessary additional software and libraries

## Installation
### Clone the Repository
`git clone --branch V1.0 --single-branch --recursive git@github.com:sanghun17/risk-aware_planning.git ./risk-aware_planning/src`\
`cd risk-aware_planning`\
`catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release`\
`catkin build`

### Setup AirSim
For installing the AirSim package, refer to [AirSim GitHub](https://microsoft.github.io/AirSim/airsim_ros_pkgs/).\
And risk-aware_planning uses self-modified AirSim_ros_pkg, so please change the file released from AirSim to AirSim_ros_pkg file in github in sanghun.\ https://github.com/sanghun17/airsim_ros_wrapper

### Additional Setup
Describe any additional setup steps here.

## Usage
### Remote Computer Setup
1. **PX4 install**: `git clone --branch v1.11.3 --single-branch --recursive https://github.com/PX4/PX4-Autopilot.git`\
2. **PX4 run**:`cd ~/PX4/PX4-Autopilot && sudo make px4_sitl_default none_iris`\
   then, the UE termirnal print and wait "Ground control connected over UDP."\
   and this terminal print and wait "starting GPS fusion"
3. **Running AirSim Environment**: `cd ~/Downloads/SEER_map3/LinuxNoEditor && ./SEER_map.sh -RenderOffscreen`\
    it should wait at "LogRenderer: Reallocating scene render targets to support 856x640 Format 10 NumSamples 1 (Frame:2)."\
    if it shoud end with "Good bye.", find the process id thought "nvtop" and kill the process throught "sudo kill -9 123456"
4. **Running AirSim Node**: `cd ~/AirSim/ros && source devel/setup.bash && roslaunch airsim_ros_pkgs airsim_node.launch`
5. **Running MAVROS**: `cd ~/risk-aware_planning && source devel/setup.bash && roslaunch mavros px4.launch`
6. **Running Camera Module**: `cd ~/risk-aware_planning && source devel/setup.bash && roslaunch exploration_manager risk-aware_planning1.launch`

### Local Computer Setup
1. **Running Rviz**: `cd ~/risk-aware_planning && source devel/setup.bash && roslaunch exploration_manager risk-aware_planning3.launch`
2. **Running QGroundControl**: `cd ~/Downloads && ./QGroundControl.AppImage`\
if QGC is already exist, find the process id thought "nvtop" and kill the process throught "sudo kill -9 123456"

### Localization method
There are two options for UAV localization.
#### Visual Localization mode
`cd ~/risk-aware_planning && source devel/setup.bash && roslaunch rtabmap_ros rtabmap_sh.launch`\
`rosrun topic_tools throttle messages /rtabmap/odom 30.0 /mavros/odometry/out`\
\
Change the PX4 parameter(EKF2_AID_MASK) to 24 through QGC for vision localization\

#### GPS localization mode
Change the PX4 parameter(EKF2_AID_MASK) to 1 through QGC for GPS localization.

#### GT pose publisher
Modify "pub_gtpose" function from camera_depth_publisher.py from SEER exploration_manager pacakage properly according to your localization mode.

#### For Planning
Take off to 2.5m through QGC. and change to position mode through QGC.\
\
`cd ~/risk-aware_planning && source devel/setup.bash && roslaunch active_3d_planning_app_reconstruction uncertainty_voxblox.launch`\
`cd ~/risk-aware_planning && source devel/setup.bash && roslaunch active_3d_planning_app_reconstruction uncertainty_planner.launch`\

#### rosservice call
`rosservice call /planner/planner_node/publish_pointclouds "{}"` # to see pointcloud(tsdf, uncertainty map) if planner\
`rosservice call /planner/planner_node/toggle_running "data: true"` # to start planning. u have to change offboard thorugh QGC immediatly after this command. \
`rosservice call /planner/planner_node/toggle_returning "data: true"` # to force return to base. ( UAV returns itself if exploration done without this command )

## Troubleshooting
If you encounter problems, refer to the build processes of [rtapmap](https://github.com/introlab/rtabmap/wiki/Installation), [voxblox_ground_truth](https://github.com/ethz-asl/voxblox_ground_truth), and [mav_voxblox_planning](https://github.com/ethz-asl/mav_voxblox_planning).
You should utilize wstool tool!!

Q.fatal error: lapacke.h: No such file or directory
A.sudo apt-get install liblapacke-dev

Q. 
A.

