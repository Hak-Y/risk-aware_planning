#!/usr/bin/env python

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
from tf.transformations import quaternion_matrix, quaternion_from_matrix

def odom_callback(msg):
       # Define the transformation matrix for the received odom message
    original_position = np.array([
        [msg.pose.pose.position.x],
        [msg.pose.pose.position.y],
        [msg.pose.pose.position.z],
        [1.0]  # Homogeneous coordinate
    ])
    original_orientation = np.array([
        [msg.pose.pose.orientation.x],
        [msg.pose.pose.orientation.y],
        [msg.pose.pose.orientation.z],
        [msg.pose.pose.orientation.w]
    ])
    original_rotation_matrix = quaternion_matrix(original_orientation.flatten())

    received_transformation_matrix = np.eye(4)
    received_transformation_matrix[:3, :3] = original_rotation_matrix[:3, :3]
    received_transformation_matrix[:3, 3] = original_position[:3, 0]
    # Define the transformation matrix for x+0.5, z+0.1
    translation_matrix = np.eye(4)
    translation_matrix[0, 3] = 0.5
    translation_matrix[2, 3] = 0.1

    # Perform matrix multiplication
    transformed_matrix = np.matmul(received_transformation_matrix,translation_matrix)

    # Create a Pose message with the transformed coordinates and orientation
    pose_msg = PoseStamped()
    pose_msg.header = msg.header
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.pose.position.x = transformed_matrix[0, 3]
    pose_msg.pose.position.y = transformed_matrix[1, 3]
    pose_msg.pose.position.z = transformed_matrix[2, 3]
    transformed_orientation = quaternion_from_matrix(transformed_matrix)
    pose_msg.pose.orientation.x = transformed_orientation[0]
    pose_msg.pose.orientation.y = transformed_orientation[1]
    pose_msg.pose.orientation.z = transformed_orientation[2]
    pose_msg.pose.orientation.w = transformed_orientation[3]

    # Publish the transformed pose
    pose_publisher.publish(pose_msg)

if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('odom_to_pose_publisher', anonymous=True)

    # Subscriber for the odometry topic
    rospy.Subscriber('odom', Odometry, odom_callback)

    # Publisher for the pose topic
    pose_publisher = rospy.Publisher('transformed_pose', PoseStamped, queue_size=10)

    # Spin to keep the script alive
    rospy.spin()