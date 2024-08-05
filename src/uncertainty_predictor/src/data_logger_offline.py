import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import threading
import datetime
import numpy as np
import os
import struct
import ctypes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rosbag
from tqdm import tqdm

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data_gps/dataset2')
print("saving data at ",data_dir)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
last_save_time = datetime.datetime.now()
saved_data_cnt = 1
save_interval = 0.2  # Minimum time interval between saves (in seconds)

# Define the path to the data directory relative to the current Python file
class DataLogger:
    def __init__(self):
        print("DataLogger")

    def pointcloud_callback( msg1, msg2, msg3):
            position = msg2.pose.pose.position
            orientation = msg2.pose.pose.orientation
            odom1_data = np.hstack([position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w])

            position = msg3.pose.pose.position
            orientation = msg3.pose.pose.orientation
            odom2_data = np.hstack([position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w])
                                        
            xyz = np.empty((0, 3))  # Initialize an empty array for XYZ coordinates
            rgb = np.empty((0, 3))  # Initialize an empty array for RGB values
            gen = pc2.read_points(msg1, field_names=("x", "y", "z", "rgb"), skip_nans=False)
            int_data = list(gen)
            sampling_rate=2
            sampled_int_data = int_data[::sampling_rate]
            for x in tqdm(sampled_int_data, desc="Processing data"):
                test = x[3] 
                # Cast float32 to int so that bitwise operations are possible
                s = struct.pack('>f', test)
                i = struct.unpack('>l', s)[0]
                # Get RGB values using bitwise operations
                pack = ctypes.c_uint32(i).value
                r = (pack & 0x00FF0000) >> 16
                g = (pack & 0x0000FF00) >> 8
                b = (pack & 0x000000FF)
                # Append XYZ and RGB values to arrays
                xyz = np.append(xyz, [[x[0], x[1], x[2]]], axis=0)
                rgb = np.append(rgb, [[r, g, b]], axis=0)
            # Combine XYZ and RGB arrays horizontally
            pointcloud_data = np.hstack((xyz, rgb))

            # Check if enough time has elapsed since the last save
            current_time = datetime.datetime.now()
            elapsed_time = (current_time - last_save_time).total_seconds()
            if elapsed_time >= save_interval:
                # Save the data to a file
                filename = current_time.strftime("%y%m%d%H%M%S%f")[:-3] + '_data.npz'
                filepath = os.path.join(data_dir, filename)
                np.savez(filepath, pointcloud=pointcloud_data, est_odom=odom1_data, gt_odom=odom2_data)
                
                # Update the last save time
                last_save_time = current_time
                
                # Print out the data saved, the interval, and the data saving time
                print(f"{saved_data_cnt} Data saved!, Interval: {elapsed_time}, Time: {current_time}")
                saved_data_cnt += 1

def main():
   # Initialize variables to store messages from different topics
    pointcloud_msg = None
    odom1_msg = None
    odom2_msg = None
    logger=DataLogger
    bag = rosbag.Bag('/home/ml/risk-aware_planning2/src/uncertainty_predictor/gps_dataset1_2024-04-01-15-36-58.bag')
    # Iterate over messages in the bag file
    for topic, msg, t in bag.read_messages(topics=['/camera/depth/image_raw/points', '/mavros/local_position/odom', '/airsim_node/hmcl/gt_odom']):
        if topic == '/camera/depth/image_raw/points':
            pointcloud_msg = msg
        elif topic == '/mavros/local_position/odom':
            odom1_msg = msg
        elif topic == '/airsim_node/hmcl/gt_odom':
            odom2_msg = msg

        # If messages from all topics are available, call the callback function
        if pointcloud_msg is not None and odom1_msg is not None and odom2_msg is not None:
            if abs((odom2_msg.header.stamp-odom1_msg.header.stamp).to_sec()) < 0.2 and abs((odom1_msg.header.stamp-pointcloud_msg.header.stamp).to_sec()) < 0.2 and  abs((odom2_msg.header.stamp-pointcloud_msg.header.stamp).to_sec()) < 0.2:
                logger.pointcloud_callback(pointcloud_msg, odom1_msg, odom2_msg)
                print("data_saved!")
                # Reset messages after processing
                pointcloud_msg = None
                odom1_msg = None
                odom2_msg = None
            else:
                print("data not synced")

    bag.close()

if __name__ == '__main__':
    main()
