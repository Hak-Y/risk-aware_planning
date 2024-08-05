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

# Define the path to the data directory relative to the current Python file
class DataLogger:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data_gps/dataset2')
        print("saving data at ",self.data_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.last_save_time = datetime.datetime.now()
        self.pointcloud_data = None
        self.odom1_data = None
        self.odom2_data = None
        self.lock = threading.Lock()
        self.saved_data_cnt = 1
        self.save_interval = 0.2  # Minimum time interval between saves (in seconds)

    def pointcloud_callback(self, msg1, msg2, msg3):
        with self.lock:
            # Message from pointcloud_sub
            # Message from odom1_sub
            # Message from odom2_sub
        
            position = msg2.pose.pose.position
            orientation = msg2.pose.pose.orientation
            self.odom1_data = np.hstack([position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w])

            position = msg3.pose.pose.position
            orientation = msg3.pose.pose.orientation
            self.odom2_data = np.hstack([position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w])
                                        
            xyz = np.empty((0, 3))  # Initialize an empty array for XYZ coordinates
            rgb = np.empty((0, 3))  # Initialize an empty array for RGB values
            gen = pc2.read_points(msg1, field_names=("x", "y", "z", "rgb"), skip_nans=False)
            int_data = list(gen)
            sampling_rate=2
            sampled_int_data = int_data[::sampling_rate]
            for x in sampled_int_data:
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
            self.pointcloud_data = np.hstack((xyz, rgb))

            # fig_debug_pointcloud_raw = plt.figure()
            # ax_debug_pointcloud_raw = fig_debug_pointcloud_raw.add_subplot(111, projection='3d')
            # norm = plt.Normalize(np.min(xyz,0)[2], np.max(xyz,0)[2])
            # cmap = plt.cm.viridis  # You can choose any colormap you prefer
            # scatter = ax_debug_pointcloud_raw.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2], cmap=cmap, norm=norm, marker='s', alpha=0.5)
            # ax_debug_pointcloud_raw.set_xlabel('X')
            # ax_debug_pointcloud_raw.set_ylabel('Y')
            # ax_debug_pointcloud_raw.set_zlabel('Z')
            # cbar = plt.colorbar(scatter)
            # cbar.set_label('Z')
            # plt.show()

            # Check if enough time has elapsed since the last save
            current_time = datetime.datetime.now()
            elapsed_time = (current_time - self.last_save_time).total_seconds()
            if elapsed_time >= self.save_interval:
                # Save the data to a file
                filename = current_time.strftime("%y%m%d%H%M%S%f")[:-3] + '_data.npz'
                filepath = os.path.join(self.data_dir, filename)
                np.savez(filepath, pointcloud=self.pointcloud_data, est_odom=self.odom1_data, gt_odom=self.odom2_data)
                
                # Update the last save time
                self.last_save_time = current_time
                
                # Print out the data saved, the interval, and the data saving time
                print(f"{self.saved_data_cnt} Data saved!, Interval: {elapsed_time}, Time: {current_time}")
                self.saved_data_cnt += 1

                # Clear the pointcloud_data array to prevent memory issues
                self.pointcloud_data = None
                self.odom1_data = None
                self.odom2_data = None

def main():
    rospy.init_node('data_saver', anonymous=True)

    # Create a DataLogger instance
    logger = DataLogger()

    # Subscribe to the point cloud topic
    pointcloud_sub = Subscriber('/voxel_grid/output', PointCloud2)
    # pointcloud_sub = Subscriber('/camera/depth/image_raw/points', PointCloud2)
    odom1_sub = Subscriber('/mavros/local_position/odom', Odometry)
    odom2_sub = Subscriber('/airsim_node/hmcl/gt_odom', Odometry)
    sync = ApproximateTimeSynchronizer([pointcloud_sub, odom1_sub, odom2_sub], queue_size=20, slop=0.1)
    sync.registerCallback(logger.pointcloud_callback)

    rospy.spin()

if __name__ == '__main__':
    main()
