#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import airsim
import cv2
import numpy as np
from nav_msgs.msg import Odometry
import quaternion

CLAHE_ENABLED = False  # when enabled, RGB image is enhanced using CLAHE

CAMERA_FX = 386.126953125
CAMERA_FY = 386.126953125
CAMERA_CX = 320
CAMERA_CY = 240

CAMERA_K1 = 0.0
CAMERA_K2 = 0.0
CAMERA_P1 = 0.0
CAMERA_P2 = 0.0
CAMERA_P3 = 0.0

IMAGE_WIDTH = 640  # resolution should match values in settings.json
IMAGE_HEIGHT = 480

class KinectPublisher:
    def __init__(self):
        self.bridge_rgb = CvBridge()
        self.msg_rgb = Image()
        self.bridge_d = CvBridge()
        self.msg_d = Image()
        self.msg_info = CameraInfo()
        self.msg_tf = TFMessage()
        self.odom_msg = Odometry()

    def odom_callback(self, msg):
        self.odom_msg = msg

    def getDepthImage(self, response_d):
        img_depth = np.array(response_d.image_data_float, dtype=np.float32)
        img_depth = img_depth.reshape(response_d.height, response_d.width)
        return img_depth

    def getRGBImage(self, response_rgb):
        img1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response_rgb.height, response_rgb.width, 3)
        img_rgb = img_rgb[..., :3][..., ::-1]
        return img_rgb
    
    def GetCurrentTime(self):
        self.ros_time = rospy.Time.now()

    def CreateDMessage(self, img_depth):
        self.msg_d.header.stamp = rospy.Time.now()
        self.msg_d.header.frame_id = "camera_depth_optical_frame"
        self.msg_d.encoding = "32FC1"
        self.msg_d.height = IMAGE_HEIGHT
        self.msg_d.width = IMAGE_WIDTH
        self.msg_d.data = self.bridge_d.cv2_to_imgmsg(img_depth, "32FC1").data
        self.msg_d.is_bigendian = 0
        self.msg_d.step = self.msg_d.width * 4
        return self.msg_d

    def CreateRGBMessage(self, img_rgb):
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        self.msg_rgb.header.stamp = rospy.Time.now()
        self.msg_rgb.header.frame_id = "camera_depth_optical_frame"
        self.msg_rgb.encoding = "bgr8"
        self.msg_rgb.height = IMAGE_HEIGHT
        self.msg_rgb.width = IMAGE_WIDTH
        self.msg_rgb.data = self.bridge_rgb.cv2_to_imgmsg(img_rgb, "bgr8").data
        self.msg_rgb.is_bigendian = 0
        self.msg_rgb.step = self.msg_rgb.width * 3
        return self.msg_rgb
    
    def CreateInfoMessage(self, frame_id):
        self.msg_info.header.frame_id = frame_id
        self.msg_info.height = self.msg_d.height
        self.msg_info.width = self.msg_d.width
        self.msg_info.distortion_model = "plumb_bob"

        self.msg_info.D = [CAMERA_K1, CAMERA_K2, CAMERA_P1, CAMERA_P2, CAMERA_P3]

        self.msg_info.K = [0] * 9
        self.msg_info.K[0] = CAMERA_FX
        self.msg_info.K[2] = CAMERA_CX
        self.msg_info.K[4] = CAMERA_FY
        self.msg_info.K[5] = CAMERA_CY
        self.msg_info.K[8] = 1

        self.msg_info.R = [0] * 9
        self.msg_info.R[0] = 1
        self.msg_info.R[4] = 1
        self.msg_info.R[8] = 1

        self.msg_info.P = [0] * 12
        self.msg_info.P[0] = CAMERA_FX
        self.msg_info.P[2] = CAMERA_CX
        self.msg_info.P[5] = CAMERA_FY
        self.msg_info.P[6] = CAMERA_CY
        self.msg_info.P[10] = 1

        self.msg_info.binning_x = self.msg_info.binning_y = 0
        self.msg_info.roi.x_offset = self.msg_info.roi.y_offset = self.msg_info.roi.height = self.msg_info.roi.width = 0
        self.msg_info.roi.do_rectify = False
        self.msg_info.header.stamp = self.msg_d.header.stamp
        return self.msg_info

    def CreateTFMessage(self):
        self.msg_tf.transforms.append(TransformStamped())
        self.msg_tf.transforms[0].header.stamp = rospy.Time.now()
        self.msg_tf.transforms[0].header.frame_id = "camera_link"
        self.msg_tf.transforms[0].child_frame_id = "camera_rgb_frame"
        self.msg_tf.transforms[0].transform.translation.x = 0.000
        self.msg_tf.transforms[0].transform.translation.y = 0
        self.msg_tf.transforms[0].transform.translation.z = 0.000
        self.msg_tf.transforms[0].transform.rotation.x = 0.00
        self.msg_tf.transforms[0].transform.rotation.y = 0.00
        self.msg_tf.transforms[0].transform.rotation.z = 0.00
        self.msg_tf.transforms[0].transform.rotation.w = 1.00

        self.msg_tf.transforms.append(TransformStamped())
        self.msg_tf.transforms[1].header.stamp = rospy.Time.now()
        self.msg_tf.transforms[1].header.frame_id = "camera_rgb_frame"
        self.msg_tf.transforms[1].child_frame_id = "camera_rgb_optical_frame"
        self.msg_tf.transforms[1].transform.translation.x = 0.000
        self.msg_tf.transforms[1].transform.translation.y = 0.000
        self.msg_tf.transforms[1].transform.translation.z = 0.000
        self.msg_tf.transforms[1].transform.rotation.x = -0.500
        self.msg_tf.transforms[1].transform.rotation.y = 0.500
        self.msg_tf.transforms[1].transform.rotation.z = -0.500
        self.msg_tf.transforms[1].transform.rotation.w = 0.500

        self.msg_tf.transforms.append(TransformStamped())
        self.msg_tf.transforms[2].header.stamp = rospy.Time.now()
        self.msg_tf.transforms[2].header.frame_id = "camera_link"
        self.msg_tf.transforms[2].child_frame_id = "camera_depth_frame"
        self.msg_tf.transforms[2].transform.translation.x = 0
        self.msg_tf.transforms[2].transform.translation.y = 0
        self.msg_tf.transforms[2].transform.translation.z = 0
        self.msg_tf.transforms[2].transform.rotation.x = 0.00
        self.msg_tf.transforms[2].transform.rotation.y = 0.00
        self.msg_tf.transforms[2].transform.rotation.z = 0.00
        self.msg_tf.transforms[2].transform.rotation.w = 1.00

        self.msg_tf.transforms.append(TransformStamped())
        self.msg_tf.transforms[3].header.stamp = rospy.Time.now()
        self.msg_tf.transforms[3].header.frame_id = "camera_depth_frame"
        self.msg_tf.transforms[3].child_frame_id = "camera_depth_optical_frame"
        self.msg_tf.transforms[3].transform.translation.x = 0.000
        self.msg_tf.transforms[3].transform.translation.y = 0.000
        self.msg_tf.transforms[3].transform.translation.z = 0.000
        self.msg_tf.transforms[3].transform.rotation.x = -0.500
        self.msg_tf.transforms[3].transform.rotation.y = 0.500
        self.msg_tf.transforms[3].transform.rotation.z = -0.500
        self.msg_tf.transforms[3].transform.rotation.w = 0.50

        return self.msg_tf

    def pub_gtpose(self, res_gt):
        position = res_gt.position
        orientation = res_gt.orientation
        # Create a Pose message with the transformed coordinates and orientation
        odom_msg = Odometry()
        odom_msg.child_frame_id = "base_link"
        odom_msg.header.frame_id = "odom"
        odom_msg.header.stamp = rospy.Time.now()

        ## Visual mode
        # odom_msg.pose.pose.position.x = -position.x_val
        # odom_msg.pose.pose.position.y = position.y_val
        # odom_msg.pose.pose.position.z = -position.z_val+0.1806
        # odom_msg.pose.pose.orientation.x = orientation.x_val
        # odom_msg.pose.pose.orientation.y = orientation.y_val
        # odom_msg.pose.pose.orientation.z = orientation.w_val
        # odom_msg.pose.pose.orientation.w = orientation.z_val

        ## GPS mode
        q = quaternion.from_float_array([orientation.x_val, orientation.y_val, orientation.w_val, orientation.z_val]) # idk why this order...
        R = quaternion.as_rotation_matrix(q)
        angle = np.pi/2
        R_roll = np.array([[1, 0, 0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)]])
        R_rotated = np.dot(R_roll, R)
        q_rotated = quaternion.from_rotation_matrix(R_rotated)
        odom_msg.pose.pose.position.x = position.y_val
        odom_msg.pose.pose.position.y = position.x_val
        odom_msg.pose.pose.position.z = -position.z_val + 0.1806
        odom_msg.pose.pose.orientation.x = q_rotated.w # idk why this order...
        odom_msg.pose.pose.orientation.y = q_rotated.x # idk why this order...
        odom_msg.pose.pose.orientation.z = q_rotated.y # idk why this order...
        odom_msg.pose.pose.orientation.w = q_rotated.z # idk why this order...

        # Publish the transformed pose
        return odom_msg

if __name__ == "__main__":
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    # client.armDisarm(True)
    rospy.init_node('airsim_publisher', anonymous=True)
    publisher_d = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=1)
    publisher_rgb = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=1)
    publisher_info_depth = rospy.Publisher('/camera/depth/camera_info', CameraInfo, queue_size=1)
    publisher_info_rgb = rospy.Publisher('/camera/rgb/camera_info', CameraInfo, queue_size=1)
    publisher_tf = rospy.Publisher('/tf', TFMessage, queue_size=1)
    gtpose_pub = rospy.Publisher('/airsim_node/hmcl/gt_odom', Odometry, queue_size=10)
    rate = rospy.Rate(10)  # 30hz
    pub = KinectPublisher()

    rospy.Subscriber('/mavros/local_position/odom', Odometry, pub.odom_callback)

    while not rospy.is_shutdown():
        responses = client.simGetImages([
            airsim.ImageRequest('depth_cam', airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False),
            airsim.ImageRequest('scene_cam', airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        ])
        res_gt = client.simGetVehiclePose("hmcl")
        odom_msg = pub.pub_gtpose(res_gt)
        gtpose_pub.publish(odom_msg)
        img_depth = pub.getDepthImage(responses[0])
        img_rgb = pub.getRGBImage(responses[1])
        pub.GetCurrentTime()
        
        msg_d = pub.CreateDMessage(img_depth)
        msg_rgb = pub.CreateRGBMessage(img_rgb)

        msg_info_depth = pub.CreateInfoMessage("camera_depth_optical_frame")
        msg_info_rgb = pub.CreateInfoMessage("camera_rgb_optical_frame")
        msg_tf = pub.CreateTFMessage()

        publisher_rgb.publish(msg_rgb)
        publisher_d.publish(msg_d)
        publisher_info_depth.publish(msg_info_depth)
        publisher_info_rgb.publish(msg_info_rgb)
        publisher_tf.publish(msg_tf)

        del pub.msg_info.D[:]
        del pub.msg_tf.transforms[:]

        rate.sleep()
#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import airsim
import cv2
import numpy as np
from nav_msgs.msg import Odometry
import quaternion

CLAHE_ENABLED = False  # when enabled, RGB image is enhanced using CLAHE

CAMERA_FX = 386.126953125
CAMERA_FY = 386.126953125
CAMERA_CX = 320
CAMERA_CY = 240

CAMERA_K1 = 0.0
CAMERA_K2 = 0.0
CAMERA_P1 = 0.0
CAMERA_P2 = 0.0
CAMERA_P3 = 0.0

IMAGE_WIDTH = 640  # resolution should match values in settings.json
IMAGE_HEIGHT = 480

class KinectPublisher:
    def __init__(self):
        self.bridge_rgb = CvBridge()
        self.msg_rgb = Image()
        self.bridge_d = CvBridge()
        self.msg_d = Image()
        self.msg_info = CameraInfo()
        self.msg_tf = TFMessage()
        self.odom_msg = Odometry()

    def odom_callback(self, msg):
        self.odom_msg = msg

    def getDepthImage(self, response_d):
        img_depth = np.array(response_d.image_data_float, dtype=np.float32)
        img_depth = img_depth.reshape(response_d.height, response_d.width)
        return img_depth

    def getRGBImage(self, response_rgb):
        img1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response_rgb.height, response_rgb.width, 3)
        img_rgb = img_rgb[..., :3][..., ::-1]
        return img_rgb
    
    def GetCurrentTime(self):
        self.ros_time = rospy.Time.now()

    def CreateDMessage(self, img_depth):
        self.msg_d.header.stamp = rospy.Time.now()
        self.msg_d.header.frame_id = "camera_depth_optical_frame"
        self.msg_d.encoding = "32FC1"
        self.msg_d.height = IMAGE_HEIGHT
        self.msg_d.width = IMAGE_WIDTH
        self.msg_d.data = self.bridge_d.cv2_to_imgmsg(img_depth, "32FC1").data
        self.msg_d.is_bigendian = 0
        self.msg_d.step = self.msg_d.width * 4
        return self.msg_d

    def CreateRGBMessage(self, img_rgb):
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        self.msg_rgb.header.stamp = rospy.Time.now()
        self.msg_rgb.header.frame_id = "camera_depth_optical_frame"
        self.msg_rgb.encoding = "bgr8"
        self.msg_rgb.height = IMAGE_HEIGHT
        self.msg_rgb.width = IMAGE_WIDTH
        self.msg_rgb.data = self.bridge_rgb.cv2_to_imgmsg(img_rgb, "bgr8").data
        self.msg_rgb.is_bigendian = 0
        self.msg_rgb.step = self.msg_rgb.width * 3
        return self.msg_rgb
    
    def CreateInfoMessage(self, frame_id):
        self.msg_info.header.frame_id = frame_id
        self.msg_info.height = self.msg_d.height
        self.msg_info.width = self.msg_d.width
        self.msg_info.distortion_model = "plumb_bob"

        self.msg_info.D = [CAMERA_K1, CAMERA_K2, CAMERA_P1, CAMERA_P2, CAMERA_P3]

        self.msg_info.K = [0] * 9
        self.msg_info.K[0] = CAMERA_FX
        self.msg_info.K[2] = CAMERA_CX
        self.msg_info.K[4] = CAMERA_FY
        self.msg_info.K[5] = CAMERA_CY
        self.msg_info.K[8] = 1

        self.msg_info.R = [0] * 9
        self.msg_info.R[0] = 1
        self.msg_info.R[4] = 1
        self.msg_info.R[8] = 1

        self.msg_info.P = [0] * 12
        self.msg_info.P[0] = CAMERA_FX
        self.msg_info.P[2] = CAMERA_CX
        self.msg_info.P[5] = CAMERA_FY
        self.msg_info.P[6] = CAMERA_CY
        self.msg_info.P[10] = 1

        self.msg_info.binning_x = self.msg_info.binning_y = 0
        self.msg_info.roi.x_offset = self.msg_info.roi.y_offset = self.msg_info.roi.height = self.msg_info.roi.width = 0
        self.msg_info.roi.do_rectify = False
        self.msg_info.header.stamp = self.msg_d.header.stamp
        return self.msg_info

    def CreateTFMessage(self):
        self.msg_tf.transforms.append(TransformStamped())
        self.msg_tf.transforms[0].header.stamp = rospy.Time.now()
        self.msg_tf.transforms[0].header.frame_id = "camera_link"
        self.msg_tf.transforms[0].child_frame_id = "camera_rgb_frame"
        self.msg_tf.transforms[0].transform.translation.x = 0.000
        self.msg_tf.transforms[0].transform.translation.y = 0
        self.msg_tf.transforms[0].transform.translation.z = 0.000
        self.msg_tf.transforms[0].transform.rotation.x = 0.00
        self.msg_tf.transforms[0].transform.rotation.y = 0.00
        self.msg_tf.transforms[0].transform.rotation.z = 0.00
        self.msg_tf.transforms[0].transform.rotation.w = 1.00

        self.msg_tf.transforms.append(TransformStamped())
        self.msg_tf.transforms[1].header.stamp = rospy.Time.now()
        self.msg_tf.transforms[1].header.frame_id = "camera_rgb_frame"
        self.msg_tf.transforms[1].child_frame_id = "camera_rgb_optical_frame"
        self.msg_tf.transforms[1].transform.translation.x = 0.000
        self.msg_tf.transforms[1].transform.translation.y = 0.000
        self.msg_tf.transforms[1].transform.translation.z = 0.000
        self.msg_tf.transforms[1].transform.rotation.x = -0.500
        self.msg_tf.transforms[1].transform.rotation.y = 0.500
        self.msg_tf.transforms[1].transform.rotation.z = -0.500
        self.msg_tf.transforms[1].transform.rotation.w = 0.500

        self.msg_tf.transforms.append(TransformStamped())
        self.msg_tf.transforms[2].header.stamp = rospy.Time.now()
        self.msg_tf.transforms[2].header.frame_id = "camera_link"
        self.msg_tf.transforms[2].child_frame_id = "camera_depth_frame"
        self.msg_tf.transforms[2].transform.translation.x = 0
        self.msg_tf.transforms[2].transform.translation.y = 0
        self.msg_tf.transforms[2].transform.translation.z = 0
        self.msg_tf.transforms[2].transform.rotation.x = 0.00
        self.msg_tf.transforms[2].transform.rotation.y = 0.00
        self.msg_tf.transforms[2].transform.rotation.z = 0.00
        self.msg_tf.transforms[2].transform.rotation.w = 1.00

        self.msg_tf.transforms.append(TransformStamped())
        self.msg_tf.transforms[3].header.stamp = rospy.Time.now()
        self.msg_tf.transforms[3].header.frame_id = "camera_depth_frame"
        self.msg_tf.transforms[3].child_frame_id = "camera_depth_optical_frame"
        self.msg_tf.transforms[3].transform.translation.x = 0.000
        self.msg_tf.transforms[3].transform.translation.y = 0.000
        self.msg_tf.transforms[3].transform.translation.z = 0.000
        self.msg_tf.transforms[3].transform.rotation.x = -0.500
        self.msg_tf.transforms[3].transform.rotation.y = 0.500
        self.msg_tf.transforms[3].transform.rotation.z = -0.500
        self.msg_tf.transforms[3].transform.rotation.w = 0.50

        return self.msg_tf

    def pub_gtpose(self, res_gt):
        position = res_gt.position
        orientation = res_gt.orientation
        # Create a Pose message with the transformed coordinates and orientation
        odom_msg = Odometry()
        odom_msg.child_frame_id = "base_link"
        odom_msg.header.frame_id = "odom"
        odom_msg.header.stamp = rospy.Time.now()

        ## Visual mode
        # odom_msg.pose.pose.position.x = -position.x_val
        # odom_msg.pose.pose.position.y = position.y_val
        # odom_msg.pose.pose.position.z = -position.z_val+0.1806
        # odom_msg.pose.pose.orientation.x = orientation.x_val
        # odom_msg.pose.pose.orientation.y = orientation.y_val
        # odom_msg.pose.pose.orientation.z = orientation.w_val
        # odom_msg.pose.pose.orientation.w = orientation.z_val

        ## GPS mode
        q = quaternion.from_float_array([orientation.x_val, orientation.y_val, orientation.w_val, orientation.z_val]) # idk why this order...
        R = quaternion.as_rotation_matrix(q)
        angle = np.pi/2
        R_roll = np.array([[1, 0, 0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)]])
        R_rotated = np.dot(R_roll, R)
        q_rotated = quaternion.from_rotation_matrix(R_rotated)
        odom_msg.pose.pose.position.x = position.y_val
        odom_msg.pose.pose.position.y = position.x_val
        odom_msg.pose.pose.position.z = -position.z_val + 0.1806
        odom_msg.pose.pose.orientation.x = q_rotated.w # idk why this order...
        odom_msg.pose.pose.orientation.y = q_rotated.x # idk why this order...
        odom_msg.pose.pose.orientation.z = q_rotated.y # idk why this order...
        odom_msg.pose.pose.orientation.w = q_rotated.z # idk why this order...

        # Publish the transformed pose
        return odom_msg

if __name__ == "__main__":
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    # client.armDisarm(True)
    rospy.init_node('airsim_publisher', anonymous=True)
    publisher_d = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=1)
    publisher_rgb = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=1)
    publisher_info_depth = rospy.Publisher('/camera/depth/camera_info', CameraInfo, queue_size=1)
    publisher_info_rgb = rospy.Publisher('/camera/rgb/camera_info', CameraInfo, queue_size=1)
    publisher_tf = rospy.Publisher('/tf', TFMessage, queue_size=1)
    gtpose_pub = rospy.Publisher('/airsim_node/hmcl/gt_odom', Odometry, queue_size=10)
    rate = rospy.Rate(10)  # 30hz
    pub = KinectPublisher()

    rospy.Subscriber('/mavros/local_position/odom', Odometry, pub.odom_callback)

    while not rospy.is_shutdown():
        responses = client.simGetImages([
            airsim.ImageRequest('depth_cam', airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False),
            airsim.ImageRequest('scene_cam', airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        ])
        res_gt = client.simGetVehiclePose("hmcl")
        odom_msg = pub.pub_gtpose(res_gt)
        gtpose_pub.publish(odom_msg)
        img_depth = pub.getDepthImage(responses[0])
        img_rgb = pub.getRGBImage(responses[1])
        pub.GetCurrentTime()
        
        msg_d = pub.CreateDMessage(img_depth)
        msg_rgb = pub.CreateRGBMessage(img_rgb)

        msg_info_depth = pub.CreateInfoMessage("camera_depth_optical_frame")
        msg_info_rgb = pub.CreateInfoMessage("camera_rgb_optical_frame")
        msg_tf = pub.CreateTFMessage()

        publisher_rgb.publish(msg_rgb)
        publisher_d.publish(msg_d)
        publisher_info_depth.publish(msg_info_depth)
        publisher_info_rgb.publish(msg_info_rgb)
        publisher_tf.publish(msg_tf)

        del pub.msg_info.D[:]
        del pub.msg_tf.transforms[:]

        rate.sleep()

