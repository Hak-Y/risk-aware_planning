#include "ros/ros.h"
#include "visualization_msgs/Marker.h"
#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Odometry.h"
#include <mutex>

ros::Publisher pub;
visualization_msgs::Marker marker_msg;
visualization_msgs::Marker marker_yaw_msg;
nav_msgs::Odometry pose_msg;
bool newMarkerReceived = false;
bool newMarkerYawReceived = false;
int cur_target_idx = 1;

double check_distance(){
    geometry_msgs::Point target = marker_msg.points.at(cur_target_idx);
    double target_x = target.x;
    double target_y = target.y;
    double cur_x = pose_msg.pose.pose.position.x;
    double cur_y = pose_msg.pose.pose.position.y;
    double dist = (target_x-cur_x)*(target_x-cur_x) + (target_y-cur_y)*(target_y-cur_y);
    return dist;
}

void callback(const visualization_msgs::Marker::ConstPtr& msg)
{
    if (!msg->points.empty())  
    {   
        std::cout << "callback" << std::endl;
        marker_msg = *msg;
        newMarkerReceived = true;
        cur_target_idx=1;
    }
}

void timerCallback(const ros::TimerEvent&)
{   
    if (newMarkerYawReceived)
    {
        geometry_msgs::PoseStamped poseStamped;
        poseStamped.header.stamp = ros::Time::now();
        poseStamped.header.frame_id = "map"; 
        if (newMarkerReceived){
            geometry_msgs::Point targetPoint = marker_msg.points.at(cur_target_idx);
            // Copy the saved lastPoint to PoseStamped
            poseStamped.pose.position.x = targetPoint.x;
            poseStamped.pose.position.y = targetPoint.y;
            poseStamped.pose.position.z = 0.5;
        }
        else{
            poseStamped.pose.position.x = pose_msg.pose.pose.position.x;
            poseStamped.pose.position.y = pose_msg.pose.pose.position.y;
            poseStamped.pose.position.z = 0.5;
        }
        poseStamped.pose.orientation = marker_yaw_msg.pose.orientation;
        pub.publish(poseStamped);
    }
}

void callback_yaw(const visualization_msgs::Marker::ConstPtr& msg)
{
        marker_yaw_msg = *msg;
        newMarkerYawReceived = true;
}

void callback_odom(const nav_msgs::Odometry::ConstPtr& msg)
{
        pose_msg = *msg;

        if (newMarkerReceived){
            double dist = check_distance();
            if (dist < 0.3){
               if (cur_target_idx < marker_msg.points.size() - 1){
                    cur_target_idx++;
                }
            }
        }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "marker_to_pose_converter");
    ros::NodeHandle n;

    pub = n.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 10);
    ros::Subscriber sub = n.subscribe("/planning_vis/optimal_list", 10, callback);
    ros::Subscriber sub2 = n.subscribe("/behavior_goal_vis/next_goal", 10, callback_yaw);
    ros::Subscriber sub3 = n.subscribe("/mavros/local_position/odom", 10, callback_odom);

    // Create a timer with a callback function to publish the saved lastPoint
    ros::Timer timer = n.createTimer(ros::Duration(0.1), timerCallback);  // Set the desired interval

    ros::spin();

    return 0;
}
