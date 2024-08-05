#include <plan_env/sdf_map.h>
#include <plan_env/map_ros.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/io/ply_io.h>

#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>

#include <visualization_msgs/Marker.h>

#include <fstream>


#include <iostream>
#include <Eigen/Dense>



namespace bg_planner {
MapROS::MapROS() {
}

MapROS::~MapROS() {
}

void MapROS::setMap(SDFMap* map) {
  this->map_ = map;
}

void MapROS::init() {
  node_.param("map_ros/esdf_slice_height", esdf_slice_height_, -0.1);
  node_.param("map_ros/visualization_truncate_height", visualization_truncate_height_, -0.1);
  node_.param("map_ros/visualization_truncate_low", visualization_truncate_low_, -0.1);
  node_.param("map_ros/show_occ_time", show_occ_time_, false);
  node_.param("map_ros/show_esdf_time", show_esdf_time_, false);
  node_.param("map_ros/show_all_map", show_all_map_, false);
  node_.param("map_ros/frame_id", frame_id_, string("map"));
  node_.param("map_ros/expl_stat_fname", out_fname_, string("~/Code/pm_ws/benchmark_gazebo/bg1_vol.txt"));
  init_time_ = false;
  int pose_type; // 1 is pose, 2 is odom
  node_.param("map_ros/pose_type", pose_type, 1);
  // cam2body_ << 0.0, 0.0, 1.0, 0.0,
  //              -1.0, 0.0, 0.0, 0.0,
  //              0.0, -1.0, 0.0, 0.0,
  //              0.0, 0.0, 0.0, 1.0;

  proj_points_cnt = 0;

  local_updated_ = false;
  esdf_need_update_ = false;
  fuse_time_ = 0.0;
  esdf_time_ = 0.0;
  max_fuse_time_ = 0.0;
  max_esdf_time_ = 0.0;
  fuse_num_ = 0;
  esdf_num_ = 0;
  known_volume = 0;
  depth_image_.reset(new cv::Mat);

  rand_noise_ = normal_distribution<double>(0, 0.1);
  random_device rd;
  eng_ = default_random_engine(rd());

  //esdf_timer_ = node_.createTimer(ros::Duration(0.05), &MapROS::updateESDFCallback, this);
  vis_timer_ = node_.createTimer(ros::Duration(0.05), &MapROS::visCallback, this);

  map_all_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_all", 10);
  map_inflate_all_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_inflate_all", 10);
  map_local_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_local", 10);
  map_local_inflate_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_local_inflate", 10);
  unknown_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/unknown", 10);
  esdf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/esdf", 10);
  update_range_pub_ = node_.advertise<visualization_msgs::Marker>("/sdf_map/update_range", 10);
  depth_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/depth_cloud", 10);

  cloud_sub_.reset(
      new message_filters::Subscriber<sensor_msgs::PointCloud2>(node_, "/map_ros/cloud", 50));
  pose_sub_.reset(
      new message_filters::Subscriber<geometry_msgs::PoseStamped>(node_, "/map_ros/pose", 50));

  if (pose_type == 1)
  {

    sync_cloud_pose_.reset(new message_filters::Synchronizer<MapROS::SyncPolicyCloudPose>(
        MapROS::SyncPolicyCloudPose(100), *cloud_sub_, *pose_sub_));
    sync_cloud_pose_->registerCallback(boost::bind(&MapROS::cloudPoseCallback, this, _1, _2));

  }
  else if (pose_type == 2)
  {
  }
  map_start_time_ = ros::Time::now();
}

void MapROS::visCallback(const ros::TimerEvent& e) {
  publishMapLocal();
  if (show_all_map_) {
    // Limit the frequency of all map
    static double tpass = 0.0;
    tpass += (e.current_real - e.last_real).toSec();
    if (tpass > 0.2) {
      publishMapAll();
      publishMapInflateAll();
      tpass = 0.0;
    }
  }
}

void MapROS::cloudPoseCallback(const sensor_msgs::PointCloud2ConstPtr& msg,
                               const geometry_msgs::PoseStampedConstPtr& pose) {
  // std::cout << "cloudPoseCallback" << std::endl;
  camera_pos_(0) = pose->pose.position.x;
  camera_pos_(1) = pose->pose.position.y;
  camera_pos_(2) = pose->pose.position.z;
  // // camera_q_ = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x,
  // //                                pose->pose.orientation.y, pose->pose.orientation.z);
  // pcl::PointCloud<pcl::PointXYZ> cloud;
  // pcl::fromROSMsg(*msg, cloud);
  // int num = cloud.points.size();
  // map_->inputPointCloud(cloud, num, camera_pos_);

  // if (local_updated_) {
  //   map_->clearAndInflateLocalMap();
  //   esdf_need_update_ = true;
  //   local_updated_ = false;
  // }

    pcl::PointCloud<pcl::PointXYZ> cloud_in;
    pcl::fromROSMsg(*msg, cloud_in);

    // Create a 4x4 transformation matrix (cam_pose) from translation and rotation
    Eigen::Affine3f cam_pose = Eigen::Affine3f::Identity();

    // Extract translation and rotation from the pose message
    Eigen::Vector3f translation(pose->pose.position.x, pose->pose.position.y, pose->pose.position.z);
    Eigen::Quaternionf rotation(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z);

    cam_pose.translation() = translation;
    cam_pose.rotate(rotation);

    Eigen::Affine3f cam2optical = Eigen::Affine3f::Identity();

    Eigen::Vector3f translation2(0, 0, 0);
    Eigen::Quaternionf rotation2(-0.5, +0.5, -0.5, +0.5);

    cam2optical.translation() = translation2;
    cam2optical.rotate(rotation2);

    Eigen::Affine3f c_to_w = cam_pose*cam2optical ;

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(cloud_in, *transformed_cloud, c_to_w);

    int num = transformed_cloud->points.size();
    map_->inputPointCloud(*transformed_cloud, num, camera_pos_);

    if (local_updated_) {
      map_->clearAndInflateLocalMap();
      esdf_need_update_ = true;
      local_updated_ = false;
    }

}

void MapROS::getPointCloud(pcl::PointCloud<pcl::PointXYZ> &cloud){


  cloud = latest_cloud_;
  
}

void MapROS::publishMapAll() {
  if (!init_time_) {
    map_start_time_ = ros::Time::now();
    init_time_ = true;
  }
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud1, cloud2;

  // std::cout << "x: "<<  map_->mp_->box_min_(0) << ", " << map_->mp_->box_max_(0) << std::endl;
  // std::cout << "y: "<<  map_->mp_->box_min_(1) << ", " << map_->mp_->box_max_(1) << std::endl;
  // std::cout << "z: "<<  map_->mp_->box_min_(2) << ", " << map_->mp_->box_max_(2) << std::endl;

  for (int x = map_->mp_->box_min_(0) /* + 1 */; x < map_->mp_->box_max_(0); ++x)
    for (int y = map_->mp_->box_min_(1) /* + 1 */; y < map_->mp_->box_max_(1); ++y)
      for (int z = map_->mp_->box_min_(2) /* + 1 */; z < map_->mp_->box_max_(2); ++z) {
        // Eigen::Vector3d pos;
        // map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
        // std::cout << "point: " << pos << std::endl;
        // std::cout << "occ: " << map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] << std::endl;
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] > map_->mp_->min_occupancy_log_)
        //if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y, z)] == 1)
        {
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          // std::cout << "point: " << pos << std::endl;
          if (pos(2) > visualization_truncate_height_) continue;
          if (pos(2) < visualization_truncate_low_) continue;
          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud1.push_back(pt);
        }
      }
  cloud1.width = cloud1.points.size();
  cloud1.height = 1;
  cloud1.is_dense = true;
  cloud1.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud1, cloud_msg);
  map_all_pub_.publish(cloud_msg);

  // Output time and known volumn
  double time_now = (ros::Time::now() - map_start_time_).toSec();
  known_volume = 0;
  for (int x = map_->mp_->box_min_(0) /* + 1 */; x < map_->mp_->box_max_(0); ++x)
    for (int y = map_->mp_->box_min_(1) /* + 1 */; y < map_->mp_->box_max_(1); ++y)
      for (int z = map_->mp_->box_min_(2) /* + 1 */; z < map_->mp_->box_max_(2); ++z) {
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] > map_->mp_->clamp_min_log_ - 1e-3)
          known_volume += 0.1 * 0.1 * 0.1;
      }
  ROS_INFO_STREAM("[VOLUME]:" << known_volume);
}


void MapROS::publishMapInflateAll() {
  //cout << "Publish map inflate all" << endl;
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud1, cloud2;

  for (int x = map_->mp_->box_min_(0) /* + 1 */; x < map_->mp_->box_max_(0); ++x)
    for (int y = map_->mp_->box_min_(1) /* + 1 */; y < map_->mp_->box_max_(1); ++y)
      for (int z = map_->mp_->box_min_(2) /* + 1 */; z < map_->mp_->box_max_(2); ++z) {
        if (map_->md_->occupancy_buffer_inflate_global_[map_->toAddress(x, y, z)] >= map_->mp_->min_occupancy_log_)
        //if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y, z)] == 1)
        {
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > visualization_truncate_height_) continue;
          if (pos(2) < visualization_truncate_low_) continue;
          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud1.push_back(pt);
        }
      }
  cloud1.width = cloud1.points.size();
  cloud1.height = 1;
  cloud1.is_dense = true;
  cloud1.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  latest_cloud_ = cloud1;
  pcl::toROSMsg(cloud1, cloud_msg);
  map_inflate_all_pub_.publish(cloud_msg);
}


void MapROS::publishMapLocal() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::PointCloud<pcl::PointXYZ> cloud2;
  Eigen::Vector3i min_cut = map_->md_->local_bound_min_;
  Eigen::Vector3i max_cut = map_->md_->local_bound_max_;
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  // for (int z = min_cut(2); z <= max_cut(2); ++z)
  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = map_->mp_->box_min_(2); z < map_->mp_->box_max_(2); ++z) {
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] > map_->mp_->min_occupancy_log_) {
          // Occupied cells
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > visualization_truncate_height_) continue;
          if (pos(2) < visualization_truncate_low_) continue;

          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud.push_back(pt);
        }
        // else if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y, z)] == 1)
        // {
        //   // Inflated occupied cells
        //   Eigen::Vector3d pos;
        //   map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
        //   if (pos(2) > visualization_truncate_height_)
        //     continue;
        //   if (pos(2) < visualization_truncate_low_)
        //     continue;

        //   pt.x = pos(0);
        //   pt.y = pos(1);
        //   pt.z = pos(2);
        //   cloud2.push_back(pt);
        // }
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  cloud2.width = cloud2.points.size();
  cloud2.height = 1;
  cloud2.is_dense = true;
  cloud2.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  map_local_pub_.publish(cloud_msg);
  pcl::toROSMsg(cloud2, cloud_msg);
  map_local_inflate_pub_.publish(cloud_msg);
}

void MapROS::publishUnknown() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  Eigen::Vector3i min_cut = map_->md_->local_bound_min_;
  Eigen::Vector3i max_cut = map_->md_->local_bound_max_;
  map_->boundIndex(max_cut);
  map_->boundIndex(min_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z) {
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] < map_->mp_->clamp_min_log_ - 1e-3) {
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > visualization_truncate_height_) continue;
          if (pos(2) < visualization_truncate_low_) continue;
          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud.push_back(pt);
        }
      }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  unknown_pub_.publish(cloud_msg);
}

void MapROS::publishDepth() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  for (int i = 0; i < proj_points_cnt; ++i) {
    cloud.push_back(point_cloud_.points[i]);
  }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  depth_pub_.publish(cloud_msg);
}

void MapROS::publishUpdateRange() {
  Eigen::Vector3d esdf_min_pos, esdf_max_pos, cube_pos, cube_scale;
  visualization_msgs::Marker mk;
  map_->indexToPos(map_->md_->local_bound_min_, esdf_min_pos);
  map_->indexToPos(map_->md_->local_bound_max_, esdf_max_pos);

  cube_pos = 0.5 * (esdf_min_pos + esdf_max_pos);
  cube_scale = esdf_max_pos - esdf_min_pos;
  mk.header.frame_id = frame_id_;
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::CUBE;
  mk.action = visualization_msgs::Marker::ADD;
  mk.id = 0;
  mk.pose.position.x = cube_pos(0);
  mk.pose.position.y = cube_pos(1);
  mk.pose.position.z = cube_pos(2);
  mk.scale.x = cube_scale(0);
  mk.scale.y = cube_scale(1);
  mk.scale.z = cube_scale(2);
  mk.color.a = 0.3;
  mk.color.r = 1.0;
  mk.color.g = 0.0;
  mk.color.b = 0.0;
  mk.pose.orientation.w = 1.0;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;

  update_range_pub_.publish(mk);
}

// void MapROS::publishESDF() {
//   double dist;
//   pcl::PointCloud<pcl::PointXYZI> cloud;
//   pcl::PointXYZI pt;

//   const double min_dist = 0.0;
//   const double max_dist = 3.0;

//   Eigen::Vector3i min_cut = map_->md_->local_bound_min_ - Eigen::Vector3i(map_->mp_->local_map_margin_,
//                                                                           map_->mp_->local_map_margin_,
//                                                                           map_->mp_->local_map_margin_);
//   Eigen::Vector3i max_cut = map_->md_->local_bound_max_ + Eigen::Vector3i(map_->mp_->local_map_margin_,
//                                                                           map_->mp_->local_map_margin_,
//                                                                           map_->mp_->local_map_margin_);
//   map_->boundIndex(min_cut);
//   map_->boundIndex(max_cut);

//   for (int x = min_cut(0); x <= max_cut(0); ++x)
//     for (int y = min_cut(1); y <= max_cut(1); ++y) {
//       Eigen::Vector3d pos;
//       map_->indexToPos(Eigen::Vector3i(x, y, 1), pos);
//       pos(2) = esdf_slice_height_;
//       dist = map_->getDistance(pos);
//       dist = min(dist, max_dist);
//       dist = max(dist, min_dist);
//       pt.x = pos(0);
//       pt.y = pos(1);
//       pt.z = -0.2;
//       pt.intensity = (dist - min_dist) / (max_dist - min_dist);
//       cloud.push_back(pt);
//     }

//   cloud.width = cloud.points.size();
//   cloud.height = 1;
//   cloud.is_dense = true;
//   cloud.header.frame_id = frame_id_;
//   sensor_msgs::PointCloud2 cloud_msg;
//   pcl::toROSMsg(cloud, cloud_msg);

//   esdf_pub_.publish(cloud_msg);

//   // ROS_INFO("pub esdf");
// }
}