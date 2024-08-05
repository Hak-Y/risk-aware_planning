#include "voxblox_ros/uncertainty_server.h"

#include "voxblox_ros/conversions.h"
#include "voxblox_ros/ros_params.h"

namespace voxblox {

UncertaintyServer::UncertaintyServer(const ros::NodeHandle& nh,
                       const ros::NodeHandle& nh_private)
    : UncertaintyServer(nh, nh_private, getUncertaintyMapConfigFromRosParam(nh_private),
                 getUncertaintyIntegratorConfigFromRosParam(nh_private),
                 getTsdfMapConfigFromRosParam(nh_private),
                 getTsdfIntegratorConfigFromRosParam(nh_private),
                 getMeshIntegratorConfigFromRosParam(nh_private)) {}

UncertaintyServer::UncertaintyServer(const ros::NodeHandle& nh,
                       const ros::NodeHandle& nh_private,
                       const UncertaintyMap::Config& uncertainty_config,
                       const UncertaintyIntegrator::Config& uncertainty_integrator_config,
                       const TsdfMap::Config& tsdf_config,
                       const TsdfIntegratorBase::Config& tsdf_integrator_config,
                       const MeshIntegratorConfig& mesh_config)
    : TsdfServer(nh, nh_private, tsdf_config, tsdf_integrator_config,
                 mesh_config),
      clear_sphere_for_planning_(false),
      publish_uncertainty_map_(false),
      publish_traversable_(false),
      traversability_radius_(1.0),
      incremental_update_(true),
      num_subscribers_uncertainty_map_(0) {
  // Set up map and integrator.
  uncertainty_map_.reset(new UncertaintyMap(uncertainty_config));
  uncertainty_integrator_.reset(new UncertaintyIntegrator(uncertainty_integrator_config,
                                            tsdf_map_->getTsdfLayerPtr(),
                                            uncertainty_map_->getUncertaintyLayerPtr()));

  setupRos();
}

void UncertaintyServer::setupRos() {
  // Set up publisher.
  uncertainty_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZI> >("uncertainty_pointcloud",
                                                              1, true);
  uncertainty_slice_pub_ = nh_private_.advertise<pcl::PointCloud<pcl::PointXYZI> >(
      "uncertainty_slice", 1, true);
  traversable_pub_ = nh_private_.advertise<pcl::PointCloud<pcl::PointXYZI> >(
      "traversable", 1, true);

  uncertainty_map_pub_ =
      nh_private_.advertise<voxblox_msgs::Layer>("uncertainty_map_out", 1, false);

  // Set up subscriber.
  uncertainty_map_sub_ = nh_private_.subscribe("uncertainty_map_in", 1,
                                        &UncertaintyServer::uncertaintyMapCallback, this);

  // Whether to clear each new pose as it comes in, and then set a sphere
  // around it to occupied.
  nh_private_.param("clear_sphere_for_planning", clear_sphere_for_planning_,
                    clear_sphere_for_planning_);
  nh_private_.param("publish_uncertainty_map", publish_uncertainty_map_, publish_uncertainty_map_);

  // Special output for traversable voxels. Publishes all voxels with distance
  // at least traversibility radius.
  nh_private_.param("publish_traversable", publish_traversable_,
                    publish_traversable_);
  nh_private_.param("traversability_radius", traversability_radius_,
                    traversability_radius_);

  double update_uncertainty_every_n_sec = 1.0;
  nh_private_.param("update_uncertainty_every_n_sec", update_uncertainty_every_n_sec,
                    update_uncertainty_every_n_sec);

  if (update_uncertainty_every_n_sec > 0.0) {
    update_uncertainty_timer_ =
        nh_private_.createTimer(ros::Duration(update_uncertainty_every_n_sec),
                                &UncertaintyServer::updateUncertaintyEvent, this);
  }
}

void UncertaintyServer::publishAllUpdatedUncertaintyVoxels() {
  // Create a pointcloud with distance = intensity.
  pcl::PointCloud<pcl::PointXYZI> pointcloud;

  createDistancePointcloudFromUncertaintyLayer(uncertainty_map_->getUncertaintyLayer(), &pointcloud);

  pointcloud.header.frame_id = world_frame_;
  uncertainty_pointcloud_pub_.publish(pointcloud);
}

void UncertaintyServer::publishSlices() {
  TsdfServer::publishSlices();

  pcl::PointCloud<pcl::PointXYZI> pointcloud;

  constexpr int kZAxisIndex = 2;
  createDistancePointcloudFromUncertaintyLayerSlice(
      uncertainty_map_->getUncertaintyLayer(), kZAxisIndex, slice_level_, &pointcloud);

  pointcloud.header.frame_id = world_frame_;
  uncertainty_slice_pub_.publish(pointcloud);
}

bool UncertaintyServer::generateUncertaintyCallback(
    std_srvs::Empty::Request& /*request*/,      // NOLINT
    std_srvs::Empty::Response& /*response*/) {  // NOLINT
  const bool clear_uncertainty = true;
  if (clear_uncertainty) {
    uncertainty_integrator_->updateFromTsdfLayerBatch();
  } else {
    const bool clear_updated_flag = true;
    uncertainty_integrator_->updateFromTsdfLayer(clear_updated_flag);
  }
  publishAllUpdatedUncertaintyVoxels();
  publishSlices();
  return true;
}

void UncertaintyServer::updateUncertaintyEvent(const ros::TimerEvent& /*event*/) {
  // std:: cout << "UncertaintyServer::updateUncertaintyEvent" << std::endl;
  updateUncertainty();
}

void UncertaintyServer::publishPointclouds() {
  publishAllUpdatedUncertaintyVoxels();
  if (publish_slices_) {
    publishSlices();
  }

  if (publish_traversable_) {
    publishTraversable();
  }

  TsdfServer::publishPointclouds();
}

void UncertaintyServer::publishTraversable() {
  pcl::PointCloud<pcl::PointXYZI> pointcloud;
  createFreePointcloudFromUncertaintyLayer(uncertainty_map_->getUncertaintyLayer(),
                                    traversability_radius_, &pointcloud);
  pointcloud.header.frame_id = world_frame_;
  traversable_pub_.publish(pointcloud);
}

void UncertaintyServer::publishMap(bool reset_remote_map) {
  if (!publish_uncertainty_map_) {
    return;
  }

  int subscribers = this->uncertainty_map_pub_.getNumSubscribers();
  if (subscribers > 0) {
    if (num_subscribers_uncertainty_map_ < subscribers) {
      // Always reset the remote map and send all when a new subscriber
      // subscribes. A bit of overhead for other subscribers, but better than
      // inconsistent map states.
      reset_remote_map = true;
    }
    const bool only_updated = !reset_remote_map;
    timing::Timer publish_map_timer("map/publish_uncertainty");
    voxblox_msgs::Layer layer_msg;
    serializeLayerAsMsg<UncertaintyVoxel>(this->uncertainty_map_->getUncertaintyLayer(),
                                   only_updated, &layer_msg);
    if (reset_remote_map) {
      layer_msg.action = static_cast<uint8_t>(MapDerializationAction::kReset);
    }
    this->uncertainty_map_pub_.publish(layer_msg);
    publish_map_timer.Stop();
  }
  num_subscribers_uncertainty_map_ = subscribers;
  TsdfServer::publishMap();
}

bool UncertaintyServer::saveMap(const std::string& file_path) {
  // Output TSDF map first, then UNCERTAINTY.
  const bool success = TsdfServer::saveMap(file_path);

  constexpr bool kClearFile = false;
  return success &&
         io::SaveLayer(uncertainty_map_->getUncertaintyLayer(), file_path, kClearFile);
}

bool UncertaintyServer::loadMap(const std::string& file_path) {
  // Load in the same order: TSDF first, then UNCERTAINTY.
  bool success = TsdfServer::loadMap(file_path);

  constexpr bool kMultipleLayerSupport = true;
  return success &&
         io::LoadBlocksFromFile(
             file_path, Layer<UncertaintyVoxel>::BlockMergingStrategy::kReplace,
             kMultipleLayerSupport, uncertainty_map_->getUncertaintyLayerPtr());
}

void UncertaintyServer::updateUncertainty() {
  // std:: cout << "UncertaintyServer::updateUncertainty()" << std::endl;
  if (tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks() > 0) {
    const bool clear_updated_flag_uncertainty = true;
    uncertainty_integrator_->updateFromTsdfLayer(clear_updated_flag_uncertainty);
  }
}

void UncertaintyServer::updateUncertaintyBatch(bool full_euclidean) {
  if (tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks() > 0) {
    uncertainty_integrator_->setFullEuclidean(full_euclidean);
    uncertainty_integrator_->updateFromTsdfLayerBatch();
  }
}

float UncertaintyServer::getUncertaintyMaxDistance() const {
  return uncertainty_integrator_->getUncertaintyMaxDistance();
}

void UncertaintyServer::setUncertaintyMaxDistance(float max_distance) {
  uncertainty_integrator_->setUncertaintyMaxDistance(max_distance);
}

float UncertaintyServer::getTraversabilityRadius() const {
  return traversability_radius_;
}

void UncertaintyServer::setTraversabilityRadius(float traversability_radius) {
  traversability_radius_ = traversability_radius;
}

void UncertaintyServer::newPoseCallback(const Transformation& T_G_C) {
  if (clear_sphere_for_planning_) {
    uncertainty_integrator_->addNewRobotPosition(T_G_C.getPosition());
  }

  timing::Timer block_remove_timer("remove_distant_blocks");
  uncertainty_map_->getUncertaintyLayerPtr()->removeDistantBlocks(
      T_G_C.getPosition(), max_block_distance_from_body_);
  block_remove_timer.Stop();
}

void UncertaintyServer::uncertaintyMapCallback(const voxblox_msgs::Layer& layer_msg) {
  timing::Timer receive_map_timer("map/receive_uncertainty");

  bool success =
      deserializeMsgToLayer<UncertaintyVoxel>(layer_msg, uncertainty_map_->getUncertaintyLayerPtr());

  if (!success) {
    ROS_ERROR_THROTTLE(10, "Got an invalid UNCERTAINTY map message!");
  } else {
    ROS_INFO_ONCE("Got an UNCERTAINTY map from ROS topic!");
    if (publish_pointclouds_) {
      publishPointclouds();
    }
  }
}

void UncertaintyServer::clear() {
  uncertainty_map_->getUncertaintyLayerPtr()->removeAllBlocks();
  uncertainty_integrator_->clear();
  CHECK_EQ(uncertainty_map_->getUncertaintyLayerPtr()->getNumberOfAllocatedBlocks(), 0u);

  TsdfServer::clear();

  // Publish a message to reset the map to all subscribers.
  constexpr bool kResetRemoteMap = true;
  publishMap(kResetRemoteMap);
}

}  // namespace voxblox
