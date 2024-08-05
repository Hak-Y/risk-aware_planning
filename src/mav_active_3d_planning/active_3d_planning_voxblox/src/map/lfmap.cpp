#include "active_3d_planning_voxblox/map/lf.h"

#include <voxblox_ros/ros_params.h>

#include "active_3d_planning_core/data/system_constraints.h"

namespace active_3d_planning {
namespace map {

ModuleFactoryRegistry::Registration<LFMap> LFMap::registration(
    "LFMap");

LFMap::LFMap(PlannerI& planner) : UNCERTAINTYMap(planner) {}

voxblox::UncertaintyServer& LFMap::getUncertaintyServer() { return *uncertainty_server_; }

void LFMap::setupFromParamMap(Module::ParamMap* param_map) {
  // create an uncertainty server
  ros::NodeHandle nh("");
  ros::NodeHandle nh_private("~");
  uncertainty_server_.reset(new voxblox::UncertaintyServer(nh, nh_private));
  uncertainty_server_->setTraversabilityRadius(
      planner_.getSystemConstraints().collision_radius);

  // cache constants
  c_voxel_size_ = uncertainty_server_->getUncertaintyMapPtr()->voxel_size();
  c_block_size_ = uncertainty_server_->getUncertaintyMapPtr()->block_size();
  c_maximum_weight_ = voxblox::getTsdfIntegratorConfigFromRosParam(nh_private)
                          .max_weight;  // direct access is not exposed
}

bool LFMap::isTraversable(const Eigen::Vector3d& position,
                               const Eigen::Quaterniond& orientation) {
  double distance = 0.0;
  if (uncertainty_server_->getUncertaintyMapPtr()->getDistanceAtPosition(position,
                                                           &distance)) {
    // This means the voxel is observed
    return (distance > planner_.getSystemConstraints().collision_radius);
  }
  return false;
}

bool LFMap::isObserved(const Eigen::Vector3d& point) {
  // std::cout << "LFMap::isObserved" << std::endl;
  return uncertainty_server_->getUncertaintyMapPtr()->isObserved(point);
}

// get occupancy
unsigned char LFMap::getVoxelState(const Eigen::Vector3d& point) {
  double distance = 0.0;
  if (uncertainty_server_->getUncertaintyMapPtr()->getDistanceAtPosition(point, &distance)) {
    // This means the voxel is observed
    if (distance < c_voxel_size_) {
      return LFMap::OCCUPIED;
    } else {
      return LFMap::FREE;
    }
  } else {
    return LFMap::UNKNOWN;
  }
}


// get voxel size
double LFMap::getVoxelSize() { return c_voxel_size_; }

// get the center of a voxel from input point
bool LFMap::getVoxelCenter(Eigen::Vector3d* center,
                                const Eigen::Vector3d& point) {
  voxblox::BlockIndex block_id = uncertainty_server_->getUncertaintyMapPtr()
                                     ->getUncertaintyLayerPtr()
                                     ->computeBlockIndexFromCoordinates(
                                         point.cast<voxblox::FloatingPoint>());
  *center = voxblox::getOriginPointFromGridIndex(block_id, c_block_size_)
                .cast<double>();
  voxblox::VoxelIndex voxel_id =
      voxblox::getGridIndexFromPoint<voxblox::VoxelIndex>(
          (point - *center).cast<voxblox::FloatingPoint>(),
          1.0 / c_voxel_size_);
  *center += voxblox::getCenterPointFromGridIndex(voxel_id, c_voxel_size_)
                 .cast<double>();
  return true;
}

// get the stored TSDF distance
double LFMap::getVoxelDistance(const Eigen::Vector3d& point) {
  voxblox::Point voxblox_point(point.x(), point.y(), point.z());
  voxblox::Block<voxblox::TsdfVoxel>::Ptr block =
      uncertainty_server_->getTsdfMapPtr()
          ->getTsdfLayerPtr()
          ->getBlockPtrByCoordinates(voxblox_point);
  if (block) {
    voxblox::TsdfVoxel* tsdf_voxel =
        block->getVoxelPtrByCoordinates(voxblox_point);
    if (tsdf_voxel) {
      return tsdf_voxel->distance;
    }
  }
  return 0.0;
}

// get the stored latent feature
double LFMap::getVoxelFeature(const Eigen::Vector3d& point) {
  voxblox::Point voxblox_point(point.x(), point.y(), point.z());
  voxblox::Block<voxblox::UncertaintyVoxel>::Ptr block =
      uncertainty_server_->getUncertaintyMapPtr()
          ->getUncertaintyLayerPtr()
          ->getBlockPtrByCoordinates(voxblox_point);
  if (block) {
    voxblox::UncertaintyVoxel* uncertainty_voxel =
        block->getVoxelPtrByCoordinates(voxblox_point);
    if (uncertainty_voxel) {
      return uncertainty_voxel->feature;
      // return 0.5678;
    }
  }
  return 0.1234;
}


// get the stored weight
double LFMap::getVoxelWeight(const Eigen::Vector3d& point) {
  voxblox::Point voxblox_point(point.x(), point.y(), point.z());
  voxblox::Block<voxblox::TsdfVoxel>::Ptr block =
      uncertainty_server_->getTsdfMapPtr()
          ->getTsdfLayerPtr()
          ->getBlockPtrByCoordinates(voxblox_point);
  if (block) {
    voxblox::TsdfVoxel* tsdf_voxel =
        block->getVoxelPtrByCoordinates(voxblox_point);
    if (tsdf_voxel) {
      return tsdf_voxel->weight;
    }
  }
  return 0.0;
}

// get the maximum allowed weight (return 0 if using uncapped weights)
double LFMap::getMaximumWeight() { return c_maximum_weight_; }

}  // namespace map
}  // namespace active_3d_planning
