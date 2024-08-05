#ifndef ACTIVE_3D_PLANNING_VOXBLOX_MAP_UNCERTAINTY_H_
#define ACTIVE_3D_PLANNING_VOXBLOX_MAP_UNCERTAINTY_H_

#include <memory>

#include <active_3d_planning_core/module/module_factory_registry.h>
#include <voxblox_ros/uncertainty_server.h>

#include "active_3d_planning_core/map/uncertainty_map.h"
// #include "active_3d_planning_core/map/tsdf_map.h"


namespace active_3d_planning {
namespace map {

// Voxblox as a map representation
class LFMap : public UNCERTAINTYMap {
// class LFMap : public TSDFMap {
 public:
  explicit LFMap(PlannerI& planner);  // NOLINT

  // implement virtual methods
  void setupFromParamMap(Module::ParamMap* param_map) override;

  // check collision for a single pose
  bool isTraversable(const Eigen::Vector3d& position,
                     const Eigen::Quaterniond& orientation) override;

  // check whether point is part of the map
  bool isObserved(const Eigen::Vector3d& point) override;

  // get occupancy
  unsigned char getVoxelState(const Eigen::Vector3d& point) override;

  // get voxel size
  double getVoxelSize() override;

  // get the center of a voxel from input point
  bool getVoxelCenter(Eigen::Vector3d* center,
                      const Eigen::Vector3d& point) override;

  // get the stored distance
  double getVoxelDistance(const Eigen::Vector3d& point) override;
  
  double getVoxelFeature(const Eigen::Vector3d& point);

  // get the stored weight
  double getVoxelWeight(const Eigen::Vector3d& point) override;

  // get the maximum allowed weight (return 0 if using uncapped weights)
  double getMaximumWeight() override;

  // accessor to the server for specialized planners
  voxblox::UncertaintyServer& getUncertaintyServer();

 protected:
  static ModuleFactoryRegistry::Registration<LFMap> registration;

  // esdf server that contains the map, subscribe to external ESDF/TSDF updates
  std::unique_ptr<voxblox::UncertaintyServer> uncertainty_server_;

  // cache constants
  double c_voxel_size_;
  double c_block_size_;
  double c_maximum_weight_;
};

}  // namespace map
}  // namespace active_3d_planning

#endif  // ACTIVE_3D_PLANNING_VOXBLOX_MAP_UNCERTAINTY_H_
