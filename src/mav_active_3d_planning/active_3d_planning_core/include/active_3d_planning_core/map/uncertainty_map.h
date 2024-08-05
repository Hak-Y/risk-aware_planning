#ifndef ACTIVE_3D_PLANNING_CORE_MAP_UNCERTAINTY_MAP_H_
#define ACTIVE_3D_PLANNING_CORE_MAP_UNCERTAINTY_MAP_H_

#include "active_3d_planning_core/data/trajectory.h"
#include "active_3d_planning_core/map/occupancy_map.h"

namespace active_3d_planning {
namespace map {

// base interface for Latent Feature voxelgrid maps
class UNCERTAINTYMap : public OccupancyMap {
 public:
  explicit UNCERTAINTYMap(PlannerI& planner) : OccupancyMap(planner) {}  // NOLINT

  virtual ~UNCERTAINTYMap() = default;

  // get the stored distance
  virtual double getVoxelDistance(const Eigen::Vector3d& point) = 0;

  virtual double getVoxelFeature(const Eigen::Vector3d& point) = 0;

  // get the stored weight
  virtual double getVoxelWeight(const Eigen::Vector3d& point) = 0;

  // get the maximum allowed weight (return 0 if using uncapped weights)
  virtual double getMaximumWeight() = 0;
};

}  // namespace map
}  // namespace active_3d_planning

#endif  // ACTIVE_3D_PLANNING_CORE_MAP_LATENT_FEATURE_MAP_H_
