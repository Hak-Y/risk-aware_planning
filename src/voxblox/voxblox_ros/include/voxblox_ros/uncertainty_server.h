#ifndef VOXBLOX_ROS_UNCERTAINTY_SERVER_H_
#define VOXBLOX_ROS_UNCERTAINTY_SERVER_H_

#include <memory>
#include <string>

#include <voxblox/core/uncertainty_map.h>
#include <voxblox/integrator/uncertainty_integrator.h>
#include <voxblox_msgs/Layer.h>

#include "voxblox_ros/tsdf_server.h"

namespace voxblox {

class UncertaintyServer : public TsdfServer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  UncertaintyServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  UncertaintyServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private,
             const UncertaintyMap::Config& uncertainty_config,
             const UncertaintyIntegrator::Config& uncertainty_integrator_config,
             const TsdfMap::Config& tsdf_config,
             const TsdfIntegratorBase::Config& tsdf_integrator_config,
             const MeshIntegratorConfig& mesh_config);
  virtual ~UncertaintyServer() {}

  bool generateUncertaintyCallback(std_srvs::Empty::Request& request,     // NOLINT
                            std_srvs::Empty::Response& response);  // NOLINT

  void publishAllUpdatedUncertaintyVoxels();
  virtual void publishSlices();
  void publishTraversable();

  virtual void publishPointclouds();
  virtual void newPoseCallback(const Transformation& T_G_C);
  virtual void publishMap(bool reset_remote_map = false);
  virtual bool saveMap(const std::string& file_path);
  virtual bool loadMap(const std::string& file_path);

  void updateUncertaintyEvent(const ros::TimerEvent& event);

  /// Call this to update the Uncertainty based on latest state of the TSDF map,
  /// considering only the newly updated parts of the TSDF map (checked with
  /// the Uncertainty updated bit in Update::Status).
  void updateUncertainty();
  /// Update the Uncertainty all at once; clear the existing map.
  void updateUncertaintyBatch(bool full_euclidean = false);

  // Overwrites the layer with what's coming from the topic!
  void uncertaintyMapCallback(const voxblox_msgs::Layer& layer_msg);

  inline std::shared_ptr<UncertaintyMap> getUncertaintyMapPtr() { return uncertainty_map_; }
  inline std::shared_ptr<const UncertaintyMap> getUncertaintyMapPtr() const {
    return uncertainty_map_;
  }

  bool getClearSphere() const { return clear_sphere_for_planning_; }
  void setClearSphere(bool clear_sphere_for_planning) {
    clear_sphere_for_planning_ = clear_sphere_for_planning;
  }
  float getUncertaintyMaxDistance() const;
  void setUncertaintyMaxDistance(float max_distance);
  float getTraversabilityRadius() const;
  void setTraversabilityRadius(float traversability_radius);

  /**
   * These are for enabling or disabling incremental update of the Uncertainty. Use
   * carefully.
   */
  void disableIncrementalUpdate() { incremental_update_ = false; }
  void enableIncrementalUpdate() { incremental_update_ = true; }

  virtual void clear();

 protected:
  /// Sets up publishing and subscribing. Should only be called from
  /// constructor.
  void setupRos();

  /// Publish markers for visualization.
  ros::Publisher uncertainty_pointcloud_pub_;
  ros::Publisher uncertainty_slice_pub_;
  ros::Publisher traversable_pub_;

  /// Publish the complete map for other nodes to consume.
  ros::Publisher uncertainty_map_pub_;

  /// Subscriber to subscribe to another node generating the map.
  ros::Subscriber uncertainty_map_sub_;

  /// Services.
  ros::ServiceServer generate_uncertainty_srv_;

  /// Timers.
  ros::Timer update_uncertainty_timer_;

  bool clear_sphere_for_planning_;
  bool publish_uncertainty_map_;
  bool publish_traversable_;
  float traversability_radius_;
  bool incremental_update_;
  int num_subscribers_uncertainty_map_;

  // Uncertainty maps.
  std::shared_ptr<UncertaintyMap> uncertainty_map_;
  std::unique_ptr<UncertaintyIntegrator> uncertainty_integrator_;
};

}  // namespace voxblox

#endif  // VOXBLOX_ROS_UNCERTAINTY_SERVER_H_
