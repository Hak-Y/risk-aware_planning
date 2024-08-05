#include "active_3d_planning_core/module/trajectory_evaluator/uncertainty_evaluator.h"

#include <algorithm>
#include <string>

#include "active_3d_planning_core/module/module_factory.h"
#include "active_3d_planning_core/planner/planner_I.h"

namespace active_3d_planning {
namespace trajectory_evaluator {

// Factory Registration
ModuleFactoryRegistry::Registration<UncertaintyEvaluator>
    UncertaintyEvaluator ::registration("UncertaintyEvaluator");

UncertaintyEvaluator::UncertaintyEvaluator(PlannerI& planner)
    : SimulatedSensorEvaluator(planner) {}

void UncertaintyEvaluator::setupFromParamMap(Module::ParamMap* param_map) {
  // setup parent
  SimulatedSensorEvaluator::setupFromParamMap(param_map);


  // params
  setParam<double>(param_map, "gain_unknown", &p_gain_unknown_, 1.0);
  setParam<double>(param_map, "gain_occupied", &p_gain_occupied_, 0.0);
  setParam<double>(param_map, "gain_free", &p_gain_free_, 0.0);
  setParam<double>(param_map, "gain_unknown_outer", &p_gain_unknown_outer_,
                   0.0);
  setParam<double>(param_map, "gain_occupied_outer", &p_gain_occupied_outer_,
                   0.0);
  setParam<double>(param_map, "gain_free_outer", &p_gain_free_outer_, 0.0);

  // setup map
  map_ = dynamic_cast<map::UNCERTAINTYMap*>(&(planner_.getMap()));
  if (!map_) {
    planner_.printError(
        "'UncertainEvaluator' requires a map of type 'UncertaintyMap'!");
  }

    // outer volume (if not specified will not be counted)
  std::string ns = (*param_map)["param_namespace"];
  std::string outer_volume_args;
  setParam<std::string>(param_map, "outer_volume_args", &outer_volume_args,
                        ns + "/outer_volume");
  outer_volume_ = planner_.getFactory().createModule<BoundingVolume>(
      outer_volume_args, planner_, verbose_modules_);

  // constants
  c_min_gain_ = std::min({p_gain_unknown_, p_gain_occupied_, p_gain_free_,
                          p_gain_unknown_outer_, p_gain_occupied_outer_,
                          p_gain_free_outer_});
  c_max_gain_ = std::max({p_gain_unknown_, p_gain_occupied_, p_gain_free_,
                          p_gain_unknown_outer_, p_gain_occupied_outer_,
                          p_gain_free_outer_});
  

}

bool UncertaintyEvaluator::computeGainFromVisibleVoxels(TrajectorySegment* traj_in) {
  // std::cout << "debug1: " <<  std::endl;
  if (!traj_in->info) {
     traj_in->gain = 0.0;
    return false;
  }
  // std::cout << "debug2: " <<  std::endl;
  // remove all already observed voxels, count number of new voxels
  SimulatedSensorInfo* info =
      reinterpret_cast<SimulatedSensorInfo*>(traj_in->info.get());

  unsigned char voxel_state;
  std::vector<float> voxel_values;
  
  float voxel_value=1.3579;
  for (int i = 0; i < info->visible_voxels.size(); ++i) {
    // voxel_state = map_->getVoxelState(info->visible_voxels[i]);
    // std::cout << voxel_state << std::endl;
    voxel_value = map_->getVoxelFeature(info->visible_voxels[i]);
    voxel_values.push_back(voxel_value);
  }
  // std::cout << "info->visible_voxels.size() : " << info->visible_voxels.size() << std::endl;
  float cov = voxel_value;
  float collision_risk = cov;
  // std::cout << "voxel_value: " << voxel_value << std::endl;
  size_t i = 0;

  while (i < info->visible_voxels.size()) {
    if (planner_.getMap().isObserved(info->visible_voxels[i])) {
      // if (map_->isObserved(info->visible_voxels[i])) {
      info->visible_voxels.erase(info->visible_voxels.begin() + i);
    } 
    else {
      ++i;
    }
  }
  
  // std::cout << "info->visible_voxels.size()2 : " << info->visible_voxels.size()<< std::endl;
  // traj_in->gain = info->visible_voxels.size() - collision_risk;
  if (planner_.isReturnPlanningMode())
  {
    Eigen::Vector3d base_position = Eigen::Vector3d(0,0,1.5);
    Eigen::Vector3d traj_position = traj_in->trajectory.back().position_W;
    float r2b_gain = (1/(base_position - traj_position).norm())*100;
    // std::cout << r2b_gain << std::endl;
    traj_in->gain = r2b_gain;
  }
  else {traj_in->gain = info->visible_voxels.size();}
  
  
  
  return true;


}  // namespace trajectory_evaluator
}  // namespace active_3d_planning
}
