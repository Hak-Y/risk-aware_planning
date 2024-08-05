#include "voxblox/integrator/uncertainty_integrator.h"

#include "voxblox/utils/planning_utils.h"

namespace voxblox {

UncertaintyIntegrator::UncertaintyIntegrator(const Config& config,
                               Layer<TsdfVoxel>* tsdf_layer,
                               Layer<UncertaintyVoxel>* uncertainty_layer)
    : config_(config), tsdf_layer_(tsdf_layer), uncertainty_layer_(uncertainty_layer) {
  CHECK(tsdf_layer_);
  CHECK(uncertainty_layer_);

  voxels_per_side_ = uncertainty_layer_->voxels_per_side();
  voxel_size_ = uncertainty_layer_->voxel_size();

  CHECK_EQ(uncertainty_layer_->voxels_per_side(), tsdf_layer_->voxels_per_side());
  CHECK_NEAR(uncertainty_layer_->voxel_size(), tsdf_layer_->voxel_size(), 1e-6);

  open_.setNumBuckets(config_.num_buckets, config_.max_distance_m);
}

// Used for planning - allocates sphere around as observed but occupied,
// and clears space in a sphere around current position.
void UncertaintyIntegrator::addNewRobotPosition(const Point& position) {
  timing::Timer clear_timer("uncertainty/clear_radius");

  // First set all in inner sphere to free.
  HierarchicalIndexMap block_voxel_list;
  timing::Timer sphere_timer("uncertainty/clear_radius/get_sphere");
  utils::getAndAllocateSphereAroundPoint(position, config_.clear_sphere_radius,
                                         uncertainty_layer_, &block_voxel_list);
  sphere_timer.Stop();
  for (const std::pair<BlockIndex, VoxelIndexList>& kv : block_voxel_list) {
    // Get block.
    // std::cout << "Line 36: "<< std::endl;
    Block<UncertaintyVoxel>::Ptr block_ptr = uncertainty_layer_->getBlockPtrByIndex(kv.first);

    for (const VoxelIndex& voxel_index : kv.second) {
      if (!block_ptr->isValidVoxelIndex(voxel_index)) {
        continue;
      }
      UncertaintyVoxel& uncertainty_voxel = block_ptr->getVoxelByVoxelIndex(voxel_index);
      // We can clear unobserved or hallucinated voxels.
      if (!uncertainty_voxel.observed || uncertainty_voxel.hallucinated) {
        if (uncertainty_voxel.hallucinated) {
          GlobalIndex global_index = getGlobalVoxelIndexFromBlockAndVoxelIndex(
              kv.first, voxel_index, voxels_per_side_);
          raise_.push(global_index);
        }
        uncertainty_voxel.distance = config_.default_distance_m;
        uncertainty_voxel.observed = true;
        uncertainty_voxel.hallucinated = true;
        uncertainty_voxel.parent.setZero();
        updated_blocks_.insert(kv.first);
      }
    }
  }

  // Second set all remaining unknown to occupied.
  HierarchicalIndexMap block_voxel_list_occ;
  timing::Timer outer_sphere_timer("uncertainty/clear_radius/get_outer_sphere");
  utils::getAndAllocateSphereAroundPoint(position,
                                         config_.occupied_sphere_radius,
                                         uncertainty_layer_, &block_voxel_list_occ);
  outer_sphere_timer.Stop();
  for (const std::pair<BlockIndex, VoxelIndexList>& kv : block_voxel_list_occ) {
    // Get block.
    std::cout << "Line 69: "<< std::endl;
    Block<UncertaintyVoxel>::Ptr block_ptr = uncertainty_layer_->getBlockPtrByIndex(kv.first);

    for (const VoxelIndex& voxel_index : kv.second) {
      if (!block_ptr->isValidVoxelIndex(voxel_index)) {
        continue;
      }
      UncertaintyVoxel& uncertainty_voxel = block_ptr->getVoxelByVoxelIndex(voxel_index);
      if (!uncertainty_voxel.observed) {
        uncertainty_voxel.distance = -config_.default_distance_m;
        uncertainty_voxel.observed = true;
        uncertainty_voxel.hallucinated = true;
        uncertainty_voxel.parent.setZero();
        updated_blocks_.insert(kv.first);
      } else if (!uncertainty_voxel.in_queue) {
        GlobalIndex global_index = getGlobalVoxelIndexFromBlockAndVoxelIndex(
            kv.first, voxel_index, voxels_per_side_);
        open_.push(global_index, uncertainty_voxel.distance);
      }
    }
  }

  VLOG(3) << "Changed " << updated_blocks_.size()
          << " blocks from unknown to free or occupied near the robot.";
  clear_timer.Stop();
}

void UncertaintyIntegrator::updateFromTsdfLayerBatch() {
  uncertainty_layer_->removeAllBlocks();
  BlockIndexList tsdf_blocks;
  tsdf_layer_->getAllAllocatedBlocks(&tsdf_blocks);
  tsdf_blocks.insert(tsdf_blocks.end(), updated_blocks_.begin(),
                     updated_blocks_.end());
  updated_blocks_.clear();
  updateFromTsdfBlocks(tsdf_blocks);
}

void UncertaintyIntegrator::updateFromTsdfLayer(bool clear_updated_flag) {
  // std:: cout << "UncertaintyIntegrator::updateFromTsdfLayer" << std::endl;
  BlockIndexList tsdf_blocks;
  tsdf_layer_->getAllUpdatedBlocks(Update::kUncertainty, &tsdf_blocks);
  // std::cout << "tsdf_blocks.size(): " << tsdf_blocks.size() << std::endl;
  // std::cout << "updated_blocks_.size(): " << updated_blocks_.size() << std::endl;
  tsdf_blocks.insert(tsdf_blocks.end(), updated_blocks_.begin(),
                     updated_blocks_.end());
  updated_blocks_.clear();
  const bool kIncremental = true;
  updateFromTsdfBlocks(tsdf_blocks, kIncremental);

  if (clear_updated_flag) {
    for (const BlockIndex& block_index : tsdf_blocks) {
      if (tsdf_layer_->hasBlock(block_index)) {
        tsdf_layer_->getBlockByIndex(block_index)
            .updated()
            .reset(Update::kUncertainty);
      }
    }
  }
}

void UncertaintyIntegrator::updateFromTsdfBlocks(const BlockIndexList& tsdf_blocks,
                                          bool incremental) {
  CHECK_EQ(tsdf_layer_->voxels_per_side(), uncertainty_layer_->voxels_per_side());
  timing::Timer uncertainty_timer("uncertainty");

  // Go through all blocks in TSDF and copy their values for relevant voxels.
  size_t num_lower = 0u;
  size_t num_raise = 0u;
  size_t num_new = 0u;
  timing::Timer propagate_timer("uncertainty/propagate_tsdf");
  // std::cout << "[UNCERTAINTY update]: Propagating " << tsdf_blocks.size() << " updated blocks from the TSDF."<<std::endl;
  for (const BlockIndex& block_index : tsdf_blocks) {
    // std::cout << "Line 139: " << std::endl;
    Block<TsdfVoxel>::ConstPtr tsdf_block =
        tsdf_layer_->getBlockPtrByIndex(block_index);
    if (!tsdf_block) {
      continue;
    }

    // Allocate the same block in the UNCERTAINTY layer.
    // Block indices are the same across all layers.
    Block<UncertaintyVoxel>::Ptr uncertainty_block =
        uncertainty_layer_->allocateBlockPtrByIndex(block_index);
    uncertainty_block->set_updated(true);

    const size_t num_voxels_per_block = tsdf_block->num_voxels();
    for (size_t lin_index = 0u; lin_index < num_voxels_per_block; ++lin_index) {
      const TsdfVoxel& tsdf_voxel =
          tsdf_block->getVoxelByLinearIndex(lin_index);
      // If this voxel is unobserved in the original map, skip it.
      if (tsdf_voxel.weight < config_.min_weight) {
        if (!incremental && config_.add_occupied_crust) {
          // Create a little crust of occupied voxels around.
          UncertaintyVoxel& uncertainty_voxel = uncertainty_block->getVoxelByLinearIndex(lin_index);
          uncertainty_voxel.distance = -config_.default_distance_m;
          uncertainty_voxel.observed = true;
          uncertainty_voxel.hallucinated = true;
          uncertainty_voxel.fixed = false;
        }
        continue;
      }

      UncertaintyVoxel& uncertainty_voxel = uncertainty_block->getVoxelByLinearIndex(lin_index);
      VoxelIndex voxel_index =
          uncertainty_block->computeVoxelIndexFromLinearIndex(lin_index);
      GlobalIndex global_index = getGlobalVoxelIndexFromBlockAndVoxelIndex(
          block_index, voxel_index, voxels_per_side_);

      const bool tsdf_fixed = isFixed(tsdf_voxel.distance);
      // If there was nothing there before:
      if (!uncertainty_voxel.observed || uncertainty_voxel.hallucinated) {
        if (uncertainty_voxel.hallucinated) {
          raise_.push(global_index);
        }
        if (tsdf_fixed) {
          // In fixed band, just add and lock it.
          uncertainty_voxel.distance = tsdf_voxel.distance;
          uncertainty_voxel.fixed = true;
          // Also add it to open so it can update the neighbors.
          uncertainty_voxel.in_queue = true;
          open_.push(global_index, uncertainty_voxel.distance);
        } else {
          // Not in the fixed band. Just copy the sign.
          uncertainty_voxel.distance =
              signum(tsdf_voxel.distance) * (config_.default_distance_m);
          uncertainty_voxel.fixed = false;

          if (incremental) {
            if (updateVoxelFromNeighbors(global_index)) {
              uncertainty_voxel.in_queue = true;
              open_.push(global_index, uncertainty_voxel.distance);
            }
          }
        }
        // No matter what, basically, the parent is reset.
        uncertainty_voxel.parent.setZero();
        num_new++;
      } else {
        // If this voxel DID exist before.
        // There are three main options:
        // (1a) unfix: if was fixed before but not anymore, raise.
        // (1) lower: uncertainty or tsdf is fixed, and tsdf is closer to surface than
        // it used to be.
        // (2) raise: uncertainty or tsdf is fixed, and tsdf is further from surface
        // than it used to be.
        // (3) sign flip: tsdf and uncertainty have different signs, otherwise the
        // lower and raise rules apply as above.
        if (tsdf_fixed || uncertainty_voxel.fixed) {
          if (!tsdf_fixed) {
            // New case: have to raise the voxel
            uncertainty_voxel.distance =
                signum(tsdf_voxel.distance) * config_.default_distance_m;
            uncertainty_voxel.parent.setZero();
            uncertainty_voxel.fixed = false;
            raise_.push(global_index);
            uncertainty_voxel.in_queue = true;
            open_.push(global_index, uncertainty_voxel.distance);
            num_raise++;
          } else if ((uncertainty_voxel.distance > 0.0f &&
                      tsdf_voxel.distance + config_.min_diff_m <
                          uncertainty_voxel.distance) ||
                     (uncertainty_voxel.distance <= 0.0f &&
                      tsdf_voxel.distance - config_.min_diff_m >
                          uncertainty_voxel.distance)) {
            // Lower.
            uncertainty_voxel.fixed = tsdf_fixed;
            if (uncertainty_voxel.fixed) {
              uncertainty_voxel.distance = tsdf_voxel.distance;
            } else {
              uncertainty_voxel.distance =
                  signum(tsdf_voxel.distance) * config_.default_distance_m;
            }
            uncertainty_voxel.parent.setZero();
            uncertainty_voxel.in_queue = true;
            open_.push(global_index, uncertainty_voxel.distance);
            num_lower++;
          } else if ((uncertainty_voxel.distance > 0.0f &&
                      tsdf_voxel.distance - config_.min_diff_m >
                          uncertainty_voxel.distance) ||
                     (uncertainty_voxel.distance <= 0.0f &&
                      tsdf_voxel.distance + config_.min_diff_m <
                          uncertainty_voxel.distance)) {
            // Raise.
            uncertainty_voxel.fixed = tsdf_fixed;
            if (uncertainty_voxel.fixed) {
              uncertainty_voxel.distance = tsdf_voxel.distance;
            } else {
              uncertainty_voxel.distance =
                  signum(tsdf_voxel.distance) * config_.default_distance_m;
            }
            uncertainty_voxel.parent.setZero();
            raise_.push(global_index);
            uncertainty_voxel.in_queue = true;
            open_.push(global_index, uncertainty_voxel.distance);
            num_raise++;
          }
        } else if (signum(tsdf_voxel.distance) != signum(uncertainty_voxel.distance)) {
          // This means UNCERTAINTY was positive and TSDF is negative.
          // So lower.
          if (tsdf_voxel.distance < uncertainty_voxel.distance) {
            uncertainty_voxel.distance =
                signum(tsdf_voxel.distance) * config_.default_distance_m;
            uncertainty_voxel.parent.setZero();
            uncertainty_voxel.in_queue = true;
            open_.push(global_index, uncertainty_voxel.distance);
            num_lower++;
          } else {
            // Otherwise UNCERTAINTY was negative and TSDF is positive.
            // So raise.
            uncertainty_voxel.distance =
                signum(tsdf_voxel.distance) * config_.default_distance_m;
            uncertainty_voxel.parent.setZero();
            raise_.push(global_index);
            num_raise++;
          }
        }
        // Otherwise we just don't care. Not fixed voxels that match the right
        // sign can be whatever value that they want to be.
      }

      uncertainty_voxel.observed = true;
      uncertainty_voxel.hallucinated = false;
    }
  }

  propagate_timer.Stop();
  VLOG(3) << "[UNCERTAINTY update]: Lower: " << num_lower << " Raise: " << num_raise
          << " New: " << num_new;

  timing::Timer raise_timer("uncertainty/raise_uncertainty");
  processRaiseSet();
  raise_timer.Stop();

  timing::Timer update_timer("uncertainty/update_uncertainty");
  processOpenSet();
  update_timer.Stop();

  uncertainty_timer.Stop();
}

// The raise set is always empty in batch operations.
void UncertaintyIntegrator::processRaiseSet() {
  size_t num_updates = 0u;
  // For the raise set, get all the neighbors, then:
  // (1) if the neighbor's parent is the current voxel, add it to the raise
  //     queue.
  // (2) if the neighbor's parent differs, add it to open (we will have to
  //    update our current distances, of course).
  GlobalIndexVector neighbors;
  Neighborhood<>::IndexMatrix neighbor_indices;
  while (!raise_.empty()) {
    const GlobalIndex global_index = raise_.front();
    raise_.pop();

    UncertaintyVoxel* voxel = uncertainty_layer_->getVoxelPtrByGlobalIndex(global_index);
    CHECK_NOTNULL(voxel);

    // Get the global indices of neighbors.
    Neighborhood<>::getFromGlobalIndex(global_index, &neighbor_indices);

    // Go through the neighbors and see if we can update any of them.
    for (unsigned int idx = 0u; idx < neighbor_indices.cols(); ++idx) {
      const GlobalIndex& neighbor_index = neighbor_indices.col(idx);

      UncertaintyVoxel* neighbor_voxel =
          uncertainty_layer_->getVoxelPtrByGlobalIndex(neighbor_index);
      if (neighbor_voxel == nullptr) {
        continue;
      }
      // Don't touch unobserved voxels and can't do anything with fixed
      // voxels.
      if (!neighbor_voxel->observed || neighbor_voxel->fixed) {
        continue;
      }
      SignedIndex direction = (neighbor_index - global_index).cast<int>();
      bool is_neighbors_parent = (neighbor_voxel->parent == -direction);
      if (config_.full_euclidean_distance) {
        Point voxel_parent_direction =
            neighbor_voxel->parent.cast<FloatingPoint>().normalized();
        voxel_parent_direction = Point(std::round(voxel_parent_direction.x()),
                                       std::round(voxel_parent_direction.y()),
                                       std::round(voxel_parent_direction.z()));
        is_neighbors_parent =
            (voxel_parent_direction.cast<int>() == -direction);
      }
      // This will never update fixed voxels as they are their own parents.
      if (is_neighbors_parent) {
        // This is the case where we are the parent of this one, so we
        // should clear it and raise it.
        neighbor_voxel->distance =
            signum(neighbor_voxel->distance) * config_.default_distance_m;
        neighbor_voxel->parent.setZero();
        raise_.push(neighbor_index);
      } else {
        // If it's not in the queue, then add it to open so it can update
        // our weights back.
        if (!neighbor_voxel->in_queue) {
          open_.push(neighbor_index, neighbor_voxel->distance);
          neighbor_voxel->in_queue = true;
        }
      }
    }
    num_updates++;
  }
  VLOG(3) << "[UNCERTAINTY update]: raised " << num_updates << " voxels.";
}

void UncertaintyIntegrator::processOpenSet() {
  size_t num_updates = 0u;
  size_t num_inside = 0u;
  size_t num_outside = 0u;
  size_t num_flipped = 0u;
  Neighborhood<>::IndexMatrix neighbor_indices;

  while (!open_.empty()) {
    GlobalIndex global_index = open_.front();
    open_.pop();

    UncertaintyVoxel* voxel = uncertainty_layer_->getVoxelPtrByGlobalIndex(global_index);
    CHECK_NOTNULL(voxel);
    voxel->in_queue = false;

    // Skip voxels that are unobserved or outside the ranges we care about.
    if (!voxel->observed || voxel->distance >= config_.max_distance_m ||
        voxel->distance <= -config_.max_distance_m) {
      continue;
    }

    Neighborhood<>::getFromGlobalIndex(global_index, &neighbor_indices);

    // Go through the neighbors and see if we can update any of them.
    for (unsigned int idx = 0u; idx < neighbor_indices.cols(); ++idx) {
      const GlobalIndex& neighbor_index = neighbor_indices.col(idx);
      const SignedIndex& direction =
          NeighborhoodLookupTables::kOffsets.col(idx);
      FloatingPoint distance =
          NeighborhoodLookupTables::kDistances[idx] * voxel_size_;

      UncertaintyVoxel* neighbor_voxel =
          uncertainty_layer_->getVoxelPtrByGlobalIndex(neighbor_index);
      if (neighbor_voxel == nullptr) {
        continue;
      }

      // Don't touch unobserved voxels and can't do anything with fixed
      // voxels.
      if (!neighbor_voxel->observed || neighbor_voxel->fixed) {
        continue;
      }

      SignedIndex new_parent = -direction;
      if (config_.full_euclidean_distance) {
        // In this case, the new parent is is actually the parent of the
        // current voxel.
        // And the distance is... Well, complicated.
        new_parent = voxel->parent - direction;
        distance = voxel_size_ * (new_parent.cast<FloatingPoint>().norm() -
                                  voxel->parent.cast<FloatingPoint>().norm());

        if (distance < 0.0) {
          continue;
        }
      }

      // Both are OUTSIDE the surface.
      if (voxel->distance > 0 && neighbor_voxel->distance > 0) {
        if (voxel->distance + distance + config_.min_diff_m <
            neighbor_voxel->distance) {
          num_updates++;
          num_outside++;
          neighbor_voxel->distance = voxel->distance + distance;
          // Also update parent.
          neighbor_voxel->parent = new_parent;
          // Push into the queue if necessary.
          if (config_.multi_queue || !neighbor_voxel->in_queue) {
            open_.push(neighbor_index, neighbor_voxel->distance);
            neighbor_voxel->in_queue = true;
          }
        }
        // Next case is both INSIDE the surface.
      } else if (voxel->distance <= 0 && neighbor_voxel->distance <= 0) {
        if (voxel->distance - distance - config_.min_diff_m >
            neighbor_voxel->distance) {
          num_updates++;
          num_inside++;
          neighbor_voxel->distance = voxel->distance - distance;
          // Also update parent.
          neighbor_voxel->parent = new_parent;
          // Push into the queue if necessary.
          if (config_.multi_queue || !neighbor_voxel->in_queue) {
            open_.push(neighbor_index, neighbor_voxel->distance);
            neighbor_voxel->in_queue = true;
          }
        }
        // Final case is if the signs are different.
      } else {
        const FloatingPoint potential_distance =
            voxel->distance - signum(voxel->distance) * distance;
        if (std::abs(potential_distance - neighbor_voxel->distance) >
            distance) {
          if (signum(potential_distance) == neighbor_voxel->distance) {
            num_updates++;
            num_flipped++;
            neighbor_voxel->distance = potential_distance;
            // Also update parent.
            neighbor_voxel->parent = new_parent;
            // Push into the queue if necessary.
            if (config_.multi_queue || !neighbor_voxel->in_queue) {
              open_.push(neighbor_index, neighbor_voxel->distance);
              neighbor_voxel->in_queue = true;
            }
          } else {
            num_updates++;
            num_flipped++;
            neighbor_voxel->distance =
                signum(neighbor_voxel->distance) * distance;
            // Also update parent.
            neighbor_voxel->parent = new_parent;
            // Push into the queue if necessary.
            if (config_.multi_queue || !neighbor_voxel->in_queue) {
              open_.push(neighbor_index, neighbor_voxel->distance);
              neighbor_voxel->in_queue = true;
            }
          }
        }
      }
    }
  }

  VLOG(3) << "[UNCERTAINTY update]: made " << num_updates
          << " voxel updates, of which outside: " << num_outside
          << " inside: " << num_inside << " flipped: " << num_flipped;
}

bool UncertaintyIntegrator::updateVoxelFromNeighbors(const GlobalIndex& global_index) {
  UncertaintyVoxel* voxel = uncertainty_layer_->getVoxelPtrByGlobalIndex(global_index);
  CHECK_NOTNULL(voxel);
  // Get the global indices of neighbors.
  Neighborhood<>::IndexMatrix neighbor_indices;
  Neighborhood<>::getFromGlobalIndex(global_index, &neighbor_indices);

  // Go through the neighbors and see if we can update any of them.
  for (unsigned int idx = 0u; idx < neighbor_indices.cols(); ++idx) {
    const GlobalIndex& neighbor_index = neighbor_indices.col(idx);
    const FloatingPoint distance = Neighborhood<>::kDistances[idx];

    UncertaintyVoxel* neighbor_voxel =
        uncertainty_layer_->getVoxelPtrByGlobalIndex(neighbor_index);
    if (neighbor_voxel == nullptr) {
      continue;
    }
    if (!neighbor_voxel->observed ||
        neighbor_voxel->distance >= config_.max_distance_m ||
        neighbor_voxel->distance <= -config_.max_distance_m) {
      continue;
    }
    if (signum(neighbor_voxel->distance) == signum(voxel->distance)) {
      if (std::abs(neighbor_voxel->distance) < std::abs(voxel->distance)) {
        voxel->distance =
            neighbor_voxel->distance + signum(voxel->distance) * distance;
        voxel->parent = -(neighbor_index - global_index).cast<int>();
        return true;
      }
    }
  }
  return false;
}

}  // namespace voxblox
