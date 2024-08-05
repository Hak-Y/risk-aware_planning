#include "voxblox/core/uncertainty_map.h"

namespace voxblox {

bool UncertaintyMap::getDistanceAtPosition(const Eigen::Vector3d& position,
                                    double* distance) const {
  constexpr bool interpolate = true;
  return getDistanceAtPosition(position, interpolate, distance);
}

bool UncertaintyMap::getDistanceAtPosition(const Eigen::Vector3d& position,
                                    bool interpolate, double* distance) const {
  FloatingPoint distance_fp;
  bool success = interpolator_.getDistance(position.cast<FloatingPoint>(),
                                           &distance_fp, interpolate);
  if (success) {
    *distance = static_cast<double>(distance_fp);
  }
  return success;
}

bool UncertaintyMap::getDistanceAndGradientAtPosition(
    const Eigen::Vector3d& position, double* distance,
    Eigen::Vector3d* gradient) const {
  constexpr bool interpolate = true;
  return getDistanceAndGradientAtPosition(position, interpolate, distance,
                                          gradient);
}

bool UncertaintyMap::getDistanceAndGradientAtPosition(
    const Eigen::Vector3d& position, bool interpolate, double* distance,
    Eigen::Vector3d* gradient) const {
  FloatingPoint distance_fp = 0.0;
  Point gradient_fp = Point::Zero();
  bool use_adaptive = false;
  bool success = false;
  if (use_adaptive) {
    success = interpolator_.getAdaptiveDistanceAndGradient(
        position.cast<FloatingPoint>(), &distance_fp, &gradient_fp);
  } else {
    success = interpolator_.getDistance(position.cast<FloatingPoint>(),
                                        &distance_fp, interpolate);
    success &= interpolator_.getGradient(position.cast<FloatingPoint>(),
                                         &gradient_fp, interpolate);
  }

  *distance = static_cast<double>(distance_fp);
  *gradient = gradient_fp.cast<double>();

  return success;
}

bool UncertaintyMap::isObserved(const Eigen::Vector3d& position) const {
  // Get the block.
  // std::cout << "UncertaintyMap::isObserved0" << std::endl;
  Block<UncertaintyVoxel>::Ptr block_ptr =
      uncertainty_layer_->getBlockPtrByCoordinates(position.cast<FloatingPoint>());
  // std::cout << "UncertaintyMap::isObserved1" << std::endl;
  if (block_ptr) {
    // std::cout<< "am i observed2?" << std::endl;
    const UncertaintyVoxel& voxel =
        block_ptr->getVoxelByCoordinates(position.cast<FloatingPoint>());
        // std::cout<< "am i observed3?" << std::endl;
    return voxel.observed;
  }
  // std::cout<< "am i observed4?" << std::endl;
  return false;
}

// NOTE(mereweth@jpl.nasa.gov) - this function is a convenience function for
// Python bindings. std::exceptions are bound to Python exceptions by pybind11,
// allowing them to be handled in Python code idiomatically.
void UncertaintyMap::batchGetDistanceAtPosition(
    EigenDRef<const Eigen::Matrix<double, 3, Eigen::Dynamic>>& positions,
    Eigen::Ref<Eigen::VectorXd> distances,
    Eigen::Ref<Eigen::VectorXi> observed) const {
  if (distances.size() < positions.cols()) {
    throw std::runtime_error("Distances array smaller than number of queries");
  }

  if (observed.size() < positions.cols()) {
    throw std::runtime_error("Observed array smaller than number of queries");
  }

  for (int i = 0; i < positions.cols(); i++) {
    observed[i] = getDistanceAtPosition(positions.col(i), &distances[i]);
  }
}

// NOTE(mereweth@jpl.nasa.gov) - this function is a convenience function for
// Python bindings. std::exceptions are bound to Python exceptions by pybind11,
// allowing them to be handled in Python code idiomatically.
void UncertaintyMap::batchGetDistanceAndGradientAtPosition(
    EigenDRef<const Eigen::Matrix<double, 3, Eigen::Dynamic>>& positions,
    Eigen::Ref<Eigen::VectorXd> distances,
    EigenDRef<Eigen::Matrix<double, 3, Eigen::Dynamic>>& gradients,
    Eigen::Ref<Eigen::VectorXi> observed) const {
  if (distances.size() < positions.cols()) {
    throw std::runtime_error("Distances array smaller than number of queries");
  }

  if (observed.size() < positions.cols()) {
    throw std::runtime_error("Observed array smaller than number of queries");
  }

  if (gradients.cols() < positions.cols()) {
    throw std::runtime_error("Gradients matrix smaller than number of queries");
  }

  for (int i = 0; i < positions.cols(); i++) {
    Eigen::Vector3d gradient;
    observed[i] = getDistanceAndGradientAtPosition(positions.col(i),
                                                   &distances[i], &gradient);

    gradients.col(i) = gradient;
  }
}

// NOTE(mereweth@jpl.nasa.gov) - this function is a convenience function for
// Python bindings. std::exceptions are bound to Python exceptions by pybind11,
// allowing them to be handled in Python code idiomatically.
void UncertaintyMap::batchIsObserved(
    EigenDRef<const Eigen::Matrix<double, 3, Eigen::Dynamic>>& positions,
    Eigen::Ref<Eigen::VectorXi> observed) const {
  if (observed.size() < positions.cols()) {
    throw std::runtime_error("Observed array smaller than number of queries");
  }

  for (int i = 0; i < positions.cols(); i++) {
    observed[i] = isObserved(positions.col(i));
  }
}

unsigned int UncertaintyMap::coordPlaneSliceGetDistance(
    unsigned int free_plane_index, double free_plane_val,
    EigenDRef<Eigen::Matrix<double, 3, Eigen::Dynamic>>& positions,
    Eigen::Ref<Eigen::VectorXd> distances, unsigned int max_points) const {
  BlockIndexList blocks;
  uncertainty_layer_->getAllAllocatedBlocks(&blocks);

  // Cache layer settings.
  size_t vps = uncertainty_layer_->voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;

  // Temp variables.
  bool did_all_fit = true;
  unsigned int count = 0;

  // TODO(mereweth@jpl.nasa.gov) - store min/max index (per axis) allocated in
  // Layer This extra bookeeping will make this much faster
  for (const BlockIndex& index : blocks) {
    // Iterate over all voxels in said blocks.
    const Block<UncertaintyVoxel>& block = uncertainty_layer_->getBlockByIndex(index);

    Point origin = block.origin();
    if (std::abs(origin(free_plane_index) - free_plane_val) >
        block.block_size()) {
      continue;
    }

    for (size_t linear_index = 0; linear_index < num_voxels_per_block;
         ++linear_index) {
      Point coord = block.computeCoordinatesFromLinearIndex(linear_index);
      const UncertaintyVoxel& voxel = block.getVoxelByLinearIndex(linear_index);
      if (std::abs(coord(free_plane_index) - free_plane_val) <=
          block.voxel_size()) {
        double distance;
        if (voxel.observed) {
          distance = voxel.distance;
        } else {
          continue;
        }

        if (count < positions.cols()) {
          positions.col(count) =
              Eigen::Vector3d(coord.x(), coord.y(), coord.z());
        } else {
          did_all_fit = false;
        }
        if (count < distances.size()) {
          distances(count) = distance;
        } else {
          did_all_fit = false;
        }
        count++;
        if (count >= max_points) {
          return count;
        }
      }
    }
  }

  if (!did_all_fit) {
    throw std::runtime_error(std::string("Unable to store ") +
                             std::to_string(count) + " values.");
  }

  return count;
}

}  // namespace voxblox
