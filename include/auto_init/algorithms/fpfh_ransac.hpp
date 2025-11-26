/**
 * @file fpfh_ransac.hpp
 * @brief FPFH + RANSAC global registration using PCL.
 *
 * Algorithm:
 * 1. Downsample point clouds
 * 2. Estimate normals
 * 3. Compute FPFH features
 * 4. RANSAC-based registration (PCL SampleConsensusPrerejective)
 * 5. ICP refinement
 */

#ifndef AUTO_INIT__ALGORITHMS__FPFH_RANSAC_HPP_
#define AUTO_INIT__ALGORITHMS__FPFH_RANSAC_HPP_

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <memory>
#include <string>

namespace auto_init {
namespace core {
class ThreadPool;  // Forward declaration
}

namespace algorithms {

/**
 * @brief FPFH + RANSAC global registration using PCL.
 *
 * Uses PCL SampleConsensusPrerejective for feature-based registration,
 * with multi-start trials and ICP refinement for robustness.
 */
class FpfhRansac {
 public:
  /**
   * @brief Configuration parameters.
   */
  struct Config {
    float voxel_size{0.5F};                   ///< Voxel size for downsampling.
    float fov_crop{30.0F};                    ///< FOV radius for map cropping.
    float max_correspondence_distance{2.0F};  ///< RANSAC correspondence dist.
    float fitness_threshold{0.50F};           ///< Min fitness for success.
    std::uint32_t ransac_n{3U};               ///< Min points for RANSAC.
    std::uint32_t max_iterations{100000U};    ///< Max RANSAC iterations.

    // Multi-start parameters (QuatroMulti style)
    bool enable_multi_start{true};      ///< Enable multi-start trials.
    std::uint32_t num_trials{9U};       ///< Number of multi-start trials (1, 5, 9, 17).
    float trial_offset{2.0F};           ///< Primary offset distance (m).
    float trial_offset_small{1.0F};     ///< Secondary offset distance (m).
    bool parallel_trials{true};         ///< Run trials in parallel using ThreadPool.
  };

  /**
   * @brief Registration result.
   */
  struct Result {
    bool success{false};             ///< Success if fitness >= threshold.
    Eigen::Matrix4f transformation;  ///< 4x4 transformation matrix.
    float fitness{0.0F};             ///< Inlier ratio [0, 1].
    float rmse{0.0F};                ///< Root mean square error.
    double time_ms{0.0};             ///< Processing time in ms.
  };

  /**
   * @brief Constructs FPFH+RANSAC registrator.
   * @param config Configuration parameters.
   * @param thread_pool Optional ThreadPool for parallel trials (nullptr = sequential).
   */
  explicit FpfhRansac(const Config& config,
                      core::ThreadPool* thread_pool = nullptr);

  ~FpfhRansac();

  FpfhRansac(const FpfhRansac&) = delete;
  auto operator=(const FpfhRansac&) -> FpfhRansac& = delete;
  FpfhRansac(FpfhRansac&&) noexcept;
  auto operator=(FpfhRansac&&) noexcept -> FpfhRansac&;

  /**
   * @brief Performs global registration.
   *
   * @param source Source point cloud (scan).
   * @param target Target point cloud (map, will be cropped).
   * @param initial_guess Initial transformation guess from ScanContext.
   * @return Registration result.
   */
  [[nodiscard]] auto Register(
      const pcl::PointCloud<pcl::PointXYZ>& source,
      const pcl::PointCloud<pcl::PointXYZ>& target,
      const Eigen::Matrix4f& initial_guess = Eigen::Matrix4f::Identity())
      -> Result;

  /**
   * @brief Gets current configuration.
   */
  [[nodiscard]] auto GetConfig() const -> Config;

 private:
  class Implementation;
  std::unique_ptr<Implementation> impl_;
};

}  // namespace algorithms
}  // namespace auto_init

#endif  // AUTO_INIT__ALGORITHMS__FPFH_RANSAC_HPP_
