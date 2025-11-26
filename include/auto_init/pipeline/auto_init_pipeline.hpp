/**
 * @file auto_init_pipeline.hpp
 * @brief Main pipeline for global localization auto-initialization.
 * @version 1.0.0
 * @date 2025-11-26
 *
 * Pipeline: ScanContext + RANSAC/FPFH
 *
 * Features: Sequential candidate processing with multi-start RANSAC.
 */

#ifndef AUTO_INIT__PIPELINE__AUTO_INIT_PIPELINE_HPP_
#define AUTO_INIT__PIPELINE__AUTO_INIT_PIPELINE_HPP_

#include <memory>
#include <string>
#include <vector>
#include <future>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "auto_init/algorithms/scan_context.hpp"
#include "auto_init/algorithms/fpfh_ransac.hpp"

namespace auto_init {
namespace pipeline {

/**
 * @class AutoInitPipeline
 * @brief Main pipeline: ScanContext + RANSAC/FPFH.
 *
 * ## Algorithm
 *
 * 1. **ScanContext Place Recognition**
 *    - Generate SC descriptor from query scan
 *    - Search NPY database for Top-K matches
 *
 * 2. **RANSAC + FPFH Registration**
 *    - Process candidates (can be parallel)
 *    - Multi-start trials with ICP refinement
 *
 * 3. **Result**
 *    - 2D pose (x, y, yaw) with fitness score
 *
 * ## Usage
 *
 * @code
 * AutoInitPipeline pipeline(config);
 *
 * // Step 1: Get SC candidates (fast, ~2s)
 * auto candidates = pipeline.GetSCCandidates(query_scan, 4);
 *
 * // Step 2: Process primary candidate
 * auto result1 = pipeline.ProcessCandidate(query_scan, candidates[0]);
 *
 * // Step 3: If primary fails, process backup candidates sequentially
 * for (size_t i = 1; i < candidates.size(); ++i) {
 *     auto backup_result = pipeline.ProcessCandidate(query_scan, candidates[i], i);
 *     // try with backup...
 * }
 * @endcode
 */
class AutoInitPipeline {
 public:
  /**
   * @brief Pipeline configuration.
   */
  struct Config {
    // Input
    std::string input_source{"external"};  ///< "external", "pcd", "ros1_bag"
    std::string input_path;                ///< Bag/PCD file path
    std::string topic_name;                ///< Topic name

    // Database (Python NPY format)
    std::string db_path;                   ///< ScanContext DB (.npy)
    std::string map_pcd_path;              ///< Map PCD path

    // ScanContext parameters
    algorithms::ScanContext::Params sc_params;

    // Registration parameters (RANSAC + FPFH)
    algorithms::FpfhRansac::Config reg_config;

    // SC Top-K candidates (for robustness)
    std::uint32_t sc_top_k{1U};           ///< Number of SC candidates to try (1 = original behavior)

    // System
    std::uint32_t num_threads{4U};
  };

  /**
   * @brief ScanContext candidate from database search.
   */
  struct SCCandidate {
    double x{0.0};                   ///< X position from DB
    double y{0.0};                   ///< Y position from DB
    float yaw{0.0F};                 ///< Yaw angle (rad)
    float similarity{0.0F};          ///< SC similarity score
    std::uint32_t db_index{0U};      ///< Database entry index
  };

  /**
   * @brief Pipeline result.
   */
  struct Result {
    bool success{false};             ///< Success flag
    std::uint32_t candidate_index{0U}; ///< Which SC candidate was used

    // Pose (2D + yaw)
    double x{0.0};                   ///< X position (m)
    double y{0.0};                   ///< Y position (m)
    double z{0.0};                   ///< Z position (m)
    double yaw{0.0};                 ///< Yaw angle (rad)

    // Confidence
    float fitness{0.0F};             ///< Registration fitness [0, 1]
    float sc_similarity{0.0F};       ///< ScanContext similarity

    // Timing
    double total_time_ms{0.0};
    double sc_time_ms{0.0};
    double reg_time_ms{0.0};

    // Debug
    double sc_x{0.0};                ///< ScanContext initial x
    double sc_y{0.0};                ///< ScanContext initial y
    double sc_yaw{0.0};              ///< ScanContext initial yaw
  };

  explicit AutoInitPipeline(const Config& config);
  ~AutoInitPipeline();

  AutoInitPipeline(const AutoInitPipeline&) = delete;
  auto operator=(const AutoInitPipeline&) -> AutoInitPipeline& = delete;
  AutoInitPipeline(AutoInitPipeline&&) noexcept;
  auto operator=(AutoInitPipeline&&) noexcept -> AutoInitPipeline&;

  /**
   * @brief Run pipeline with external query scan (legacy, processes all candidates).
   */
  auto Run(const pcl::PointCloud<pcl::PointXYZ>& query_scan) -> Result;

  /**
   * @brief Get ScanContext candidates from database (Step 1).
   *
   * Fast operation (~2-3 seconds). Returns Top-K candidates sorted by similarity.
   *
   * @param query_scan Input point cloud
   * @param top_k Number of candidates to return (default: 4)
   * @return Vector of SC candidates sorted by similarity (descending)
   */
  auto GetSCCandidates(const pcl::PointCloud<pcl::PointXYZ>& query_scan,
                       std::uint32_t top_k = 4U) -> std::vector<SCCandidate>;

  /**
   * @brief Process a single SC candidate with RANSAC+ICP (Step 2).
   *
   * Runs multi-start RANSAC trials and ICP refinement for one candidate.
   *
   * @param query_scan Input point cloud
   * @param candidate SC candidate to process
   * @param candidate_index Index of this candidate (for result tracking)
   * @return Result with pose and fitness
   */
  auto ProcessCandidate(const pcl::PointCloud<pcl::PointXYZ>& query_scan,
                        const SCCandidate& candidate,
                        std::uint32_t candidate_index = 0U) -> Result;

  /**
   * @brief Process multiple SC candidates asynchronously (Step 3).
   *
   * Launches parallel processing for backup candidates while primary is being verified.
   *
   * @param query_scan Input point cloud
   * @param candidates All SC candidates
   * @param start_index Starting index (skip primary)
   * @param end_index Ending index (exclusive)
   * @return Vector of futures for each candidate result
   */
  auto ProcessCandidatesAsync(const pcl::PointCloud<pcl::PointXYZ>& query_scan,
                              const std::vector<SCCandidate>& candidates,
                              std::uint32_t start_index,
                              std::uint32_t end_index)
      -> std::vector<std::future<Result>>;

  /**
   * @brief Check if database is loaded.
   */
  [[nodiscard]] auto IsDatabaseLoaded() const -> bool;

  /**
   * @brief Get database size.
   */
  [[nodiscard]] auto GetDatabaseSize() const -> std::uint32_t;

 private:
  class Implementation;
  std::unique_ptr<Implementation> impl_;
};

}  // namespace pipeline
}  // namespace auto_init

#endif  // AUTO_INIT__PIPELINE__AUTO_INIT_PIPELINE_HPP_
