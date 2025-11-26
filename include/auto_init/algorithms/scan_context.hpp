/**
 * @file scan_context.hpp
 * @brief ScanContext descriptor for place recognition.
 * @version 1.0.0
 * @date 2025-11-26
 */

#ifndef AUTO_INIT__ALGORITHMS__SCAN_CONTEXT_HPP_
#define AUTO_INIT__ALGORITHMS__SCAN_CONTEXT_HPP_

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "auto_init/core/thread_pool.hpp"

namespace auto_init {
namespace algorithms {

/**
 * @brief ScanContext descriptor generator and matcher.
 * @note Supports both DB generation (offline) and place recognition (online).
 */
class ScanContext {
 public:
  /**
   * @brief Configuration parameters.
   */
  struct Params {
    std::uint32_t num_sectors{120U};   ///< Azimuth bins (theta).
    std::uint32_t num_rings{40U};      ///< Radial bins (r).
    float max_radius{80.0F};           ///< Maximum radius (meters).
    std::uint32_t top_k{50U};          ///< Top-K candidates to return.
    float min_height{-5.0F};           ///< Minimum z value for filtering.
    float max_height{10.0F};           ///< Maximum z value for filtering.
  };

  /**
   * @brief Place recognition candidate.
   */
  struct Candidate {
    std::uint32_t db_index{0U};  ///< Database index.
    float yaw{0.0F};             ///< Estimated yaw angle (radians).
    float similarity{0.0F};      ///< Cosine similarity [0, 1].

    auto operator<(const Candidate& other) const -> bool {
      return similarity > other.similarity;  // Descending order
    }
  };

  /**
   * @brief Constructs ScanContext with thread pool.
   * @param params Configuration parameters.
   * @param thread_pool Thread pool for parallel processing.
   * @note ThreadPool required for both DB generation and matching.
   */
  ScanContext(const Params& params, core::ThreadPool& thread_pool);

  ~ScanContext();

  ScanContext(const ScanContext&) = delete;
  auto operator=(const ScanContext&) -> ScanContext& = delete;
  ScanContext(ScanContext&&) noexcept;
  auto operator=(ScanContext&&) noexcept -> ScanContext&;

  // ============================================================================
  // Core Function - Descriptor Generation
  // ============================================================================

  /**
   * @brief Generates ScanContext descriptor from point cloud.
   * @param cloud Input point cloud.
   * @param center Center point for polar coordinate transformation.
   * @return ScanContext descriptor matrix (num_sectors x num_rings).
   * @note Used for both DB generation and query.
   */
  [[nodiscard]] auto MakeDescriptor(
      const pcl::PointCloud<pcl::PointXYZ>& cloud,
      const Eigen::Vector3f& center = Eigen::Vector3f::Zero()) const
      -> Eigen::MatrixXf;

  // ============================================================================
  // Database Management (Optional)
  // ============================================================================

  /**
   * @brief Loads database from binary file.
   * @param db_path Path to sc_db.bin.
   * @throws std::runtime_error if file format is invalid.
   */
  void LoadDatabase(const std::string& db_path);

  /**
   * @brief Adds descriptor to in-memory database.
   * @param descriptor ScanContext descriptor to add.
   * @note For incremental DB building.
   */
  void AddToDatabase(const Eigen::MatrixXf& descriptor);

  /**
   * @brief Saves database to binary file.
   * @param db_path Path to save sc_db.bin.
   */
  void SaveDatabase(const std::string& db_path) const;

  /**
   * @brief Gets database size.
   * @return Number of descriptors in database.
   */
  [[nodiscard]] auto GetDatabaseSize() const -> std::uint32_t;

  /**
   * @brief Clears database.
   */
  void ClearDatabase();

  /**
   * @brief Checks if database is loaded.
   * @return True if database has at least one descriptor.
   */
  [[nodiscard]] auto HasDatabase() const -> bool;

  // ============================================================================
  // Search Functions (Requires DB + ThreadPool)
  // ============================================================================

  /**
   * @brief Searches database for top-K candidates.
   * @param query_descriptor Query ScanContext descriptor.
   * @return Top-K candidates sorted by similarity (descending).
   * @throws std::runtime_error if database is empty or no thread pool.
   * @note Requires LoadDatabase() or AddToDatabase() first.
   * @note Requires thread pool (passed in constructor).
   */
  [[nodiscard]] auto SearchDatabase(const Eigen::MatrixXf& query_descriptor)
      -> std::vector<Candidate>;

  /**
   * @brief Finds best match between two descriptors.
   * @param query_desc Query descriptor.
   * @param db_desc Database descriptor.
   * @return Pair of (yaw, similarity).
   * @note Tries all circular shifts to find best alignment.
   */
  [[nodiscard]] auto FindBestMatch(const Eigen::MatrixXf& query_desc,
                                    const Eigen::MatrixXf& db_desc) const
      -> std::pair<float, float>;

  // ============================================================================
  // Utility
  // ============================================================================

  /**
   * @brief Gets descriptor dimensions.
   * @return Pair of (num_sectors, num_rings).
   */
  [[nodiscard]] auto GetDescriptorDimensions() const
      -> std::pair<std::uint32_t, std::uint32_t>;

 private:
  class Implementation;  ///< Forward declaration for pImpl idiom.
  std::unique_ptr<Implementation> impl_;  ///< Pointer to implementation.
};

}  // namespace algorithms
}  // namespace auto_init

#endif  // AUTO_INIT__ALGORITHMS__SCAN_CONTEXT_HPP_
