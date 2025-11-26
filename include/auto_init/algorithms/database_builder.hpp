/**
 * @file database_builder.hpp
 * @brief ScanContext database builder using waypoints and connections.
 * @version 1.0.0
 * @date 2025-11-26
 *
 * Multi-threaded database generation using existing ScanContext implementation.
 */

#ifndef AUTO_INIT__ALGORITHMS__DATABASE_BUILDER_HPP_
#define AUTO_INIT__ALGORITHMS__DATABASE_BUILDER_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "auto_init/algorithms/scan_context.hpp"
#include "auto_init/core/thread_pool.hpp"

namespace auto_init {
namespace algorithms {

/**
 * @brief Waypoint data structure.
 */
struct Waypoint {
  int id{0};
  int floor{0};
  float x{0.0F};
  float y{0.0F};
  float z{0.0F};
};

/**
 * @brief Connection between waypoints.
 */
struct Connection {
  int from_id{0};
  int to_id{0};
};

/**
 * @class DatabaseBuilder
 * @brief Builds ScanContext database from map PCD and waypoints.
 *
 * ## Algorithm
 *
 * 1. Load waypoints.csv and connection.csv
 * 2. Sample positions along connections at specified interval
 * 3. For each position:
 *    - Extract local point cloud from map
 *    - Generate ScanContext descriptor
 * 4. Save database as NPY-compatible binary format
 *
 * ## Multi-threading
 *
 * Uses ThreadPool to parallelize descriptor generation.
 */
class DatabaseBuilder {
 public:
  struct Config {
    // Paths
    std::string map_pcd_path;         ///< Map PCD file
    std::string waypoint_csv_path;    ///< waypoints.csv
    std::string connection_csv_path;  ///< connection.csv
    std::string output_path;          ///< Output .npy path

    // Sampling
    float sampling_distance{2.0F};    ///< Sample interval (m)
    float lateral_distance{0.0F};     ///< Lateral offset (m), 0 = center only
    std::vector<int> floors;          ///< Floors to include

    // ScanContext
    ScanContext::Params sc_params;

    // System
    std::uint32_t num_threads{4U};
    float extraction_radius{80.0F};   ///< Local cloud extraction radius (m)
  };

  struct Result {
    bool success{false};
    std::uint32_t num_entries{0U};
    double build_time_ms{0.0};
    std::string message;
  };

  /**
   * @brief Build database.
   *
   * @param config Build configuration
   * @return Result with success status and statistics
   */
  static auto Build(const Config& config) -> Result;

 private:
  /**
   * @brief Load waypoints from CSV.
   */
  static auto LoadWaypoints(const std::string& path, const std::vector<int>& floors)
      -> std::vector<Waypoint>;

  /**
   * @brief Load connections from CSV.
   */
  static auto LoadConnections(const std::string& path) -> std::vector<Connection>;

  /**
   * @brief Sample positions along connections.
   */
  static auto SamplePositions(const std::vector<Waypoint>& waypoints,
                              const std::vector<Connection>& connections,
                              float sampling_distance,
                              float lateral_distance)
      -> std::vector<std::pair<float, float>>;

  /**
   * @brief Save database to NPY-compatible binary format.
   */
  static auto SaveDatabase(const std::string& path,
                           const std::vector<std::pair<float, float>>& positions,
                           const std::vector<Eigen::MatrixXf>& descriptors,
                           std::uint32_t num_sectors,
                           std::uint32_t num_rings) -> bool;
};

}  // namespace algorithms
}  // namespace auto_init

#endif  // AUTO_INIT__ALGORITHMS__DATABASE_BUILDER_HPP_
