/**
 * @file ginit_robot.hpp
 * @brief ROS integration library for global localization initialization.
 * @version 1.0.0
 * @date 2025-11-26
 *
 * Library for integrating AutoInitPipeline with ROS1.
 * Handles point cloud conversion, pose publishing, and command execution.
 *
 * Features: Sequential candidate processing with external command trigger.
 * - Top-K candidates from ScanContext
 * - Sequential RANSAC+ICP refinement per candidate
 * - Execute configured command after publishing pose
 * - Process terminates on first successful command response
 */

#ifndef AUTO_INIT__ROS__GINIT_ROBOT_HPP_
#define AUTO_INIT__ROS__GINIT_ROBOT_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <sensor_msgs/PointCloud2.h>

namespace auto_init {
namespace ros_integration {

/**
 * @class GInitRobot
 * @brief Global localization initialization library for ROS.
 *
 * ## Overview
 *
 * Sequential pipeline with external command execution:
 *
 * 1. **SC Search**: Get Top-K candidates (~2-3s)
 * 2. **Sequential Processing**: For each candidate:
 *    - Run RANSAC+FPFH registration with multi-start
 *    - Run ICP refinement
 *    - Publish to /initialpose
 *    - Execute configured command
 *    - If command returns true → terminate successfully
 *    - If command returns false → try next candidate
 * 3. **Final**: If all candidates fail, print "move robot" message
 *
 * ## Usage
 *
 * @code
 * auto_init::ros_integration::GInitRobot ginit("config/robot_config.yaml");
 *
 * void cloudCallback(const sensor_msgs::PointCloud2& msg) {
 *     bool success = ginit.run(msg);
 *     // Automatically retries with next candidate if command fails
 * }
 * @endcode
 */
class GInitRobot {
 public:
  /**
   * @brief Configuration for ROS integration.
   */
  struct Config {
    // ROS settings
    std::string input_topic;          ///< Point cloud topic
    std::string map_frame;            ///< Map frame (e.g., "map_3d")
    std::string base_frame;           ///< Robot base frame

    // Map settings
    std::string launch_file_path;     ///< Launch file for map_3d arg
    std::string map_directory;        ///< PCD files directory

    // Database settings
    std::string waypoint_data_path;   ///< waypoints.csv, connection.csv
    std::vector<int> floors;          ///< Floors to use (e.g., {0, 1})
    float sampling_distance{2.0F};    ///< DB sampling interval (m)
    float lateral_distance{0.0F};     ///< Lateral sampling (m)
    bool use_connections{true};       ///< Use connection.csv

    // Timing
    bool enable_timing{true};         ///< Enable timing measurement
    bool print_timing{true};          ///< Print timing to console

    // Localization trigger settings
    std::string loc_command;          ///< Bash command to execute (e.g., "loc_init")
    int floor{0};                     ///< Floor parameter (set as LOC_FLOOR env var)

    // System settings
    std::uint32_t num_threads{4U};    ///< Number of threads for processing

    // Multi-start RANSAC settings (QuatroMulti style)
    bool enable_multi_start{true};    ///< Enable multi-start trials
    std::uint32_t num_trials{9U};     ///< Number of trials (1, 5, 9, 17)
    float trial_offset{2.0F};         ///< Primary offset distance (m)
    float trial_offset_small{1.0F};   ///< Secondary offset distance (m)
    bool parallel_trials{true};       ///< Run trials in parallel

    // Registration settings
    float fitness_threshold{0.01F};   ///< Min fitness for RANSAC success

    // SC Top-K settings
    std::uint32_t sc_top_k{3U};       ///< Backup candidates for retry
    std::uint32_t max_retries{3U};    ///< Maximum retry count (default: 3)
  };

  /**
   * @brief Result of pipeline execution.
   */
  struct Result {
    bool success{false};              ///< Overall success

    // Timing (ms)
    double pipeline_time_ms{0.0};     ///< Pipeline execution time
    double service_time_ms{0.0};      ///< Command execution time
    double total_time_ms{0.0};        ///< Total time

    // Pose
    Eigen::Matrix4f pose{Eigen::Matrix4f::Identity()};  ///< 4x4 transform
    double x{0.0};                    ///< X position (m)
    double y{0.0};                    ///< Y position (m)
    double z{0.0};                    ///< Z position (m)
    double yaw{0.0};                  ///< Yaw angle (rad)

    // Confidence
    float fitness{0.0F};              ///< Registration fitness [0, 1]

    // Command response
    bool service_success{false};      ///< Command execution success
    std::string service_message;      ///< Command response message

    // Retry info
    std::uint32_t attempt_count{0U};  ///< Number of attempts made (1-4)
    std::uint32_t candidate_used{0U}; ///< Which SC candidate was successful
  };

  /**
   * @brief Construct with YAML config file path.
   *
   * @param config_path Path to robot_config.yaml
   * @throws std::runtime_error If config loading fails
   */
  explicit GInitRobot(const std::string& config_path);

  /**
   * @brief Construct with Config struct.
   *
   * @param config Configuration struct
   */
  explicit GInitRobot(const Config& config);

  /**
   * @brief Destructor.
   */
  ~GInitRobot();

  // Non-copyable, movable
  GInitRobot(const GInitRobot&) = delete;
  auto operator=(const GInitRobot&) -> GInitRobot& = delete;
  GInitRobot(GInitRobot&&) noexcept;
  auto operator=(GInitRobot&&) noexcept -> GInitRobot&;

  /**
   * @brief Run localization pipeline and execute command.
   *
   * 1. Convert PointCloud2 to PCL
   * 2. Run ScanContext to get Top-K candidates
   * 3. For each candidate:
   *    - Run RANSAC+FPFH + ICP refinement
   *    - Publish /initialpose
   *    - Execute configured command
   *    - Return true if command succeeds
   *
   * @param cloud Input point cloud
   * @return true if successful (pipeline + command both succeed)
   */
  auto run(const sensor_msgs::PointCloud2& cloud) -> bool;

  /**
   * @brief Get result of last run() call.
   *
   * @return Result struct with timing, pose, and service response
   */
  auto getLastResult() const -> Result;

  /**
   * @brief Check if database exists for current map.
   *
   * @return true if .npy file exists
   */
  auto isDatabaseReady() const -> bool;

  /**
   * @brief Build database if not exists.
   *
   * Uses waypoints.csv and connection.csv to generate .npy database.
   *
   * @return true if database ready (existing or newly built)
   * @throws std::runtime_error If building fails
   */
  auto ensureDatabase() -> bool;

  /**
   * @brief Get current configuration.
   *
   * @return Config struct
   */
  auto getConfig() const -> Config;

 private:
  class Implementation;
  std::unique_ptr<Implementation> impl_;
};

}  // namespace ros_integration
}  // namespace auto_init

#endif  // AUTO_INIT__ROS__GINIT_ROBOT_HPP_
