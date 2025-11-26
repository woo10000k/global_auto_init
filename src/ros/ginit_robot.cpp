/**
 * @file ginit_robot.cpp
 * @brief Implementation of GInitRobot ROS integration library.
 * @version 1.0.0
 * @date 2025-11-26
 *
 * Algorithm: ScanContext + RANSAC/FPFH
 * Sequential candidate processing with external command trigger.
 */

#include "auto_init/ros/ginit_robot.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <regex>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <cstdlib>  // for system(), setenv()

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <yaml-cpp/yaml.h>

#include "auto_init/pipeline/auto_init_pipeline.hpp"
#include "auto_init/algorithms/database_builder.hpp"

namespace auto_init {
namespace ros_integration {

namespace {

/**
 * @brief Timer utility for measuring execution time.
 */
class ScopedTimer {
 public:
  explicit ScopedTimer(double* output_ms)
      : output_ms_(output_ms),
        start_(std::chrono::high_resolution_clock::now()) {}

  ~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start_);
    if (output_ms_ != nullptr) {
      *output_ms_ = static_cast<double>(duration.count()) / 1000.0;
    }
  }

 private:
  double* output_ms_;
  std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief Extract map name from launch file.
 *
 * Parses launch file to find: <arg name="map_3d" default="XXX"/>
 */
auto extractMapNameFromLaunch(const std::string& launch_path) -> std::string {
  std::ifstream file(launch_path);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open launch file: " + launch_path);
  }

  std::string line;
  std::regex pattern(R"(<arg\s+name=\"map_3d\"\s+default=\"([^\"]+)\")");

  while (std::getline(file, line)) {
    std::smatch match;
    if (std::regex_search(line, match, pattern)) {
      return match[1].str();
    }
  }

  throw std::runtime_error("map_3d arg not found in launch file");
}

}  // namespace

/**
 * @brief Implementation class (pImpl pattern).
 */
class GInitRobot::Implementation {
 public:
  explicit Implementation(const Config& config) : config_(config) {
    // Initialize ROS node handle if not already done
    if (!ros::isInitialized()) {
      int argc = 0;
      char** argv = nullptr;
      ros::init(argc, argv, "ginit_robot", ros::init_options::NoSigintHandler);
    }

    nh_ = std::make_unique<ros::NodeHandle>();

    // Setup publisher
    pose_pub_ = nh_->advertise<geometry_msgs::PoseWithCovarianceStamped>(
        "/initialpose", 1, true);

    // Get map name from launch file
    map_name_ = extractMapNameFromLaunch(config_.launch_file_path);

    // Build pipeline config for new ScanContext + RANSAC/FPFH pipeline
    pipeline_config_.input_source = "external";
    pipeline_config_.db_path = config_.map_directory + "/" + map_name_ + ".npy";
    pipeline_config_.map_pcd_path = config_.map_directory + "/" + map_name_ + ".pcd";
    pipeline_config_.num_threads = config_.num_threads;

    // ScanContext params
    pipeline_config_.sc_params.num_sectors = 120;
    pipeline_config_.sc_params.num_rings = 40;
    pipeline_config_.sc_params.max_radius = 80.0F;

    // Registration params (RANSAC + FPFH, same as Python test_on_robot.py)
    pipeline_config_.reg_config.voxel_size = 0.5F;
    pipeline_config_.reg_config.fov_crop = 30.0F;
    pipeline_config_.reg_config.max_correspondence_distance = 2.0F;
    pipeline_config_.reg_config.fitness_threshold = config_.fitness_threshold;

    // Multi-start params (same as Python QuatroMulti)
    pipeline_config_.reg_config.enable_multi_start = config_.enable_multi_start;
    pipeline_config_.reg_config.num_trials = config_.num_trials;
    pipeline_config_.reg_config.trial_offset = config_.trial_offset;
    pipeline_config_.reg_config.trial_offset_small = config_.trial_offset_small;
    pipeline_config_.reg_config.parallel_trials = config_.parallel_trials;

    // SC Top-K
    pipeline_config_.sc_top_k = config_.sc_top_k;

    if (config_.print_timing) {
      std::cout << "[GInit] Initialized with map: " << map_name_ << "\n";
      std::cout << "[GInit] DB path: " << pipeline_config_.db_path << "\n";
      std::cout << "[GInit] Algorithm: ScanContext + RANSAC/FPFH\n";
    }

    // Ensure database exists (auto-generate if needed)
    if (!ensureDatabase()) {
      throw std::runtime_error("Failed to ensure database for map: " + map_name_);
    }

    // Initialize pipeline (after DB is ready)
    pipeline_ = std::make_unique<pipeline::AutoInitPipeline>(pipeline_config_);
  }

  auto run(const sensor_msgs::PointCloud2& cloud) -> bool {
    auto total_start = std::chrono::high_resolution_clock::now();
    Result result;

    // 1. Convert PointCloud2 to PCL
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(cloud, pcl_cloud);

    if (pcl_cloud.empty()) {
      result.success = false;
      result.service_message = "Empty point cloud";
      last_result_ = result;
      return false;
    }

    std::cout << "\n[GInit] ========== Auto-Init Pipeline v1.0 ==========\n";
    std::cout << "[GInit] Input: " << pcl_cloud.size() << " points\n";

    // ==========================================================================
    // STEP 1: Get SC candidates (fast, ~2-3s)
    // Top-1 (primary) + top_k (backups) = 1 + max_retries
    // ==========================================================================
    std::cout << "\n[GInit] Step 1: ScanContext Search\n";
    const std::uint32_t total_candidates = 1U + config_.sc_top_k;  // 1(primary) + backups
    auto candidates = pipeline_->GetSCCandidates(pcl_cloud, total_candidates);

    if (candidates.empty()) {
      result.success = false;
      result.service_message = "No SC candidates found";
      last_result_ = result;
      return false;
    }

    // ==========================================================================
    // STEP 2: Process primary candidate (#1)
    // ==========================================================================
    std::cout << "\n[GInit] Step 2: Processing Primary Candidate #1\n";
    auto primary_result = pipeline_->ProcessCandidate(pcl_cloud, candidates[0], 0);

    // ==========================================================================
    // STEP 3: Try primary result first
    // ==========================================================================
    std::cout << "\n[GInit] Step 3: Publishing pose & calling service\n";

    // Try primary result IMMEDIATELY
    bool loc_success = tryLocalization(primary_result, result, 1);

    if (loc_success) {
      // Success on first try!
      result.attempt_count = 1;
      result.candidate_used = 0;
      result.success = true;
      std::cout << "[GInit] SUCCESS on attempt #1!\n";
    } else {
      // ==========================================================================
      // STEP 4: Primary failed, process backup candidates SEQUENTIALLY
      // ==========================================================================
      std::cout << "\n[GInit] Step 4: Primary failed, trying backup candidates\n";

      std::uint32_t attempt = 2;
      for (size_t i = 1; i < candidates.size() && attempt <= config_.max_retries + 1; ++i) {
        std::cout << "\n[GInit] Processing backup candidate #" << (i + 1) << "...\n";

        // Process backup candidate
        auto backup_result = pipeline_->ProcessCandidate(pcl_cloud, candidates[i],
                                                          static_cast<std::uint32_t>(i));

        // Skip if RANSAC failed
        if (!backup_result.success) {
          std::cout << "[GInit] Backup #" << (i + 1) << " RANSAC failed (fitness="
                    << backup_result.fitness << "), skipping\n";
          continue;
        }

        std::cout << "[GInit] Retry #" << attempt << " with candidate #" << (i + 1) << "\n";
        loc_success = tryLocalization(backup_result, result, attempt);

        if (loc_success) {
          result.attempt_count = attempt;
          result.candidate_used = static_cast<std::uint32_t>(i);
          result.success = true;
          std::cout << "[GInit] SUCCESS on attempt #" << attempt << "!\n";
          break;
        }
        ++attempt;
      }

      // ==========================================================================
      // STEP 5: All retries failed
      // ==========================================================================
      if (!loc_success) {
        std::cout << "\n[GInit] ======================================\n";
        std::cout << "[GInit] ALL " << (attempt - 1) << " ATTEMPTS FAILED\n";
        std::cout << "[GInit] Please move the robot to a different location\n";
        std::cout << "[GInit] ======================================\n";

        result.success = false;
        result.attempt_count = attempt - 1;
        result.service_message = "All retries failed. Please move robot.";
      }
    }

    // Calculate total time
    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    result.pipeline_time_ms = result.total_time_ms - result.service_time_ms;

    last_result_ = result;

    // Print final summary
    if (config_.print_timing) {
      std::cout << "\n[GInit] ========== Final Result ==========\n";
      std::cout << "[GInit] Success: " << (result.success ? "YES" : "NO") << "\n";
      std::cout << "[GInit] Attempts: " << result.attempt_count << "/" << (config_.max_retries + 1) << "\n";
      std::cout << "[GInit] Candidate used: #" << (result.candidate_used + 1) << "\n";
      std::cout << "[GInit] Pose: (" << result.x << ", " << result.y
                << "), yaw=" << (result.yaw * 180.0 / M_PI) << " deg\n";
      std::cout << "[GInit] Fitness: " << result.fitness << "\n";
      std::cout << "[GInit] Total time: " << result.total_time_ms << " ms\n";
      std::cout << "[GInit] Service: " << result.service_message << "\n";
    }

    return result.success;
  }

  /**
   * @brief Try localization with a pipeline result.
   *
   * 1. Update result struct
   * 2. Publish /initialpose
   * 3. Execute configured command
   *
   * @return true if command returns success
   */
  auto tryLocalization(const pipeline::AutoInitPipeline::Result& pipeline_result,
                       Result& result, std::uint32_t attempt) -> bool {
    std::cout << "[GInit] Attempt #" << attempt << ": ";
    std::cout << "pose=(" << pipeline_result.x << ", " << pipeline_result.y << "), ";
    std::cout << "yaw=" << (pipeline_result.yaw * 180.0 / M_PI) << " deg, ";
    std::cout << "fitness=" << pipeline_result.fitness << "\n";

    // Update result
    result.x = pipeline_result.x;
    result.y = pipeline_result.y;
    result.z = pipeline_result.z;
    result.yaw = pipeline_result.yaw;
    result.fitness = pipeline_result.fitness;

    // Build transformation matrix
    result.pose = Eigen::Matrix4f::Identity();
    result.pose(0, 3) = static_cast<float>(result.x);
    result.pose(1, 3) = static_cast<float>(result.y);
    result.pose(2, 3) = static_cast<float>(result.z);

    float cos_yaw = std::cos(static_cast<float>(result.yaw));
    float sin_yaw = std::sin(static_cast<float>(result.yaw));
    result.pose(0, 0) = cos_yaw;
    result.pose(0, 1) = -sin_yaw;
    result.pose(1, 0) = sin_yaw;
    result.pose(1, 1) = cos_yaw;

    // Publish /initialpose
    publishInitialPose(result);

    // Execute configured command
    double service_time = 0.0;
    bool service_success = false;
    {
      ScopedTimer timer(&service_time);
      service_success = executeLocCommand(result.service_message);
    }
    result.service_time_ms += service_time;
    result.service_success = service_success;

    std::cout << "[GInit] Attempt #" << attempt << " service result: "
              << (service_success ? "SUCCESS" : "FAILED")
              << " (" << result.service_message << ")\n";

    return service_success;
  }

  auto getLastResult() const -> Result {
    return last_result_;
  }

  auto isDatabaseReady() const -> bool {
    std::string db_path = config_.map_directory + "/" + map_name_ + ".npy";
    std::ifstream file(db_path);
    return file.good();
  }

  auto ensureDatabase() -> bool {
    if (isDatabaseReady()) {
      return true;
    }

    std::string db_path = config_.map_directory + "/" + map_name_ + ".npy";
    std::string map_path = config_.map_directory + "/" + map_name_ + ".pcd";

    std::cout << "[GInit] Database not found: " << db_path << "\n";
    std::cout << "[GInit] Attempting to build database using C++ DatabaseBuilder...\n";

    // Check if waypoint data path is configured
    if (config_.waypoint_data_path.empty()) {
      std::cerr << "[GInit] waypoint_data_path not configured. Cannot auto-generate DB.\n";
      return false;
    }

    // Build database using C++ DatabaseBuilder (multi-threaded ScanContext)
    algorithms::DatabaseBuilder::Config builder_config;
    builder_config.map_pcd_path = map_path;
    builder_config.waypoint_csv_path = config_.waypoint_data_path + "/waypoints.csv";
    builder_config.connection_csv_path = config_.waypoint_data_path + "/connection.csv";
    builder_config.output_path = db_path;
    builder_config.sampling_distance = config_.sampling_distance;
    builder_config.lateral_distance = config_.lateral_distance;
    builder_config.floors = config_.floors;
    builder_config.num_threads = pipeline_config_.num_threads;
    builder_config.extraction_radius = pipeline_config_.sc_params.max_radius;

    // ScanContext params (same as pipeline)
    builder_config.sc_params = pipeline_config_.sc_params;

    auto result = algorithms::DatabaseBuilder::Build(builder_config);

    if (!result.success) {
      std::cerr << "[GInit] Database generation failed: " << result.message << "\n";
      return false;
    }

    // Verify database was created
    if (isDatabaseReady()) {
      std::cout << "[GInit] Database created successfully: " << db_path << "\n";
      std::cout << "[GInit] Entries: " << result.num_entries
                << ", Time: " << result.build_time_ms << " ms\n";
      return true;
    }

    std::cerr << "[GInit] Database file not found after generation.\n";
    return false;
  }

  auto getConfig() const -> Config {
    return config_;
  }

 private:
  void publishInitialPose(const Result& result) {
    geometry_msgs::PoseWithCovarianceStamped msg;

    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = config_.map_frame;

    msg.pose.pose.position.x = result.x;
    msg.pose.pose.position.y = result.y;
    msg.pose.pose.position.z = result.z;

    // Quaternion from yaw
    double half_yaw = result.yaw / 2.0;
    msg.pose.pose.orientation.x = 0.0;
    msg.pose.pose.orientation.y = 0.0;
    msg.pose.pose.orientation.z = std::sin(half_yaw);
    msg.pose.pose.orientation.w = std::cos(half_yaw);

    // Covariance based on fitness
    double position_var = (1.0 - result.fitness) * 1.0 + 0.1;
    double orientation_var = (1.0 - result.fitness) * 0.1 + 0.01;

    msg.pose.covariance[0] = position_var;
    msg.pose.covariance[7] = position_var;
    msg.pose.covariance[14] = 0.1;
    msg.pose.covariance[21] = 0.01;
    msg.pose.covariance[28] = 0.01;
    msg.pose.covariance[35] = orientation_var;

    pose_pub_.publish(msg);
  }

  auto executeLocCommand(std::string& message) -> bool {
    if (config_.loc_command.empty()) {
      message = "No localization command configured";
      return false;
    }

    std::cout << "[GInit] Executing: " << config_.loc_command << "\n";

    // Run command in separate thread to avoid blocking/signal issues
    std::string output;
    int status = -1;
    bool completed = false;

    auto cmd_future = std::async(std::launch::async, [this, &output, &status]() -> bool {
      // Set floor as environment variable
      setenv("LOC_FLOOR", std::to_string(config_.floor).c_str(), 1);

      // Execute command (non-interactive)
      std::string cmd = config_.loc_command + " 2>&1";
      FILE* pipe = popen(cmd.c_str(), "r");
      if (pipe == nullptr) {
        return false;
      }

      char buffer[256];
      while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
      }

      int exit_code = pclose(pipe);
      status = WEXITSTATUS(exit_code);
      return true;
    });

    // Wait with timeout (30 seconds)
    auto wait_result = cmd_future.wait_for(std::chrono::seconds(30));
    if (wait_result == std::future_status::timeout) {
      message = "Command timed out (30s)";
      std::cout << "[GInit] Command result: TIMEOUT\n";
      return false;
    }

    completed = cmd_future.get();
    if (!completed) {
      message = "Failed to execute command";
      return false;
    }

    // Trim trailing newline
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
      output.pop_back();
    }

    // Parse output format: "success: True/False\nmessage: ..."
    bool success = false;

    // Look for "success: True" or "success: False" pattern
    std::string output_lower = output;
    std::transform(output_lower.begin(), output_lower.end(), output_lower.begin(), ::tolower);

    if (output_lower.find("success: true") != std::string::npos) {
      success = true;
    } else if (output_lower.find("success: false") != std::string::npos) {
      success = false;
    } else {
      // Fallback: use exit code
      success = (status == 0);
    }

    // Extract message field if present
    size_t msg_pos = output.find("message:");
    if (msg_pos != std::string::npos) {
      message = output.substr(msg_pos + 8);  // skip "message:"
      // Trim whitespace and quotes
      while (!message.empty() && (message.front() == ' ' || message.front() == '"')) {
        message.erase(0, 1);
      }
      while (!message.empty() && (message.back() == '"' || message.back() == '\n')) {
        message.pop_back();
      }
    } else {
      message = output.empty() ? (success ? "Command succeeded" : "Command failed") : output;
    }

    std::cout << "[GInit] Command result: " << (success ? "SUCCESS" : "FAILED")
              << " (" << message << ")\n";

    return success;
  }

  Config config_;
  pipeline::AutoInitPipeline::Config pipeline_config_;
  Result last_result_;
  std::string map_name_;

  std::unique_ptr<ros::NodeHandle> nh_;
  ros::Publisher pose_pub_;

  std::unique_ptr<pipeline::AutoInitPipeline> pipeline_;
};

// ============================================================================
// Public Interface
// ============================================================================

GInitRobot::GInitRobot(const std::string& config_path) {
  YAML::Node yaml = YAML::LoadFile(config_path);

  Config config;

  // ROS settings
  if (yaml["ros"]) {
    config.input_topic = yaml["ros"]["input_topic"].as<std::string>("/ouster/points");
    config.map_frame = yaml["ros"]["map_frame"].as<std::string>("map_3d");
    config.base_frame = yaml["ros"]["base_frame"].as<std::string>("base_link_3d");
  }

  // Map settings
  if (yaml["map"]) {
    config.launch_file_path = yaml["map"]["launch_file"].as<std::string>();
    config.map_directory = yaml["map"]["map_directory"].as<std::string>();
  }

  // Database settings (for auto-generation)
  if (yaml["database"]) {
    config.waypoint_data_path = yaml["database"]["waypoint_data_path"].as<std::string>("");
    if (yaml["database"]["floors"]) {
      for (const auto& floor : yaml["database"]["floors"]) {
        config.floors.push_back(floor.as<int>());
      }
    }
    config.sampling_distance = yaml["database"]["sampling_distance"].as<float>(2.0F);
    config.lateral_distance = yaml["database"]["lateral_distance"].as<float>(0.0F);
    config.use_connections = yaml["database"]["use_connections"].as<bool>(true);
  }

  // Timing settings
  if (yaml["timing"]) {
    config.enable_timing = yaml["timing"]["enable_timing"].as<bool>(true);
    config.print_timing = yaml["timing"]["print_timing"].as<bool>(true);
  }

  // Localization trigger settings
  if (yaml["localization"]) {
    config.loc_command = yaml["localization"]["command"].as<std::string>("");
    config.floor = yaml["localization"]["floor"].as<int>(0);
  }

  // System settings
  if (yaml["system"]) {
    config.num_threads = yaml["system"]["num_threads"].as<std::uint32_t>(4U);
  }

  // Registration settings
  if (yaml["registration"]) {
    config.fitness_threshold = yaml["registration"]["fitness_threshold"].as<float>(0.01F);
  }

  // Multi-start RANSAC settings
  if (yaml["ransac"]) {
    config.enable_multi_start = yaml["ransac"]["enable_multi_start"].as<bool>(true);
    config.num_trials = yaml["ransac"]["num_trials"].as<std::uint32_t>(9U);
    config.trial_offset = yaml["ransac"]["trial_offset"].as<float>(2.0F);
    config.trial_offset_small = yaml["ransac"]["trial_offset_small"].as<float>(1.0F);
    config.parallel_trials = yaml["ransac"]["parallel_trials"].as<bool>(true);
  }

  // SC Top-K settings
  if (yaml["scan_context"]) {
    config.sc_top_k = yaml["scan_context"]["top_k"].as<std::uint32_t>(4U);
  }

  // Retry settings
  if (yaml["retry"]) {
    config.max_retries = yaml["retry"]["max_retries"].as<std::uint32_t>(3U);
  }

  impl_ = std::make_unique<Implementation>(config);
}

GInitRobot::GInitRobot(const Config& config)
    : impl_(std::make_unique<Implementation>(config)) {}

GInitRobot::~GInitRobot() = default;

GInitRobot::GInitRobot(GInitRobot&&) noexcept = default;
auto GInitRobot::operator=(GInitRobot&&) noexcept -> GInitRobot& = default;

auto GInitRobot::run(const sensor_msgs::PointCloud2& cloud) -> bool {
  return impl_->run(cloud);
}

auto GInitRobot::getLastResult() const -> Result {
  return impl_->getLastResult();
}

auto GInitRobot::isDatabaseReady() const -> bool {
  return impl_->isDatabaseReady();
}

auto GInitRobot::ensureDatabase() -> bool {
  return impl_->ensureDatabase();
}

auto GInitRobot::getConfig() const -> Config {
  return impl_->getConfig();
}

}  // namespace ros_integration
}  // namespace auto_init
