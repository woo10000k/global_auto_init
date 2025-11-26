/**
 * @file ginit_node.cpp
 * @brief ROS1 node for global localization initialization.
 * @version 1.0.0
 * @date 2025-11-26
 *
 * Algorithm: ScanContext + RANSAC/FPFH
 * Features: Sequential candidate processing with automatic retry.
 *
 * One-shot node: receives point cloud, runs localization, publishes result, exits.
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include "auto_init/ros/ginit_robot.hpp"

/**
 * @brief One-shot callback for point cloud processing.
 */
class GInitNode {
 public:
  explicit GInitNode(const std::string& config_path)
      : ginit_(config_path), received_(false) {
    ros::NodeHandle nh;
    auto config = ginit_.getConfig();

    ROS_INFO("[GInit] Subscribing to: %s", config.input_topic.c_str());
    sub_ = nh.subscribe(config.input_topic, 1, &GInitNode::callback, this);
    ROS_INFO("[GInit] Waiting for point cloud...");
  }

  void callback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    if (received_) {
      return;  // One-shot: ignore subsequent messages
    }

    // Data validation
    const int num_points = static_cast<int>(msg->width * msg->height);
    const int min_points = 100000;  // Minimum required points for reliable localization

    if (num_points < min_points) {
      ROS_WARN("[GInit] Incomplete frame: %d points (need >= %d), waiting for next...",
               num_points, min_points);
      return;  // Skip incomplete frames
    }

    // Check for valid point cloud structure
    if (msg->data.empty() || msg->point_step == 0) {
      ROS_WARN("[GInit] Invalid point cloud structure, waiting for next...");
      return;
    }

    // Wait for stable frame (skip first few frames after startup)
    ++frame_count_;
    if (frame_count_ < 3) {
      ROS_INFO("[GInit] Warming up: frame %d/3, waiting for stable data...", frame_count_);
      return;
    }

    received_ = true;

    ROS_INFO("[GInit] Received point cloud (%d points), starting localization...",
             num_points);

    // Run localization pipeline
    bool success = ginit_.run(*msg);
    auto result = ginit_.getLastResult();

    // Print result
    ROS_INFO("============================================");
    ROS_INFO("[GInit] Result: %s", success ? "SUCCESS" : "FAILED");
    ROS_INFO("[GInit] Attempts: %u (candidate #%u)",
             result.attempt_count, result.candidate_used + 1);
    ROS_INFO("[GInit] Pose: (%.2f, %.2f, %.2f) m", result.x, result.y, result.z);
    ROS_INFO("[GInit] Yaw: %.1f deg", result.yaw * 180.0 / M_PI);
    ROS_INFO("[GInit] Fitness: %.3f", result.fitness);
    ROS_INFO("--------------------------------------------");
    ROS_INFO("[GInit] Pipeline time: %.1f ms", result.pipeline_time_ms);
    ROS_INFO("[GInit] Service time: %.1f ms", result.service_time_ms);
    ROS_INFO("[GInit] Total time: %.1f ms", result.total_time_ms);
    ROS_INFO("--------------------------------------------");
    ROS_INFO("[GInit] Service (%s): %s",
             result.service_success ? "OK" : "FAIL",
             result.service_message.c_str());
    ROS_INFO("============================================");

    // Shutdown after processing
    ROS_INFO("[GInit] Done. Shutting down.");
    ros::shutdown();
  }

  [[nodiscard]] auto hasReceived() const -> bool { return received_; }

 private:
  auto_init::ros_integration::GInitRobot ginit_;
  ros::Subscriber sub_;
  bool received_;
  int frame_count_{0};
};

/**
 * @brief Main entry point.
 */
auto main(int argc, char** argv) -> int {
  ros::init(argc, argv, "ginit_node");
  ros::NodeHandle nh("~");

  // Get config path from parameter
  std::string config_path;
  if (!nh.getParam("config", config_path)) {
    ROS_ERROR("[GInit] Missing required parameter: _config:=<path>");
    ROS_ERROR("[GInit] Usage: rosrun auto_init ginit_node _config:=/path/to/config.yaml");
    return 1;
  }

  ROS_INFO("[GInit] Loading config: %s", config_path.c_str());

  try {
    GInitNode node(config_path);

    // Spin until shutdown (triggered by callback)
    ros::spin();

    return 0;
  } catch (const std::exception& e) {
    ROS_ERROR("[GInit] Exception: %s", e.what());
    return 1;
  }
}
