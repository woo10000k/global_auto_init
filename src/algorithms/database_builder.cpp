/**
 * @file database_builder.cpp
 * @brief ScanContext database builder implementation.
 * @version 1.0.0
 * @date 2025-11-26
 */

#include "auto_init/algorithms/database_builder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/crop_box.h>

namespace auto_init {
namespace algorithms {

namespace {

/**
 * @brief Parse CSV line into tokens.
 */
auto ParseCsvLine(const std::string& line) -> std::vector<std::string> {
  std::vector<std::string> tokens;
  std::stringstream ss(line);
  std::string token;
  while (std::getline(ss, token, ',')) {
    // Trim whitespace
    size_t start = token.find_first_not_of(" \t\r\n");
    size_t end = token.find_last_not_of(" \t\r\n");
    if (start != std::string::npos && end != std::string::npos) {
      tokens.push_back(token.substr(start, end - start + 1));
    } else {
      tokens.push_back("");
    }
  }
  return tokens;
}

/**
 * @brief Extract local point cloud around a position.
 */
auto ExtractLocalCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& map,
                       float x, float y, float radius)
    -> pcl::PointCloud<pcl::PointXYZ>::Ptr {
  pcl::PointCloud<pcl::PointXYZ>::Ptr local(new pcl::PointCloud<pcl::PointXYZ>);

  pcl::CropBox<pcl::PointXYZ> crop;
  crop.setInputCloud(map);
  crop.setMin(Eigen::Vector4f(x - radius, y - radius, -100.0F, 1.0F));
  crop.setMax(Eigen::Vector4f(x + radius, y + radius, 100.0F, 1.0F));
  crop.filter(*local);

  // Transform to local frame (center at origin for X, Y only)
  // Keep Z as-is (same as Python implementation)
  for (auto& pt : local->points) {
    pt.x -= x;
    pt.y -= y;
    // Z is NOT transformed - use absolute height (same as Python)
  }

  return local;
}

}  // namespace

auto DatabaseBuilder::LoadWaypoints(const std::string& path,
                                    const std::vector<int>& floors)
    -> std::vector<Waypoint> {
  std::vector<Waypoint> waypoints;
  std::ifstream file(path);

  if (!file.is_open()) {
    std::cerr << "[DBBuilder] Cannot open waypoints file: " << path << "\n";
    return waypoints;
  }

  std::string line;
  bool header_skipped = false;

  // Create floor set for fast lookup
  std::set<int> floor_set(floors.begin(), floors.end());

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    // Skip header line
    if (!header_skipped) {
      if (line.find("wpt_id") != std::string::npos ||
          line.find("id") != std::string::npos) {
        header_skipped = true;
        continue;
      }
    }

    auto tokens = ParseCsvLine(line);
    if (tokens.size() < 5) continue;

    try {
      Waypoint wp;
      wp.id = std::stoi(tokens[0]);
      wp.floor = std::stoi(tokens[1]);
      wp.x = std::stof(tokens[2]);
      wp.y = std::stof(tokens[3]);
      wp.z = std::stof(tokens[4]);

      // Filter by floor
      if (floor_set.empty() || floor_set.count(wp.floor) > 0) {
        waypoints.push_back(wp);
      }
    } catch (const std::exception& e) {
      // Skip invalid lines
      continue;
    }
  }

  std::cout << "[DBBuilder] Loaded " << waypoints.size() << " waypoints\n";
  return waypoints;
}

auto DatabaseBuilder::LoadConnections(const std::string& path)
    -> std::vector<Connection> {
  std::vector<Connection> connections;
  std::ifstream file(path);

  if (!file.is_open()) {
    std::cerr << "[DBBuilder] Cannot open connections file: " << path << "\n";
    return connections;
  }

  std::string line;
  bool header_skipped = false;

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    // Skip header line
    if (!header_skipped) {
      if (line.find("from") != std::string::npos ||
          line.find("to") != std::string::npos) {
        header_skipped = true;
        continue;
      }
    }

    auto tokens = ParseCsvLine(line);
    if (tokens.size() < 2) continue;

    try {
      Connection conn;
      conn.from_id = std::stoi(tokens[0]);
      conn.to_id = std::stoi(tokens[1]);
      connections.push_back(conn);
    } catch (const std::exception& e) {
      continue;
    }
  }

  std::cout << "[DBBuilder] Loaded " << connections.size() << " connections\n";
  return connections;
}

auto DatabaseBuilder::SamplePositions(const std::vector<Waypoint>& waypoints,
                                      const std::vector<Connection>& connections,
                                      float sampling_distance,
                                      float lateral_distance)
    -> std::vector<std::pair<float, float>> {
  std::vector<std::pair<float, float>> positions;
  std::set<std::pair<int, int>> added;  // Avoid duplicates (quantized)

  // Build waypoint lookup
  std::unordered_map<int, const Waypoint*> wp_map;
  for (const auto& wp : waypoints) {
    wp_map[wp.id] = &wp;
  }

  auto quantize = [](float x, float y) -> std::pair<int, int> {
    return {static_cast<int>(x * 10), static_cast<int>(y * 10)};  // 0.1m precision
  };

  // Add waypoint positions
  for (const auto& wp : waypoints) {
    auto key = quantize(wp.x, wp.y);
    if (added.find(key) == added.end()) {
      positions.emplace_back(wp.x, wp.y);
      added.insert(key);
    }
  }

  // Sample along connections
  for (const auto& conn : connections) {
    auto it_from = wp_map.find(conn.from_id);
    auto it_to = wp_map.find(conn.to_id);

    if (it_from == wp_map.end() || it_to == wp_map.end()) continue;

    const Waypoint* from = it_from->second;
    const Waypoint* to = it_to->second;

    float dx = to->x - from->x;
    float dy = to->y - from->y;
    float dist = std::sqrt(dx * dx + dy * dy);

    if (dist < 0.1F) continue;  // Skip very short connections

    // Normalize direction
    float nx = dx / dist;
    float ny = dy / dist;

    // Sample along the connection
    int num_samples = static_cast<int>(dist / sampling_distance);
    for (int i = 1; i < num_samples; ++i) {
      float t = static_cast<float>(i) / num_samples;
      float x = from->x + dx * t;
      float y = from->y + dy * t;

      auto key = quantize(x, y);
      if (added.find(key) == added.end()) {
        positions.emplace_back(x, y);
        added.insert(key);
      }

      // Lateral samples (perpendicular to connection)
      if (lateral_distance > 0.0F) {
        float lx = -ny * lateral_distance;
        float ly = nx * lateral_distance;

        auto key_left = quantize(x + lx, y + ly);
        if (added.find(key_left) == added.end()) {
          positions.emplace_back(x + lx, y + ly);
          added.insert(key_left);
        }

        auto key_right = quantize(x - lx, y - ly);
        if (added.find(key_right) == added.end()) {
          positions.emplace_back(x - lx, y - ly);
          added.insert(key_right);
        }
      }
    }
  }

  std::cout << "[DBBuilder] Sampled " << positions.size() << " positions\n";
  return positions;
}

auto DatabaseBuilder::SaveDatabase(const std::string& path,
                                   const std::vector<std::pair<float, float>>& positions,
                                   const std::vector<Eigen::MatrixXf>& descriptors,
                                   std::uint32_t num_sectors,
                                   std::uint32_t num_rings) -> bool {
  if (positions.size() != descriptors.size()) {
    std::cerr << "[DBBuilder] Position/descriptor count mismatch\n";
    return false;
  }

  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "[DBBuilder] Cannot create output file: " << path << "\n";
    return false;
  }

  // Header: magic, version, num_entries, num_sectors, num_rings
  std::uint32_t magic = 0x4E505944U;  // "NPYD"
  std::uint32_t version = 1U;
  std::uint32_t num_entries = static_cast<std::uint32_t>(positions.size());

  file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  file.write(reinterpret_cast<const char*>(&version), sizeof(version));
  file.write(reinterpret_cast<const char*>(&num_entries), sizeof(num_entries));
  file.write(reinterpret_cast<const char*>(&num_sectors), sizeof(num_sectors));
  file.write(reinterpret_cast<const char*>(&num_rings), sizeof(num_rings));

  // Entries: x, y, descriptor
  // NOTE: Eigen uses column-major storage, but NumPy uses row-major.
  // We need to convert to row-major format for Python compatibility.
  for (size_t i = 0; i < positions.size(); ++i) {
    float x = positions[i].first;
    float y = positions[i].second;

    file.write(reinterpret_cast<const char*>(&x), sizeof(x));
    file.write(reinterpret_cast<const char*>(&y), sizeof(y));

    // Convert column-major (Eigen default) to row-major (NumPy compatible)
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        desc_rowmajor = descriptors[i];
    file.write(reinterpret_cast<const char*>(desc_rowmajor.data()),
               desc_rowmajor.size() * sizeof(float));
  }

  std::cout << "[DBBuilder] Saved " << num_entries << " entries to " << path << "\n";
  return true;
}

auto DatabaseBuilder::Build(const Config& config) -> Result {
  auto start = std::chrono::high_resolution_clock::now();
  Result result;

  std::cout << "\n========== Database Builder ==========\n";
  std::cout << "[DBBuilder] Map: " << config.map_pcd_path << "\n";
  std::cout << "[DBBuilder] Output: " << config.output_path << "\n";
  std::cout << "[DBBuilder] Threads: " << config.num_threads << "\n";

  // 1. Load map PCD
  std::cout << "[DBBuilder] Loading map...\n";
  pcl::PointCloud<pcl::PointXYZ>::Ptr map(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile(config.map_pcd_path, *map) < 0) {
    result.message = "Failed to load map PCD: " + config.map_pcd_path;
    return result;
  }
  std::cout << "[DBBuilder] Map loaded: " << map->size() << " points\n";

  // 2. Load waypoints and connections
  auto waypoints = LoadWaypoints(config.waypoint_csv_path, config.floors);
  if (waypoints.empty()) {
    result.message = "No waypoints loaded";
    return result;
  }

  auto connections = LoadConnections(config.connection_csv_path);

  // 3. Sample positions
  auto positions = SamplePositions(waypoints, connections,
                                   config.sampling_distance,
                                   config.lateral_distance);
  if (positions.empty()) {
    result.message = "No positions sampled";
    return result;
  }

  // 4. Initialize ThreadPool and ScanContext
  core::ThreadPool thread_pool(config.num_threads, 1024U);
  ScanContext scan_context(config.sc_params, thread_pool);

  // 5. Generate descriptors (parallel)
  std::cout << "[DBBuilder] Generating " << positions.size() << " descriptors...\n";

  std::vector<Eigen::MatrixXf> descriptors(positions.size());
  std::atomic<size_t> completed{0};
  std::mutex progress_mutex;

  std::vector<std::future<void>> futures;
  futures.reserve(positions.size());

  for (size_t i = 0; i < positions.size(); ++i) {
    futures.push_back(thread_pool.Submit([&, i]() {
      float x = positions[i].first;
      float y = positions[i].second;

      // Extract local cloud
      auto local_cloud = ExtractLocalCloud(map, x, y, config.extraction_radius);

      if (!local_cloud->empty()) {
        // Generate descriptor
        descriptors[i] = scan_context.MakeDescriptor(*local_cloud);
      } else {
        // Empty descriptor
        descriptors[i] = Eigen::MatrixXf::Zero(config.sc_params.num_sectors,
                                                config.sc_params.num_rings);
      }

      // Progress
      size_t done = ++completed;
      if (done % 100 == 0 || done == positions.size()) {
        std::lock_guard<std::mutex> lock(progress_mutex);
        std::cout << "\r[DBBuilder] Progress: " << done << "/" << positions.size()
                  << " (" << (done * 100 / positions.size()) << "%)" << std::flush;
      }
    }));
  }

  // Wait for completion
  for (auto& f : futures) {
    f.get();
  }
  std::cout << "\n";

  // 6. Save database
  if (!SaveDatabase(config.output_path, positions, descriptors,
                    config.sc_params.num_sectors, config.sc_params.num_rings)) {
    result.message = "Failed to save database";
    return result;
  }

  auto end = std::chrono::high_resolution_clock::now();
  result.success = true;
  result.num_entries = static_cast<std::uint32_t>(positions.size());
  result.build_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
  result.message = "Database built successfully";

  std::cout << "[DBBuilder] Completed in " << result.build_time_ms << " ms\n";
  std::cout << "==========================================\n\n";

  return result;
}

}  // namespace algorithms
}  // namespace auto_init
