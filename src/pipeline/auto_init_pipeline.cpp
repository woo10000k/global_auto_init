/**
 * @file auto_init_pipeline.cpp
 * @brief AutoInitPipeline implementation.
 * @version 1.0.0
 * @date 2025-11-26
 *
 * Algorithm: ScanContext + RANSAC/FPFH
 * Features: Sequential candidate processing with multi-start RANSAC.
 */

#include "auto_init/pipeline/auto_init_pipeline.hpp"

#include <pcl/io/pcd_io.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>

#include "auto_init/algorithms/npy_loader.hpp"
#include "auto_init/algorithms/fpfh_ransac.hpp"
#include "auto_init/core/thread_pool.hpp"

namespace auto_init {
namespace pipeline {

// ============================================================================
// Implementation
// ============================================================================

class AutoInitPipeline::Implementation {
 public:
  explicit Implementation(const Config& config) : config_(config) {
    // Initialize thread pool (threads, queue_capacity)
    thread_pool_ = std::make_unique<core::ThreadPool>(config.num_threads, 1024U);

    // Initialize ScanContext
    scan_context_ =
        std::make_unique<algorithms::ScanContext>(config.sc_params, *thread_pool_);

    // Initialize FPFH+RANSAC with ThreadPool for parallel trials
    fpfh_ransac_ = std::make_unique<algorithms::FpfhRansac>(
        config.reg_config, thread_pool_.get());

    // Load NPY database
    LoadDatabase();

    // Load map PCD
    LoadMap();
  }

  // ==========================================================================
  // New API: GetSCCandidates
  // ==========================================================================
  auto GetSCCandidates(const pcl::PointCloud<pcl::PointXYZ>& query_scan,
                       std::uint32_t top_k) -> std::vector<SCCandidate> {
    auto sc_start = std::chrono::high_resolution_clock::now();

    if (query_scan.empty()) {
      std::cerr << "[Pipeline] Error: Query scan is empty!\n";
      return {};
    }

    // Generate query descriptor
    auto query_desc = scan_context_->MakeDescriptor(query_scan);

    std::cout << "[SC] Searching " << db_entries_.size() << " entries for Top-"
              << top_k << " candidates (parallel)...\n";

    // Internal candidate structure
    struct InternalCandidate {
      float similarity{-1.0F};
      float yaw{0.0F};
      std::uint32_t idx{0U};
      double x{0.0};
      double y{0.0};
    };

    // Parallel search using ThreadPool
    const std::uint32_t db_size = static_cast<std::uint32_t>(db_entries_.size());
    const std::uint32_t num_threads = config_.num_threads;
    const std::uint32_t batch_size = (db_size + num_threads - 1U) / num_threads;

    std::vector<std::vector<InternalCandidate>> thread_candidates(num_threads);
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (std::uint32_t t = 0; t < num_threads; ++t) {
      const std::uint32_t start_idx = t * batch_size;
      const std::uint32_t end_idx = std::min(start_idx + batch_size, db_size);

      if (start_idx >= db_size) break;

      futures.push_back(thread_pool_->Submit([&, t, start_idx, end_idx, top_k]() {
        std::vector<InternalCandidate>& local_candidates = thread_candidates[t];
        local_candidates.reserve(top_k);

        for (std::uint32_t i = start_idx; i < end_idx; ++i) {
          auto [yaw, similarity] =
              scan_context_->FindBestMatch(query_desc, db_entries_[i].descriptor);

          InternalCandidate cand;
          cand.similarity = similarity;
          cand.yaw = yaw;
          cand.idx = i;
          cand.x = db_entries_[i].x;
          cand.y = db_entries_[i].y;

          auto it = std::lower_bound(local_candidates.begin(), local_candidates.end(),
                                     cand, [](const InternalCandidate& a, const InternalCandidate& b) {
                                       return a.similarity > b.similarity;
                                     });
          local_candidates.insert(it, cand);

          if (local_candidates.size() > top_k) {
            local_candidates.pop_back();
          }
        }
      }));
    }

    for (auto& f : futures) {
      f.get();
    }

    // Merge all thread candidates
    std::vector<InternalCandidate> all_candidates;
    all_candidates.reserve(num_threads * top_k);
    for (const auto& tc : thread_candidates) {
      all_candidates.insert(all_candidates.end(), tc.begin(), tc.end());
    }

    std::sort(all_candidates.begin(), all_candidates.end(),
              [](const InternalCandidate& a, const InternalCandidate& b) {
                return a.similarity > b.similarity;
              });

    if (all_candidates.size() > top_k) {
      all_candidates.resize(top_k);
    }

    auto sc_end = std::chrono::high_resolution_clock::now();
    double sc_time_ms = std::chrono::duration<double, std::milli>(sc_end - sc_start).count();

    // Convert to public SCCandidate
    std::vector<SCCandidate> result;
    result.reserve(all_candidates.size());

    std::cout << "[SC] Top-" << all_candidates.size() << " candidates:\n";
    for (size_t i = 0; i < all_candidates.size(); ++i) {
      const auto& c = all_candidates[i];
      std::cout << "  #" << (i + 1) << ": (" << c.x << ", " << c.y
                << "), yaw=" << (c.yaw * 180.0 / M_PI) << " deg"
                << ", similarity=" << c.similarity << "\n";

      SCCandidate sc_cand;
      sc_cand.x = c.x;
      sc_cand.y = c.y;
      sc_cand.yaw = c.yaw;
      sc_cand.similarity = c.similarity;
      sc_cand.db_index = c.idx;
      result.push_back(sc_cand);
    }
    std::cout << "[SC] Time: " << sc_time_ms << " ms\n";

    return result;
  }

  // ==========================================================================
  // New API: ProcessCandidate
  // ==========================================================================
  auto ProcessCandidate(const pcl::PointCloud<pcl::PointXYZ>& query_scan,
                        const SCCandidate& candidate,
                        std::uint32_t candidate_index) -> Result {
    auto start = std::chrono::high_resolution_clock::now();
    Result result;
    result.candidate_index = candidate_index;

    std::cout << "\n[RANSAC] Processing candidate #" << (candidate_index + 1)
              << ": (" << candidate.x << ", " << candidate.y
              << "), yaw=" << (candidate.yaw * 180.0 / M_PI) << "°\n";

    // Store SC info
    result.sc_x = candidate.x;
    result.sc_y = candidate.y;
    result.sc_yaw = candidate.yaw;
    result.sc_similarity = candidate.similarity;

    // Create initial transformation from SC candidate
    Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
    initial_guess(0, 3) = static_cast<float>(candidate.x);
    initial_guess(1, 3) = static_cast<float>(candidate.y);

    float cos_yaw = std::cos(candidate.yaw);
    float sin_yaw = std::sin(candidate.yaw);
    initial_guess(0, 0) = cos_yaw;
    initial_guess(0, 1) = -sin_yaw;
    initial_guess(1, 0) = sin_yaw;
    initial_guess(1, 1) = cos_yaw;

    // Run RANSAC registration
    auto reg_result = fpfh_ransac_->Register(query_scan, *map_cloud_, initial_guess);

    // Extract final pose
    result.x = reg_result.transformation(0, 3);
    result.y = reg_result.transformation(1, 3);
    result.z = reg_result.transformation(2, 3);
    result.yaw = std::atan2(reg_result.transformation(1, 0),
                            reg_result.transformation(0, 0));
    result.fitness = reg_result.fitness;
    result.success = reg_result.success;

    auto end = std::chrono::high_resolution_clock::now();
    result.reg_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.total_time_ms = result.reg_time_ms;

    std::cout << "[RANSAC] Candidate #" << (candidate_index + 1)
              << " result: fitness=" << result.fitness
              << ", success=" << (result.success ? "YES" : "NO")
              << ", time=" << result.reg_time_ms << " ms\n";

    return result;
  }

  // ==========================================================================
  // New API: ProcessCandidatesAsync
  // ==========================================================================
  auto ProcessCandidatesAsync(const pcl::PointCloud<pcl::PointXYZ>& query_scan,
                              const std::vector<SCCandidate>& candidates,
                              std::uint32_t start_index,
                              std::uint32_t end_index)
      -> std::vector<std::future<Result>> {
    std::vector<std::future<Result>> futures;

    end_index = std::min(end_index, static_cast<std::uint32_t>(candidates.size()));

    for (std::uint32_t i = start_index; i < end_index; ++i) {
      // Capture by value to avoid lifetime issues
      auto candidate = candidates[i];
      auto index = i;

      futures.push_back(std::async(std::launch::async,
          [this, &query_scan, candidate, index]() -> Result {
            return ProcessCandidate(query_scan, candidate, index);
          }));
    }

    return futures;
  }

  // ==========================================================================
  // Legacy API: Run (processes all candidates sequentially)
  // ==========================================================================
  auto Run(const pcl::PointCloud<pcl::PointXYZ>& query_scan) -> Result {
    auto total_start = std::chrono::high_resolution_clock::now();
    Result result;

    if (query_scan.empty()) {
      std::cerr << "[Pipeline] Error: Query scan is empty!\n";
      return result;
    }

    std::cout << "\n[Pipeline] Starting localization with " << query_scan.size()
              << " points\n";

    // Stage 1: Get SC candidates
    std::cout << "\n========== Stage 1: ScanContext ==========\n";
    auto sc_start = std::chrono::high_resolution_clock::now();
    auto candidates = GetSCCandidates(query_scan, config_.sc_top_k);
    auto sc_end = std::chrono::high_resolution_clock::now();
    result.sc_time_ms = std::chrono::duration<double, std::milli>(sc_end - sc_start).count();

    if (candidates.empty()) {
      std::cerr << "[Pipeline] Error: No SC candidates found!\n";
      return result;
    }

    result.sc_x = candidates[0].x;
    result.sc_y = candidates[0].y;
    result.sc_yaw = candidates[0].yaw;
    result.sc_similarity = candidates[0].similarity;

    // Stage 2: Process candidates
    std::cout << "\n========== Stage 2: RANSAC + FPFH ==========\n";
    auto reg_start = std::chrono::high_resolution_clock::now();

    Result best_result;
    best_result.fitness = -1.0F;

    for (std::uint32_t i = 0; i < candidates.size(); ++i) {
      auto cand_result = ProcessCandidate(query_scan, candidates[i], i);

      if (cand_result.fitness > best_result.fitness) {
        best_result = cand_result;
      }

      // Early exit if good enough
      if (cand_result.success && cand_result.fitness > 0.8F) {
        std::cout << "[RANSAC] Good result found, skipping remaining candidates\n";
        break;
      }
    }

    auto reg_end = std::chrono::high_resolution_clock::now();
    result.reg_time_ms = std::chrono::duration<double, std::milli>(reg_end - reg_start).count();

    // Copy best result
    result.x = best_result.x;
    result.y = best_result.y;
    result.z = best_result.z;
    result.yaw = best_result.yaw;
    result.fitness = best_result.fitness;
    result.success = best_result.success;
    result.candidate_index = best_result.candidate_index;

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // Print result summary
    std::cout << "\n========== Result ==========\n";
    std::cout << "[Pipeline] Success: " << (result.success ? "YES" : "NO") << "\n";
    std::cout << "[Pipeline] Pose: (" << result.x << ", " << result.y << ")\n";
    std::cout << "[Pipeline] Yaw: " << (result.yaw * 180.0 / M_PI) << " deg\n";
    std::cout << "[Pipeline] Fitness: " << result.fitness << "\n";
    std::cout << "[Pipeline] Time: " << result.total_time_ms << " ms"
              << " (SC: " << result.sc_time_ms << " ms"
              << ", Reg: " << result.reg_time_ms << " ms)\n";

    return result;
  }

  auto IsDatabaseLoaded() const -> bool { return !db_entries_.empty(); }

  auto GetDatabaseSize() const -> std::uint32_t {
    return static_cast<std::uint32_t>(db_entries_.size());
  }

 private:
  void LoadDatabase() {
    if (config_.db_path.empty()) {
      throw std::runtime_error("Database path is empty");
    }

    std::cout << "[Pipeline] Loading NPY database: " << config_.db_path << "\n";
    db_entries_ = algorithms::LoadNpyDatabase(config_.db_path);
    std::cout << "[Pipeline] Loaded " << db_entries_.size() << " entries\n";
  }

  void LoadMap() {
    if (config_.map_pcd_path.empty()) {
      throw std::runtime_error("Map PCD path is empty");
    }

    std::cout << "[Pipeline] Loading map: " << config_.map_pcd_path << "\n";
    map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());

    if (pcl::io::loadPCDFile(config_.map_pcd_path, *map_cloud_) < 0) {
      throw std::runtime_error("Failed to load map PCD: " + config_.map_pcd_path);
    }

    std::cout << "[Pipeline] Map loaded: " << map_cloud_->size() << " points\n";
  }

  Config config_;
  std::unique_ptr<core::ThreadPool> thread_pool_;
  std::unique_ptr<algorithms::ScanContext> scan_context_;
  std::unique_ptr<algorithms::FpfhRansac> fpfh_ransac_;
  std::vector<algorithms::SCDatabaseEntry> db_entries_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;
};

// ============================================================================
// Public interface
// ============================================================================

AutoInitPipeline::AutoInitPipeline(const Config& config)
    : impl_(std::make_unique<Implementation>(config)) {}

AutoInitPipeline::~AutoInitPipeline() = default;

AutoInitPipeline::AutoInitPipeline(AutoInitPipeline&&) noexcept = default;
auto AutoInitPipeline::operator=(AutoInitPipeline&&) noexcept
    -> AutoInitPipeline& = default;

auto AutoInitPipeline::Run(const pcl::PointCloud<pcl::PointXYZ>& query_scan)
    -> Result {
  return impl_->Run(query_scan);
}

auto AutoInitPipeline::GetSCCandidates(const pcl::PointCloud<pcl::PointXYZ>& query_scan,
                                       std::uint32_t top_k)
    -> std::vector<SCCandidate> {
  return impl_->GetSCCandidates(query_scan, top_k);
}

auto AutoInitPipeline::ProcessCandidate(const pcl::PointCloud<pcl::PointXYZ>& query_scan,
                                        const SCCandidate& candidate,
                                        std::uint32_t candidate_index)
    -> Result {
  return impl_->ProcessCandidate(query_scan, candidate, candidate_index);
}

auto AutoInitPipeline::ProcessCandidatesAsync(
    const pcl::PointCloud<pcl::PointXYZ>& query_scan,
    const std::vector<SCCandidate>& candidates,
    std::uint32_t start_index,
    std::uint32_t end_index)
    -> std::vector<std::future<Result>> {
  return impl_->ProcessCandidatesAsync(query_scan, candidates, start_index, end_index);
}

auto AutoInitPipeline::IsDatabaseLoaded() const -> bool {
  return impl_->IsDatabaseLoaded();
}

auto AutoInitPipeline::GetDatabaseSize() const -> std::uint32_t {
  return impl_->GetDatabaseSize();
}

}  // namespace pipeline
}  // namespace auto_init
