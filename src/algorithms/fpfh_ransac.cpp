/**
 * @file fpfh_ransac.cpp
 * @brief FPFH + RANSAC global registration implementation.
 *
 * Uses PCL for:
 * - VoxelGrid downsampling
 * - Normal estimation
 * - FPFH feature computation
 * - SampleConsensusPrerejective (RANSAC) registration
 */

#include "auto_init/algorithms/fpfh_ransac.hpp"

#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/icp.h>

#include <chrono>
#include <future>
#include <iostream>
#include <mutex>

#include "auto_init/core/thread_pool.hpp"

namespace auto_init {
namespace algorithms {

// ============================================================================
// Implementation
// ============================================================================

class FpfhRansac::Implementation {
 public:
  explicit Implementation(const Config& config, core::ThreadPool* thread_pool)
      : config_(config), thread_pool_(thread_pool) {}

  /**
   * @brief Extended result with ICP refinement data
   */
  struct TrialResult {
    Result result;
    Eigen::Matrix4f initial_guess;
    // For ICP: store downsampled clouds and RANSAC transform
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_down;
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_down;
    Eigen::Matrix4f T_ransac;
    size_t trial_index{0};
    double icp_time_ms{0.0};
  };

  auto Register(const pcl::PointCloud<pcl::PointXYZ>& source,
                const pcl::PointCloud<pcl::PointXYZ>& target,
                const Eigen::Matrix4f& initial_guess) -> Result {
    auto start = std::chrono::high_resolution_clock::now();

    Result best_result;
    best_result.transformation = initial_guess;
    best_result.fitness = 0.0F;
    best_result.success = false;

    // Generate trial offsets
    std::vector<std::pair<float, float>> offsets;
    offsets.emplace_back(0.0F, 0.0F);  // #1: Original

    if (config_.enable_multi_start && config_.num_trials > 1) {
      float d = config_.trial_offset;
      float d2 = config_.trial_offset_small;
      float diag = d * 0.707F;
      float diag2 = d2 * 0.707F;

      offsets.emplace_back(-d, 0.0F);
      offsets.emplace_back(d, 0.0F);
      offsets.emplace_back(0.0F, -d);
      offsets.emplace_back(0.0F, d);
      offsets.emplace_back(-diag, -diag);
      offsets.emplace_back(-diag, diag);
      offsets.emplace_back(diag, -diag);
      offsets.emplace_back(diag, diag);
      offsets.emplace_back(-d2, 0.0F);
      offsets.emplace_back(d2, 0.0F);
      offsets.emplace_back(0.0F, -d2);
      offsets.emplace_back(0.0F, d2);
      offsets.emplace_back(-diag2, -diag2);
      offsets.emplace_back(-diag2, diag2);
      offsets.emplace_back(diag2, -diag2);
      offsets.emplace_back(diag2, diag2);
    }

    const auto num_trials = std::min(static_cast<size_t>(config_.num_trials),
                                     offsets.size());

    std::cout << "[RANSAC] Multi-start with " << num_trials << " trials"
              << (config_.parallel_trials && thread_pool_ ? " (parallel)" : " (sequential)")
              << "\n";

    // ========== Stage 1: Run RANSAC trials ==========
    std::vector<TrialResult> all_trials(num_trials);
    std::mutex print_mutex;

    auto ransac_start = std::chrono::high_resolution_clock::now();

    if (config_.parallel_trials && thread_pool_ != nullptr) {
      std::vector<std::future<void>> futures;
      futures.reserve(num_trials);

      for (size_t trial = 0; trial < num_trials; ++trial) {
        futures.push_back(thread_pool_->Submit([&, trial]() {
          Eigen::Matrix4f trial_guess = initial_guess;
          trial_guess(0, 3) += offsets[trial].first;
          trial_guess(1, 3) += offsets[trial].second;

          TrialResult trial_data = RunSingleRansacWithData(source, target, trial_guess);
          trial_data.trial_index = trial;
          all_trials[trial] = std::move(trial_data);

          {
            std::lock_guard<std::mutex> lock(print_mutex);
            float trial_x = trial_guess(0, 3);
            float trial_y = trial_guess(1, 3);
            std::cout << "  Trial #" << (trial + 1) << ": pos=(" << trial_x
                      << ", " << trial_y << ") → fitness=" << all_trials[trial].result.fitness
                      << (all_trials[trial].result.success ? " ✓" : " ✗") << "\n";
          }
        }));
      }

      for (auto& f : futures) {
        f.get();
      }
    } else {
      for (size_t trial = 0; trial < num_trials; ++trial) {
        Eigen::Matrix4f trial_guess = initial_guess;
        trial_guess(0, 3) += offsets[trial].first;
        trial_guess(1, 3) += offsets[trial].second;

        float trial_x = trial_guess(0, 3);
        float trial_y = trial_guess(1, 3);
        std::cout << "  Trial #" << (trial + 1) << ": pos=(" << trial_x << ", " << trial_y << ")";

        TrialResult trial_data = RunSingleRansacWithData(source, target, trial_guess);
        trial_data.trial_index = trial;
        all_trials[trial] = std::move(trial_data);

        std::cout << " → fitness=" << all_trials[trial].result.fitness
                  << (all_trials[trial].result.success ? " ✓" : " ✗") << "\n";
      }
    }

    auto ransac_end = std::chrono::high_resolution_clock::now();
    double ransac_time_ms = std::chrono::duration<double, std::milli>(ransac_end - ransac_start).count();

    // Collect successful RANSAC trials
    std::vector<TrialResult*> successful_trials;
    for (auto& t : all_trials) {
      if (t.result.success) {
        successful_trials.push_back(&t);
      }
    }

    std::cout << "\n[RANSAC] Found " << successful_trials.size() << "/" << num_trials
              << " successful hypotheses (time: " << ransac_time_ms << " ms)\n";

    if (successful_trials.empty()) {
      float x = initial_guess(0, 3);
      float y = initial_guess(1, 3);
      float yaw = std::atan2(initial_guess(1, 0), initial_guess(0, 0));
      std::cout << "  All failed, using SC result: pos=(" << x << ", " << y
                << "), yaw=" << (yaw * 180.0F / M_PI) << "°\n";

      auto end = std::chrono::high_resolution_clock::now();
      best_result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
      return best_result;
    }

    // ========== Stage 2: ICP Refinement on successful trials ==========
    std::cout << "\n[ICP] Refining " << successful_trials.size() << " candidates"
              << (config_.parallel_trials && thread_pool_ ? " (parallel)" : " (sequential)") << "\n";

    auto icp_start = std::chrono::high_resolution_clock::now();

    if (config_.parallel_trials && thread_pool_ != nullptr) {
      std::vector<std::future<void>> futures;
      futures.reserve(successful_trials.size());

      for (auto* trial_ptr : successful_trials) {
        futures.push_back(thread_pool_->Submit([&, trial_ptr]() {
          auto t_start = std::chrono::high_resolution_clock::now();
          RunIcpRefinement(*trial_ptr);
          auto t_end = std::chrono::high_resolution_clock::now();
          trial_ptr->icp_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

          {
            std::lock_guard<std::mutex> lock(print_mutex);
            float x = trial_ptr->result.transformation(0, 3);
            float y = trial_ptr->result.transformation(1, 3);
            float yaw = std::atan2(trial_ptr->result.transformation(1, 0),
                                   trial_ptr->result.transformation(0, 0)) * 180.0F / M_PI;
            std::cout << "  Trial #" << (trial_ptr->trial_index + 1)
                      << " [ICP]: pos=(" << x << ", " << y << "), yaw=" << yaw
                      << "°, rmse=" << trial_ptr->result.rmse
                      << " (" << trial_ptr->icp_time_ms << " ms)\n";
          }
        }));
      }

      for (auto& f : futures) {
        f.get();
      }
    } else {
      for (auto* trial_ptr : successful_trials) {
        auto t_start = std::chrono::high_resolution_clock::now();
        RunIcpRefinement(*trial_ptr);
        auto t_end = std::chrono::high_resolution_clock::now();
        trial_ptr->icp_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        float x = trial_ptr->result.transformation(0, 3);
        float y = trial_ptr->result.transformation(1, 3);
        float yaw = std::atan2(trial_ptr->result.transformation(1, 0),
                               trial_ptr->result.transformation(0, 0)) * 180.0F / M_PI;
        std::cout << "  Trial #" << (trial_ptr->trial_index + 1)
                  << " [ICP]: pos=(" << x << ", " << y << "), yaw=" << yaw
                  << "°, rmse=" << trial_ptr->result.rmse
                  << " (" << trial_ptr->icp_time_ms << " ms)\n";
      }
    }

    auto icp_end = std::chrono::high_resolution_clock::now();
    double icp_time_ms = std::chrono::duration<double, std::milli>(icp_end - icp_start).count();

    // ========== Stage 3: Position Consistency Clustering ==========
    // Group trials by position/yaw similarity, select from largest cluster

    struct Cluster {
      std::vector<TrialResult*> members;
      float center_x{0.0F};
      float center_y{0.0F};
      float center_yaw{0.0F};
    };

    std::vector<Cluster> clusters;
    const float pos_threshold = 5.0F;   // meters - trials within this distance are same cluster
    const float yaw_threshold = 30.0F;  // degrees - trials within this yaw are same cluster

    std::cout << "\n[CLUSTER] Position consistency clustering:\n";

    for (auto* t : successful_trials) {
      float x = t->result.transformation(0, 3);
      float y = t->result.transformation(1, 3);
      float yaw = std::atan2(t->result.transformation(1, 0),
                             t->result.transformation(0, 0)) * 180.0F / static_cast<float>(M_PI);

      // Find matching cluster
      bool found_cluster = false;
      for (auto& cluster : clusters) {
        float dx = x - cluster.center_x;
        float dy = y - cluster.center_y;
        float dist = std::sqrt(dx * dx + dy * dy);

        // Yaw difference (handle wraparound)
        float yaw_diff = std::abs(yaw - cluster.center_yaw);
        if (yaw_diff > 180.0F) yaw_diff = 360.0F - yaw_diff;

        if (dist < pos_threshold && yaw_diff < yaw_threshold) {
          cluster.members.push_back(t);
          // Update cluster center (running average)
          size_t n = cluster.members.size();
          cluster.center_x = (cluster.center_x * (n - 1) + x) / n;
          cluster.center_y = (cluster.center_y * (n - 1) + y) / n;
          cluster.center_yaw = (cluster.center_yaw * (n - 1) + yaw) / n;
          found_cluster = true;
          break;
        }
      }

      if (!found_cluster) {
        // Create new cluster
        Cluster new_cluster;
        new_cluster.members.push_back(t);
        new_cluster.center_x = x;
        new_cluster.center_y = y;
        new_cluster.center_yaw = yaw;
        clusters.push_back(std::move(new_cluster));
      }
    }

    // Print cluster info
    for (size_t i = 0; i < clusters.size(); ++i) {
      std::cout << "  Cluster " << (i + 1) << ": " << clusters[i].members.size()
                << " trials, center=(" << clusters[i].center_x << ", "
                << clusters[i].center_y << "), yaw=" << clusters[i].center_yaw << "°\n";
    }

    // Sort clusters by size (largest first)
    std::sort(clusters.begin(), clusters.end(),
              [](const Cluster& a, const Cluster& b) {
                return a.members.size() > b.members.size();
              });

    // Select best cluster (largest)
    // If tie, prefer cluster with higher average fitness
    Cluster* best_cluster = &clusters[0];

    // Within the best cluster, use combined score to select best trial
    std::cout << "\n[CLUSTER] Selected cluster " << 1 << " with "
              << best_cluster->members.size() << " trials\n";

    // ========== Stage 4: Re-rank within best cluster using combined score ==========
    auto& cluster_trials = best_cluster->members;

    // Find max RMSE for normalization
    float max_rmse = 0.0F;
    for (const auto* t : cluster_trials) {
      if (t->result.rmse > max_rmse && t->result.rmse < 500.0F) {  // Exclude failed ICP
        max_rmse = t->result.rmse;
      }
    }
    if (max_rmse < 1.0F) max_rmse = 1.0F;  // Avoid division issues

    // Compute combined score for each trial in best cluster
    std::cout << "\n[SCORE] Re-ranking within cluster by combined score:\n";
    for (auto* t : cluster_trials) {
      float rmse_normalized = t->result.rmse / max_rmse;  // 0 to 1
      float combined_score = t->result.fitness - (rmse_normalized * 0.5F);  // Weight RMSE at 50%

      float x = t->result.transformation(0, 3);
      float y = t->result.transformation(1, 3);
      float yaw = std::atan2(t->result.transformation(1, 0),
                             t->result.transformation(0, 0)) * 180.0F / static_cast<float>(M_PI);
      std::cout << "  Trial #" << (t->trial_index + 1)
                << ": fitness=" << t->result.fitness
                << ", rmse=" << t->result.rmse
                << ", score=" << combined_score
                << " pos=(" << x << ", " << y << "), yaw=" << yaw << "°\n";

      // Store score for sorting
      t->icp_time_ms = combined_score;  // Reuse field for score
    }

    // Sort by combined score (higher is better)
    std::sort(cluster_trials.begin(), cluster_trials.end(),
              [](const TrialResult* a, const TrialResult* b) {
                return a->icp_time_ms > b->icp_time_ms;  // Higher score first
              });

    best_result = cluster_trials[0]->result;

    float x = best_result.transformation(0, 3);
    float y = best_result.transformation(1, 3);
    float yaw = std::atan2(best_result.transformation(1, 0),
                           best_result.transformation(0, 0)) * 180.0F / static_cast<float>(M_PI);

    std::cout << "\n[RESULT] Best: Trial #" << (cluster_trials[0]->trial_index + 1)
              << " from cluster with " << best_cluster->members.size() << " trials\n"
              << "  pos=(" << x << ", " << y << "), yaw=" << yaw
              << "°, fitness=" << best_result.fitness
              << ", rmse=" << best_result.rmse
              << " (ICP total: " << icp_time_ms << " ms)\n";

    auto end = std::chrono::high_resolution_clock::now();
    best_result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return best_result;
  }

  auto GetConfig() const -> Config { return config_; }

 private:
  /**
   * @brief Run single RANSAC registration and return data for ICP
   */
  auto RunSingleRansacWithData(const pcl::PointCloud<pcl::PointXYZ>& source,
                               const pcl::PointCloud<pcl::PointXYZ>& target,
                               const Eigen::Matrix4f& initial_guess) -> TrialResult {
    TrialResult trial;
    trial.initial_guess = initial_guess;
    trial.result.transformation = initial_guess;
    trial.result.fitness = 0.0F;
    trial.result.success = false;
    trial.T_ransac = Eigen::Matrix4f::Identity();

    // 1. Transform source by initial_guess
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_transformed(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(source, *source_transformed, initial_guess);

    // 2. Crop target around initial position
    float crop_x = initial_guess(0, 3);
    float crop_y = initial_guess(1, 3);

    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cropped(
        new pcl::PointCloud<pcl::PointXYZ>);
    target_cropped->reserve(target.size() / 4);

    for (const auto& pt : target.points) {
      float dx = pt.x - crop_x;
      float dy = pt.y - crop_y;
      float dist = std::sqrt(dx * dx + dy * dy);
      if (dist < config_.fov_crop) {
        target_cropped->push_back(pt);
      }
    }

    if (target_cropped->empty()) {
      return trial;
    }

    // 3. Downsample
    trial.source_down.reset(new pcl::PointCloud<pcl::PointXYZ>);
    trial.target_down.reset(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setLeafSize(config_.voxel_size, config_.voxel_size, config_.voxel_size);

    voxel.setInputCloud(source_transformed);
    voxel.filter(*trial.source_down);

    voxel.setInputCloud(target_cropped);
    voxel.filter(*trial.target_down);

    if (trial.source_down->size() < 100 || trial.target_down->size() < 100) {
      return trial;
    }

    // 4. Estimate normals
    pcl::PointCloud<pcl::Normal>::Ptr source_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr target_normals(new pcl::PointCloud<pcl::Normal>);

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_est;
    normal_est.setRadiusSearch(config_.voxel_size * 2.0F);

    normal_est.setInputCloud(trial.source_down);
    normal_est.compute(*source_normals);

    normal_est.setInputCloud(trial.target_down);
    normal_est.compute(*target_normals);

    // 5. Compute FPFH features
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_fpfh(
        new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_fpfh(
        new pcl::PointCloud<pcl::FPFHSignature33>);

    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
    fpfh_est.setRadiusSearch(config_.voxel_size * 5.0F);

    fpfh_est.setInputCloud(trial.source_down);
    fpfh_est.setInputNormals(source_normals);
    fpfh_est.compute(*source_fpfh);

    fpfh_est.setInputCloud(trial.target_down);
    fpfh_est.setInputNormals(target_normals);
    fpfh_est.compute(*target_fpfh);

    // 6. SAC-IA (RANSAC)
    pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
    sac_ia.setInputSource(trial.source_down);
    sac_ia.setSourceFeatures(source_fpfh);
    sac_ia.setInputTarget(trial.target_down);
    sac_ia.setTargetFeatures(target_fpfh);

    sac_ia.setMaximumIterations(static_cast<int>(config_.max_iterations));
    sac_ia.setNumberOfSamples(static_cast<int>(config_.ransac_n));
    sac_ia.setCorrespondenceRandomness(5);
    sac_ia.setSimilarityThreshold(0.9F);
    sac_ia.setMaxCorrespondenceDistance(config_.max_correspondence_distance);
    sac_ia.setInlierFraction(config_.fitness_threshold);

    pcl::PointCloud<pcl::PointXYZ> aligned;
    sac_ia.align(aligned);

    trial.T_ransac = sac_ia.getFinalTransformation();
    float sac_fitness = static_cast<float>(sac_ia.getInliers().size()) /
                        static_cast<float>(trial.source_down->size());

    Eigen::Matrix4f T_final = trial.T_ransac * initial_guess;

    trial.result.fitness = sac_fitness;
    trial.result.success = sac_ia.hasConverged() && (sac_fitness >= config_.fitness_threshold);

    if (trial.result.success) {
      trial.result.transformation = T_final;
    }

    return trial;
  }

  /**
   * @brief Run ICP refinement on a successful RANSAC trial
   */
  void RunIcpRefinement(TrialResult& trial) {
    if (!trial.result.success || !trial.source_down || !trial.target_down) {
      return;
    }

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(trial.source_down);
    icp.setInputTarget(trial.target_down);
    icp.setMaximumIterations(50);
    icp.setMaxCorrespondenceDistance(config_.max_correspondence_distance);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);

    pcl::PointCloud<pcl::PointXYZ> icp_aligned;
    icp.align(icp_aligned, trial.T_ransac);

    if (icp.hasConverged()) {
      Eigen::Matrix4f T_icp = icp.getFinalTransformation();
      trial.result.transformation = T_icp * trial.initial_guess;
      trial.result.rmse = static_cast<float>(icp.getFitnessScore());
    } else {
      // ICP failed, keep RANSAC result but set high RMSE for ranking
      trial.result.rmse = 1000.0F;
    }
  }

  Config config_;
  core::ThreadPool* thread_pool_;
};

// ============================================================================
// Public interface
// ============================================================================

FpfhRansac::FpfhRansac(const Config& config, core::ThreadPool* thread_pool)
    : impl_(std::make_unique<Implementation>(config, thread_pool)) {}

FpfhRansac::~FpfhRansac() = default;

FpfhRansac::FpfhRansac(FpfhRansac&&) noexcept = default;
auto FpfhRansac::operator=(FpfhRansac&&) noexcept -> FpfhRansac& = default;

auto FpfhRansac::Register(const pcl::PointCloud<pcl::PointXYZ>& source,
                          const pcl::PointCloud<pcl::PointXYZ>& target,
                          const Eigen::Matrix4f& initial_guess) -> Result {
  return impl_->Register(source, target, initial_guess);
}

auto FpfhRansac::GetConfig() const -> Config { return impl_->GetConfig(); }

}  // namespace algorithms
}  // namespace auto_init
