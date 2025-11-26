/**
 * @file scan_context.cpp
 * @brief ScanContext implementation.
 * @version 1.0.0
 * @date 2025-11-26
 */

#include "auto_init/algorithms/scan_context.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <limits>
#include <stdexcept>
#include <vector>

namespace auto_init {
namespace algorithms {

// ============================================================================
// ScanContext::Implementation - Hidden implementation
// ============================================================================

class ScanContext::Implementation {
 public:
  Implementation(const Params& params, core::ThreadPool& thread_pool)
      : params_(params), thread_pool_(&thread_pool) {}

  // ============================================================================
  // Core - Descriptor Generation
  // ============================================================================

  auto MakeDescriptor(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                      const Eigen::Vector3f& center) const -> Eigen::MatrixXf {
    // Initialize descriptor with zeros
    Eigen::MatrixXf descriptor(params_.num_sectors, params_.num_rings);
    descriptor.setZero();

    // Convert to polar coordinates and fill bins
    // NOTE: Python-compatible implementation
    // - Uses absolute Z value (point.z), NOT relative to center
    // - No height filtering (Python has no min/max height filter)
    for (const auto& point : cloud.points) {
      // Translate X, Y to center (same as Python: points = points - center)
      const float dx = point.x - center.x();
      const float dy = point.y - center.y();
      // Use absolute Z value (Python: sc[s, r] = max(sc[s, r], points[i, 2]))
      const float z = point.z;

      // Polar coordinates (X, Y only)
      const float theta = std::atan2(dy, dx);  // [-π, π]
      const float radius = std::sqrt(dx * dx + dy * dy);

      // Filter by radius only (no height filter - same as Python)
      if (radius > params_.max_radius || radius < 0.01F) {
        continue;
      }

      // Bin indices
      const float normalized_theta =
          (theta + static_cast<float>(M_PI)) / (2.0F * static_cast<float>(M_PI));
      const float normalized_radius = radius / params_.max_radius;

      auto sector = static_cast<std::int32_t>(
          normalized_theta * static_cast<float>(params_.num_sectors));
      auto ring = static_cast<std::int32_t>(
          normalized_radius * static_cast<float>(params_.num_rings));

      // Clamp to valid range
      sector = std::clamp(sector, 0,
                          static_cast<std::int32_t>(params_.num_sectors) - 1);
      ring = std::clamp(ring, 0,
                        static_cast<std::int32_t>(params_.num_rings) - 1);

      // Max-height encoding (absolute Z, same as Python)
      descriptor(sector, ring) = std::max(descriptor(sector, ring), z);
    }

    return descriptor;
  }

  // ============================================================================
  // Database Management
  // ============================================================================

  void LoadDatabase(const std::string& db_path) {
    std::ifstream file(db_path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open database file: " + db_path);
    }

    // Read header
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::uint32_t num_descriptors = 0;
    std::uint32_t num_sectors = 0;
    std::uint32_t num_rings = 0;

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&num_descriptors), sizeof(num_descriptors));
    file.read(reinterpret_cast<char*>(&num_sectors), sizeof(num_sectors));
    file.read(reinterpret_cast<char*>(&num_rings), sizeof(num_rings));

    // Validate header
    constexpr std::uint32_t kMagicNumber = 0x53434442U;  // "SCDB"
    if (magic != kMagicNumber) {
      throw std::runtime_error("Invalid database file format (magic mismatch)");
    }

    if (num_sectors != params_.num_sectors || num_rings != params_.num_rings) {
      throw std::runtime_error(
          "Database dimensions mismatch (expected " +
          std::to_string(params_.num_sectors) + "x" +
          std::to_string(params_.num_rings) + ", got " +
          std::to_string(num_sectors) + "x" + std::to_string(num_rings) + ")");
    }

    // Read descriptors
    descriptors_.clear();
    descriptors_.reserve(num_descriptors);

    for (std::uint32_t i = 0; i < num_descriptors; ++i) {
      Eigen::MatrixXf desc(num_sectors, num_rings);
      file.read(reinterpret_cast<char*>(desc.data()),
                desc.size() * sizeof(float));
      descriptors_.push_back(std::move(desc));
    }

    has_database_ = !descriptors_.empty();
  }

  void AddToDatabase(const Eigen::MatrixXf& descriptor) {
    descriptors_.push_back(descriptor);
    has_database_ = true;
  }

  void SaveDatabase(const std::string& db_path) const {
    std::ofstream file(db_path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to create database file: " + db_path);
    }

    // Write header
    constexpr std::uint32_t kMagicNumber = 0x53434442U;  // "SCDB"
    constexpr std::uint32_t kVersion = 1U;
    const auto num_descriptors = static_cast<std::uint32_t>(descriptors_.size());

    file.write(reinterpret_cast<const char*>(&kMagicNumber), sizeof(kMagicNumber));
    file.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
    file.write(reinterpret_cast<const char*>(&num_descriptors), sizeof(num_descriptors));
    file.write(reinterpret_cast<const char*>(&params_.num_sectors),
               sizeof(params_.num_sectors));
    file.write(reinterpret_cast<const char*>(&params_.num_rings),
               sizeof(params_.num_rings));

    // Write descriptors
    for (const auto& desc : descriptors_) {
      file.write(reinterpret_cast<const char*>(desc.data()),
                 desc.size() * sizeof(float));
    }
  }

  auto GetDatabaseSize() const -> std::uint32_t {
    return static_cast<std::uint32_t>(descriptors_.size());
  }

  void ClearDatabase() {
    descriptors_.clear();
    has_database_ = false;
  }

  auto HasDatabase() const -> bool { return has_database_; }

  // ============================================================================
  // Search Functions
  // ============================================================================

  auto SearchDatabase(const Eigen::MatrixXf& query_descriptor)
      -> std::vector<Candidate> {
    if (!has_database_) {
      throw std::runtime_error("Database is empty. Call LoadDatabase() first.");
    }

    if (thread_pool_ == nullptr) {
      throw std::runtime_error("ThreadPool not available for search.");
    }

    const std::uint32_t db_size = GetDatabaseSize();
    const std::uint32_t num_threads = 8U;  // TODO: Get from ThreadPool
    const std::uint32_t batch_size =
        (db_size + num_threads - 1U) / num_threads;

    // Submit batch jobs
    std::vector<std::future<std::vector<Candidate>>> futures;
    futures.reserve(num_threads);

    for (std::uint32_t i = 0; i < db_size; i += batch_size) {
      const std::uint32_t end = std::min(i + batch_size, db_size);

      futures.push_back(thread_pool_->Submit([this, &query_descriptor, i, end]() {
        return SearchBatch(query_descriptor, i, end);
      }));
    }

    // Collect results
    std::vector<Candidate> all_candidates;
    for (auto& fut : futures) {
      auto batch_candidates = fut.get();
      all_candidates.insert(all_candidates.end(), batch_candidates.begin(),
                            batch_candidates.end());
    }

    // Sort by similarity (descending)
    std::partial_sort(
        all_candidates.begin(),
        all_candidates.begin() +
            std::min(params_.top_k,
                     static_cast<std::uint32_t>(all_candidates.size())),
        all_candidates.end(),
        [](const Candidate& a, const Candidate& b) {
          return a.similarity > b.similarity;
        });

    // Return top-K
    if (all_candidates.size() > params_.top_k) {
      all_candidates.resize(params_.top_k);
    }

    return all_candidates;
  }

  auto FindBestMatch(const Eigen::MatrixXf& query_desc,
                     const Eigen::MatrixXf& db_desc) const
      -> std::pair<float, float> {
    float best_sim = -1.0F;
    std::int32_t best_shift = 0;

    // Pre-compute norms
    const float query_norm = query_desc.norm();
    const float db_norm = db_desc.norm();

    if (query_norm < 1e-6F || db_norm < 1e-6F) {
      return {0.0F, 0.0F};
    }

    // Try all circular shifts
    for (std::uint32_t shift = 0; shift < params_.num_sectors; ++shift) {
      float dot_product = 0.0F;

      // Compute dot product with circular shift
      for (std::uint32_t ring = 0; ring < params_.num_rings; ++ring) {
        for (std::uint32_t sector = 0; sector < params_.num_sectors; ++sector) {
          const auto shifted_sector =
              static_cast<std::uint32_t>((sector + shift) % params_.num_sectors);
          dot_product += query_desc(shifted_sector, ring) * db_desc(sector, ring);
        }
      }

      // Cosine similarity
      const float sim = dot_product / (query_norm * db_norm);

      if (sim > best_sim) {
        best_sim = sim;
        best_shift = static_cast<std::int32_t>(shift);
      }
    }

    // Convert shift to yaw angle
    const float yaw = static_cast<float>(best_shift) * 2.0F *
                      static_cast<float>(M_PI) /
                      static_cast<float>(params_.num_sectors);

    return {yaw, best_sim};
  }

  auto GetDescriptorDimensions() const
      -> std::pair<std::uint32_t, std::uint32_t> {
    return {params_.num_sectors, params_.num_rings};
  }

 private:
  auto SearchBatch(const Eigen::MatrixXf& query_descriptor,
                   std::uint32_t start_idx, std::uint32_t end_idx)
      -> std::vector<Candidate> {
    std::vector<Candidate> batch_candidates;
    batch_candidates.reserve(end_idx - start_idx);

    for (std::uint32_t i = start_idx; i < end_idx; ++i) {
      auto [yaw, similarity] = FindBestMatch(query_descriptor, descriptors_[i]);

      Candidate candidate;
      candidate.db_index = i;
      candidate.yaw = yaw;
      candidate.similarity = similarity;

      batch_candidates.push_back(candidate);
    }

    return batch_candidates;
  }

  Params params_;
  core::ThreadPool* thread_pool_;
  std::vector<Eigen::MatrixXf> descriptors_;
  bool has_database_{false};
};

// ============================================================================
// ScanContext - Public interface forwarding to Implementation
// ============================================================================

ScanContext::ScanContext(const Params& params, core::ThreadPool& thread_pool)
    : impl_(std::make_unique<Implementation>(params, thread_pool)) {}

ScanContext::~ScanContext() = default;

ScanContext::ScanContext(ScanContext&&) noexcept = default;
auto ScanContext::operator=(ScanContext&&) noexcept -> ScanContext& = default;

auto ScanContext::MakeDescriptor(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                  const Eigen::Vector3f& center) const
    -> Eigen::MatrixXf {
  return impl_->MakeDescriptor(cloud, center);
}

void ScanContext::LoadDatabase(const std::string& db_path) {
  impl_->LoadDatabase(db_path);
}

void ScanContext::AddToDatabase(const Eigen::MatrixXf& descriptor) {
  impl_->AddToDatabase(descriptor);
}

void ScanContext::SaveDatabase(const std::string& db_path) const {
  impl_->SaveDatabase(db_path);
}

auto ScanContext::GetDatabaseSize() const -> std::uint32_t {
  return impl_->GetDatabaseSize();
}

void ScanContext::ClearDatabase() { impl_->ClearDatabase(); }

auto ScanContext::HasDatabase() const -> bool { return impl_->HasDatabase(); }

auto ScanContext::SearchDatabase(const Eigen::MatrixXf& query_descriptor)
    -> std::vector<Candidate> {
  return impl_->SearchDatabase(query_descriptor);
}

auto ScanContext::FindBestMatch(const Eigen::MatrixXf& query_desc,
                                 const Eigen::MatrixXf& db_desc) const
    -> std::pair<float, float> {
  return impl_->FindBestMatch(query_desc, db_desc);
}

auto ScanContext::GetDescriptorDimensions() const
    -> std::pair<std::uint32_t, std::uint32_t> {
  return impl_->GetDescriptorDimensions();
}

}  // namespace algorithms
}  // namespace auto_init
