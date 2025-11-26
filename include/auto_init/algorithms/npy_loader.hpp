/**
 * @file npy_loader.hpp
 * @brief NPY file loader for Python ScanContext database.
 *
 * Loads Python-generated .npy files containing ScanContext descriptors.
 * Format: dict {(x, y): scan_context_array}
 */

#ifndef AUTO_INIT__ALGORITHMS__NPY_LOADER_HPP_
#define AUTO_INIT__ALGORITHMS__NPY_LOADER_HPP_

#include <Eigen/Dense>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace auto_init {
namespace algorithms {

/**
 * @brief ScanContext database entry from Python NPY file.
 */
struct SCDatabaseEntry {
  float x{0.0F};                 ///< X coordinate.
  float y{0.0F};                 ///< Y coordinate.
  Eigen::MatrixXf descriptor;    ///< ScanContext descriptor.
};

/**
 * @brief Loads Python ScanContext database from NPY file.
 *
 * Python DB format: np.save(path, {(x,y): sc_array}, allow_pickle=True)
 * SC array shape: (num_sectors, num_rings) e.g., (120, 40)
 *
 * @param npy_path Path to .npy file.
 * @return Vector of database entries.
 * @throws std::runtime_error if file cannot be read.
 */
[[nodiscard]] auto LoadNpyDatabase(const std::string& npy_path)
    -> std::vector<SCDatabaseEntry>;

/**
 * @brief Gets database info without loading all data.
 *
 * @param npy_path Path to .npy file.
 * @return Tuple of (num_entries, num_sectors, num_rings).
 */
[[nodiscard]] auto GetNpyDatabaseInfo(const std::string& npy_path)
    -> std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;

}  // namespace algorithms
}  // namespace auto_init

#endif  // AUTO_INIT__ALGORITHMS__NPY_LOADER_HPP_
