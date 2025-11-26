/**
 * @file npy_loader.cpp
 * @brief NPY file loader implementation.
 *
 * Uses Python subprocess to convert NPY (pickle) to binary format.
 */

#include "auto_init/algorithms/npy_loader.hpp"

#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace auto_init {
namespace algorithms {

namespace {

/**
 * @brief Execute Python script to convert NPY to binary.
 *
 * Creates a temporary binary file that C++ can read directly.
 */
auto ConvertNpyToBinary(const std::string& npy_path,
                        const std::string& bin_path) -> bool {
  // Python script to convert NPY dict to binary format
  // Note: Using string concatenation to avoid raw string literal issues with Python comments
  std::string python_script;
  python_script += "import numpy as np\n";
  python_script += "import struct\n";
  python_script += "import sys\n";
  python_script += "\n";
  python_script += "npy_path = sys.argv[1]\n";
  python_script += "bin_path = sys.argv[2]\n";
  python_script += "\n";
  python_script += "db = np.load(npy_path, allow_pickle=True).item()\n";
  python_script += "\n";
  python_script += "sample_key = list(db.keys())[0]\n";
  python_script += "sample_val = db[sample_key]\n";
  python_script += "\n";
  python_script += "if isinstance(sample_val, dict):\n";
  python_script += "    sc = sample_val['scan_context']\n";
  python_script += "else:\n";
  python_script += "    sc = sample_val\n";
  python_script += "\n";
  python_script += "num_sectors, num_rings = sc.shape\n";
  python_script += "num_entries = len(db)\n";
  python_script += "\n";
  python_script += "print(f'[NPY Loader] Converting {num_entries} entries ({num_sectors}x{num_rings})')\n";
  python_script += "\n";
  python_script += "with open(bin_path, 'wb') as f:\n";
  python_script += "    f.write(struct.pack('I', 0x4E505944))\n";  // "NPYD" magic
  python_script += "    f.write(struct.pack('I', 1))\n";          // version
  python_script += "    f.write(struct.pack('I', num_entries))\n";
  python_script += "    f.write(struct.pack('I', num_sectors))\n";
  python_script += "    f.write(struct.pack('I', num_rings))\n";
  python_script += "\n";
  python_script += "    for (x, y), val in db.items():\n";
  python_script += "        if isinstance(val, dict):\n";
  python_script += "            sc = val['scan_context']\n";
  python_script += "        else:\n";
  python_script += "            sc = val\n";
  python_script += "\n";
  python_script += "        f.write(struct.pack('f', float(x)))\n";
  python_script += "        f.write(struct.pack('f', float(y)))\n";
  python_script += "        f.write(sc.astype(np.float32).tobytes())\n";
  python_script += "\n";
  python_script += "print(f'[NPY Loader] Written to {bin_path}')\n";

  // Write script to temp file
  std::string script_path = "/tmp/npy_converter.py";
  {
    std::ofstream script_file(script_path);
    if (!script_file.is_open()) {
      std::cerr << "[NPY Loader] Failed to create temp script\n";
      return false;
    }
    script_file << python_script;
  }

  // Execute Python script
  std::string cmd =
      "python3 " + script_path + " \"" + npy_path + "\" \"" + bin_path + "\"";

  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "[NPY Loader] Python conversion failed (exit code: " << ret
              << ")\n";
    return false;
  }

  return true;
}

}  // namespace

/**
 * @brief Check if file is C++ binary format (starts with NPYD magic).
 */
auto IsCppBinaryFormat(const std::string& path) -> bool {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  std::uint32_t magic = 0;
  file.read(reinterpret_cast<char*>(&magic), sizeof(magic));

  constexpr std::uint32_t kMagicNumber = 0x4E505944U;  // "NPYD"
  return magic == kMagicNumber;
}

auto LoadNpyDatabase(const std::string& npy_path)
    -> std::vector<SCDatabaseEntry> {
  std::vector<SCDatabaseEntry> entries;

  // Check if NPY file exists
  std::ifstream npy_check(npy_path);
  if (!npy_check.good()) {
    throw std::runtime_error("NPY file not found: " + npy_path);
  }
  npy_check.close();

  std::string bin_path;

  // Check if the .npy file is already in C++ binary format (from DatabaseBuilder)
  if (IsCppBinaryFormat(npy_path)) {
    std::cout << "[NPY Loader] Detected C++ binary format, loading directly\n";
    bin_path = npy_path;
  } else {
    // Python pickle format - need to convert
    bin_path = npy_path + ".cache.bin";

    // Check if cached binary exists
    std::ifstream bin_check(bin_path, std::ios::binary);
    bool need_convert = !bin_check.good();
    bin_check.close();

    if (need_convert) {
      std::cout << "[NPY Loader] Converting " << npy_path << " to binary...\n";
      if (!ConvertNpyToBinary(npy_path, bin_path)) {
        throw std::runtime_error("Failed to convert NPY file: " + npy_path);
      }
    } else {
      std::cout << "[NPY Loader] Using cached binary: " << bin_path << "\n";
    }
  }

  // Read binary file
  std::ifstream file(bin_path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open binary file: " + bin_path);
  }

  // Read header
  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint32_t num_entries = 0;
  std::uint32_t num_sectors = 0;
  std::uint32_t num_rings = 0;

  file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  file.read(reinterpret_cast<char*>(&version), sizeof(version));
  file.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));
  file.read(reinterpret_cast<char*>(&num_sectors), sizeof(num_sectors));
  file.read(reinterpret_cast<char*>(&num_rings), sizeof(num_rings));

  // Validate header
  constexpr std::uint32_t kMagicNumber = 0x4E505944U;  // "NPYD"
  if (magic != kMagicNumber) {
    throw std::runtime_error("Invalid binary file format (magic mismatch)");
  }

  std::cout << "[NPY Loader] Loading " << num_entries << " entries ("
            << num_sectors << "x" << num_rings << ")\n";

  // Read entries
  entries.reserve(num_entries);

  for (std::uint32_t i = 0; i < num_entries; ++i) {
    SCDatabaseEntry entry;

    file.read(reinterpret_cast<char*>(&entry.x), sizeof(entry.x));
    file.read(reinterpret_cast<char*>(&entry.y), sizeof(entry.y));

    // Read row-major data (NumPy/C++ DatabaseBuilder format) and convert to
    // Eigen column-major. This ensures compatibility with both Python-generated
    // and C++-generated databases.
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        temp(num_sectors, num_rings);
    file.read(reinterpret_cast<char*>(temp.data()),
              temp.size() * sizeof(float));
    entry.descriptor = temp;  // Automatic conversion to column-major

    entries.push_back(std::move(entry));
  }

  std::cout << "[NPY Loader] Loaded " << entries.size() << " entries\n";

  return entries;
}

auto GetNpyDatabaseInfo(const std::string& npy_path)
    -> std::tuple<std::uint32_t, std::uint32_t, std::uint32_t> {
  // Try to load just the header from cached binary
  std::string bin_path = npy_path + ".cache.bin";

  std::ifstream file(bin_path, std::ios::binary);
  if (!file.is_open()) {
    // Need to convert first
    if (!ConvertNpyToBinary(npy_path, bin_path)) {
      throw std::runtime_error("Failed to convert NPY file: " + npy_path);
    }
    file.open(bin_path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open binary file: " + bin_path);
    }
  }

  // Read header only
  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint32_t num_entries = 0;
  std::uint32_t num_sectors = 0;
  std::uint32_t num_rings = 0;

  file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  file.read(reinterpret_cast<char*>(&version), sizeof(version));
  file.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));
  file.read(reinterpret_cast<char*>(&num_sectors), sizeof(num_sectors));
  file.read(reinterpret_cast<char*>(&num_rings), sizeof(num_rings));

  return {num_entries, num_sectors, num_rings};
}

}  // namespace algorithms
}  // namespace auto_init
