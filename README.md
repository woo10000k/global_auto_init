# Global Auto-Initialization

ROS1 C++ library for automatic global localization of robots.

## Overview

A system that automatically finds the robot's current position on a building-wide map using a single LiDAR scan, when the robot restarts or its position is unknown.

### Why is this needed?

Problems with existing robot systems:
- **Manual coordinate input** required when robot restarts
- ICP localization fails if initial position is wrong
- Operator must estimate position by looking at the map - time-consuming, error-prone

After applying this library:
- **Automatic position estimation**: No manual input required
- **Fast restart**: Position confirmed within ~40-50 seconds
- **Reliable operation**: Automatic retry on failure (up to configurable retries)

---

## How It Works

### Overall Process

| Step | Process | Description | Output |
|:---:|------|------|------|
| 1 | **LiDAR Scan Reception** | Receive point cloud once | ~130,000 points |
| 2 | **ScanContext Search** | Cosine similarity comparison in DB | Top-K candidates (x, y, yaw) |
| 3 | **RANSAC + FPFH** | Feature-based precise matching | Transformation matrix |
| 4 | **ICP Refinement** | Final precision improvement | Refined pose |
| 5 | **Result Publishing** | Publish to `/initialpose` topic | 2D Pose |
| 6 | **Command Execution** | Execute configured command | Success/Failure |
| 7 | **Retry** | Try next candidate on failure | Up to max_retries |

### Sequential Candidate Processing

```
Time ─────────────────────────────────────────────────────────────────────────────────────>

[SC Search]  [Candidate #1 RANSAC+ICP]  [Publish + Command]
                                               │
                                               ├── Success → Process terminates
                                               └── Failure → Try Candidate #2
                                                   │
                                 [Candidate #2 RANSAC+ICP]  [Publish + Command]
                                                            │
                                                            ├── Success → Process terminates
                                                            └── Failure → Try next...
```

- Top-1 candidate processed first with RANSAC+ICP refinement
- After refinement, pose is published to `/initialpose`
- Configured command is executed
- If command returns **true** → process terminates successfully
- If command returns **false** → next candidate is processed
- If all candidates fail → display "robot movement required" message





## Installation

### Requirements

- Ubuntu 20.04
- ROS1 Noetic
- PCL 1.8+
- Eigen3
- yaml-cpp

### Install Dependencies (Ubuntu 20.04 + ROS Noetic)

```bash
sudo apt update
sudo apt install ros-noetic-pcl-ros ros-noetic-pcl-conversions
sudo apt install libeigen3-dev libyaml-cpp-dev
```

### Build

#### Option 1: Catkin Build 

```bash
cd ~/catkin_ws/src
git clone <repository-url>
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

#### Option 2: Standalone Build

```bash
git clone <repository-url>
cd global_auto_init
source /opt/ros/noetic/setup.bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## Usage

### Run (Catkin Build)

```bash
rosrun global_auto_init ginit_node _config:=/path/to/robot_config.yaml
```

### Run (Standalone Build)

```bash
./build/ginit_node _config:=/path/to/robot_config.yaml
```


## Automatic Database Generation

NPY database is automatically generated if not present:

1. **Load Map PCD**: Full building point cloud
2. **Parse Waypoints**: Load waypoints from CSV file
3. **Path Sampling**: Sample positions at regular intervals along connections
4. **Descriptor Generation**: Compute ScanContext descriptor at each position (multi-threaded)
5. **Save NPY**: Python-compatible binary format

Required files:
```
/path/to/waypoints/
├── waypoints.csv     # wpt_id, floor, x, y, z
└── connection.csv    # from, to

/path/to/maps/
├── map_name.pcd      # Map point cloud
└── map_name.npy      # Auto-generated DB by this library
```

## License

**MIT License**

### Open Source Libraries Used

| Library | License | Purpose |
|---------|---------|------|
| PCL (Point Cloud Library) | BSD-3-Clause | Point cloud processing, FPFH, RANSAC |
| Eigen | MPL-2.0 / LGPL | Linear algebra operations |
| yaml-cpp | MIT | YAML config file parsing |
| ROS | BSD | Robot communication framework |

All dependency libraries are commercially usable.

### Algorithm References

| Algorithm | Paper | Note |
|---------|------|------|
| ScanContext | Kim & Kim, IROS 2018 | Place recognition |
| FPFH | Rusu et al., ICRA 2009 | Feature extraction |
| RANSAC | Fischler & Bolles, 1981 | Robust estimation |

---

## Contact

- **Issues**: [https://github.com/woo10000k/global_auto_init/issues](https://github.com/woo10000k/global_auto_init/issues)

For bug reports and feature requests, please use GitHub Issues.

---

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
