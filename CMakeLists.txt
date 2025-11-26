# =============================================================================
# CMakeLists.txt - Global Auto-Init (ScanContext + RANSAC/FPFH)
# =============================================================================
# Supports both:
#   1. Catkin build: catkin_make (in catkin workspace)
#   2. Standalone build: mkdir build && cd build && cmake .. && make
# =============================================================================

cmake_minimum_required(VERSION 3.16)

project(global_auto_init
    VERSION 1.0.0
    DESCRIPTION "C++ Global Localization: ScanContext + RANSAC/FPFH"
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# C++ Standard
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Build Type
# -----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
    add_compile_definitions(NDEBUG)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
    add_compile_definitions(DEBUG_MODE)
endif()

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(BUILD_TESTS "Build unit tests" OFF)
option(STANDALONE_BUILD "Force standalone build even if catkin is found" OFF)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")

# PCL (FPFH, RANSAC)
find_package(PCL 1.10 REQUIRED COMPONENTS
    common
    io
    filters
    features
    registration
    kdtree
)
message(STATUS "Found PCL ${PCL_VERSION}")

# yaml-cpp
find_package(yaml-cpp REQUIRED)
message(STATUS "Found yaml-cpp")

# -----------------------------------------------------------------------------
# ROS1 Detection (optional)
# -----------------------------------------------------------------------------
set(CATKIN_BUILD FALSE)
set(ROS1_FOUND FALSE)

# Check if we're in a real catkin workspace (not just standalone with ROS sourced)
# Real catkin build sets CATKIN_WORKSPACES env var
if(DEFINED ENV{CATKIN_WORKSPACES} AND NOT STANDALONE_BUILD)
    find_package(catkin QUIET COMPONENTS
        roscpp
        sensor_msgs
        geometry_msgs
        std_msgs
        pcl_conversions
        pcl_ros
    )

    if(catkin_FOUND)
        message(STATUS "Building in catkin workspace")
        set(CATKIN_BUILD TRUE)
        set(ROS1_FOUND TRUE)

        catkin_package(
            INCLUDE_DIRS include
            LIBRARIES ${PROJECT_NAME} ginit_robot
            CATKIN_DEPENDS roscpp sensor_msgs geometry_msgs std_msgs pcl_conversions pcl_ros
            DEPENDS PCL Eigen3
        )

        add_definitions(-DHAS_ROS1)
        set(ROS1_INCLUDE_DIRS ${catkin_INCLUDE_DIRS})
        set(ROS1_LIBRARIES ${catkin_LIBRARIES})
    endif()
endif()

# Standalone build - try to find ROS via pkg-config
if(NOT CATKIN_BUILD)
    message(STATUS "Building in standalone mode")
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ROSCPP QUIET roscpp)
        pkg_check_modules(SENSOR_MSGS QUIET sensor_msgs)
        pkg_check_modules(GEOMETRY_MSGS QUIET geometry_msgs)
        pkg_check_modules(STD_MSGS QUIET std_msgs)
        pkg_check_modules(PCL_CONVERSIONS QUIET pcl_conversions)
        pkg_check_modules(PCL_ROS QUIET pcl_ros)

        if(ROSCPP_FOUND AND SENSOR_MSGS_FOUND)
            message(STATUS "Found ROS1 via pkg-config (standalone build)")
            set(ROS1_FOUND TRUE)
            add_definitions(-DHAS_ROS1)
            set(ROS1_INCLUDE_DIRS
                ${ROSCPP_INCLUDE_DIRS}
                ${SENSOR_MSGS_INCLUDE_DIRS}
                ${GEOMETRY_MSGS_INCLUDE_DIRS}
                ${STD_MSGS_INCLUDE_DIRS}
                ${PCL_CONVERSIONS_INCLUDE_DIRS}
                ${PCL_ROS_INCLUDE_DIRS}
            )
            set(ROS1_LIBRARIES
                ${ROSCPP_LIBRARIES}
                ${SENSOR_MSGS_LIBRARIES}
                ${GEOMETRY_MSGS_LIBRARIES}
                ${STD_MSGS_LIBRARIES}
                ${PCL_CONVERSIONS_LIBRARIES}
                ${PCL_ROS_LIBRARIES}
            )
            link_directories(
                ${ROSCPP_LIBRARY_DIRS}
                ${SENSOR_MSGS_LIBRARY_DIRS}
            )
        endif()
    endif()

    if(NOT ROS1_FOUND)
        message(WARNING "ROS1 not found. ginit_node will not be built.")
    endif()
endif()

# GoogleTest (for tests) - only for standalone build
if(BUILD_TESTS AND NOT CATKIN_BUILD)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "Fetching GoogleTest...")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

# -----------------------------------------------------------------------------
# Include Directories
# -----------------------------------------------------------------------------
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
)

if(ROS1_FOUND)
    include_directories(${ROS1_INCLUDE_DIRS})
    if(CATKIN_BUILD)
        include_directories(${catkin_INCLUDE_DIRS})
    endif()
endif()

add_definitions(${PCL_DEFINITIONS})

# -----------------------------------------------------------------------------
# Source Files
# -----------------------------------------------------------------------------
set(CORE_SOURCES
    src/core/thread_pool.cpp
    src/core/memory/pool.cpp
    src/core/memory/size.cpp
    src/core/logging/logger.cpp
)

set(ALGORITHM_SOURCES
    src/algorithms/scan_context.cpp
    src/algorithms/fpfh_ransac.cpp
    src/algorithms/npy_loader.cpp
    src/algorithms/database_builder.cpp
)

set(PIPELINE_SOURCES
    src/pipeline/auto_init_pipeline.cpp
)

# -----------------------------------------------------------------------------
# Main Library
# -----------------------------------------------------------------------------
add_library(${PROJECT_NAME} STATIC
    ${CORE_SOURCES}
    ${ALGORITHM_SOURCES}
    ${PIPELINE_SOURCES}
)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        Eigen3::Eigen
        ${PCL_LIBRARIES}
        yaml-cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/src
)

# -----------------------------------------------------------------------------
# ROS Integration (ginit_robot library + ginit_node executable)
# -----------------------------------------------------------------------------
if(ROS1_FOUND)
    message(STATUS "Building ROS integration...")

    # ginit_robot library
    add_library(ginit_robot STATIC src/ros/ginit_robot.cpp)
    target_compile_options(ginit_robot PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(ginit_robot
        PUBLIC
            ${PROJECT_NAME}
            ${ROS1_LIBRARIES}
    )
    target_include_directories(ginit_robot
        PUBLIC
            ${CMAKE_SOURCE_DIR}/include
            ${ROS1_INCLUDE_DIRS}
    )

    if(CATKIN_BUILD)
        # Catkin build: add dependencies for message generation
        add_dependencies(ginit_robot ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
        target_include_directories(ginit_robot PUBLIC ${catkin_INCLUDE_DIRS})
    endif()
    # Standalone build uses pre-generated header in include/global_auto_init/

    # ginit_node executable
    add_executable(ginit_node src/ros/ginit_node.cpp)
    target_compile_options(ginit_node PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(ginit_node PRIVATE ginit_robot)

    if(CATKIN_BUILD)
        add_dependencies(ginit_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
    endif()

    message(STATUS "  - ginit_robot library")
    message(STATUS "  - ginit_node executable")
endif()

# -----------------------------------------------------------------------------
# Tests (standalone build only)
# -----------------------------------------------------------------------------
if(BUILD_TESTS AND NOT CATKIN_BUILD)
    enable_testing()
    file(GLOB_RECURSE TEST_SOURCES tests/*.cpp)
    list(LENGTH TEST_SOURCES TEST_COUNT)

    if(TEST_COUNT GREATER 0)
        foreach(TEST_SOURCE ${TEST_SOURCES})
            get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
            add_executable(${TEST_NAME} ${TEST_SOURCE})
            target_link_libraries(${TEST_NAME}
                PRIVATE
                    ${PROJECT_NAME}
                    GTest::gtest
                    GTest::gtest_main
            )
            add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
        endforeach()
        message(STATUS "Tests: ${TEST_COUNT} files")
    endif()
endif()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  Algorithm: ScanContext + RANSAC/FPFH")
message(STATUS "========================================")
message(STATUS "C++ Standard:    C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  Eigen3:        ${EIGEN3_VERSION_STRING}")
message(STATUS "  PCL:           ${PCL_VERSION}")
message(STATUS "  yaml-cpp:      yes")
message(STATUS "  ROS1:          ${ROS1_FOUND}")
if(CATKIN_BUILD)
    message(STATUS "  Build Mode:    Catkin (use: rosrun global_auto_init ginit_node)")
else()
    message(STATUS "  Build Mode:    Standalone (use: ./ginit_node)")
endif()
message(STATUS "========================================")
message(STATUS "")
