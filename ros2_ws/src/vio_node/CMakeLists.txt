cmake_minimum_required(VERSION 3.8)
project(vio_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# ── ROS 2 dependencies ──────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)

# ── OpenCV (system) ─────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# Eigen for ESKF linear algebra
find_package(Eigen3 REQUIRED)
set(CUDA_TARGET_DIR "/usr/local/cuda/targets/${CMAKE_SYSTEM_PROCESSOR}-linux")
if(NOT EXISTS "${CUDA_TARGET_DIR}")
  set(CUDA_TARGET_DIR "/usr/local/cuda/targets/x86_64-linux")
endif()
find_path(CUDA_INCLUDE_DIR cuda_runtime.h
  PATHS "${CUDA_TARGET_DIR}/include" "/usr/local/cuda/include"
  NO_DEFAULT_PATH)
find_library(CUDA_CUBLAS_LIBRARY cublas
  PATHS "${CUDA_TARGET_DIR}/lib" "/usr/local/cuda/lib64"
  NO_DEFAULT_PATH)
find_library(CUDA_CUDART_LIBRARY cudart
  PATHS "${CUDA_TARGET_DIR}/lib" "/usr/local/cuda/lib64"
  NO_DEFAULT_PATH)
if(NOT CUDA_INCLUDE_DIR OR NOT CUDA_CUBLAS_LIBRARY OR NOT CUDA_CUDART_LIBRARY)
  message(FATAL_ERROR "CUDA/cuBLAS required but not found under /usr/local/cuda")
endif()

# ── NVIDIA VPI 3.x ──────────────────────────────────────────────────────────
# VPI installs to /opt/nvidia/vpi3 on Jetson / WSL VPI dev install.
# find_package looks for vpi3Config.cmake or vpi3-config.cmake.
find_package(vpi 3 REQUIRED)

# ── Executable ──────────────────────────────────────────────────────────────
# Only vio_node.cpp listed here — it #includes vpi_tracker.cpp as a header
# (both live in src/).  If you split them into proper header+impl, add
# src/vpi_tracker.cpp to the list below.
add_executable(vio_executable
  src/vio_node.cpp
  src/vpi_tracker.cpp
)

target_include_directories(vio_executable PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIR}
)

target_link_libraries(vio_executable
  ${OpenCV_LIBRARIES}
  vpi          # NVIDIA VPI 3 shared lib
  Eigen3::Eigen
  ${CUDA_CUBLAS_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
)

ament_target_dependencies(vio_executable
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  cv_bridge
  message_filters
  tf2_ros
)

install(TARGETS vio_executable
  DESTINATION lib/${PROJECT_NAME})

ament_package()
