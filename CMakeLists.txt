cmake_minimum_required(VERSION 3.26)

set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum supported macOS version" FORCE)

project(mlx_spatial LANGUAGES CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "mlx-spatial currently requires Apple platforms with Metal support")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

set(MLX_SPATIALKIT_METAL_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/native/spatialkit/metal/kernels/texture_bake.metal"
)

nanobind_add_module(
  _native
  STABLE_ABI
  native/spatialkit/cpp/bindings.cpp
  native/spatialkit/cpp/flexi_dual_grid.cpp
  native/spatialkit/cpp/glb_writer.cpp
  native/spatialkit/cpp/inpaint.cpp
  native/spatialkit/cpp/mesh_cleanup.cpp
  native/spatialkit/cpp/mesh_metrics.cpp
  native/spatialkit/cpp/mlx_qem_topology.cpp
  native/spatialkit/cpp/pixal3d_contracts.cpp
  native/spatialkit/cpp/simplify.cpp
  native/spatialkit/cpp/remesh.cpp
  native/spatialkit/cpp/surface_distance.cpp
  native/spatialkit/cpp/uv_metrics.cpp
  native/spatialkit/cpp/uv_unwrap.cpp
  native/spatialkit/metal/texture_bake.mm
)

target_include_directories(_native PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/native/spatialkit/cpp"
)
target_link_libraries(_native PRIVATE
  "-framework Foundation"
  "-framework Metal"
)
install(TARGETS _native LIBRARY DESTINATION mlx_spatial/spatialkit)
install(FILES "${MLX_SPATIALKIT_METAL_SOURCE}" DESTINATION mlx_spatial/spatialkit)
