cmake_minimum_required(VERSION 3.26)

project(mlx_spatialkit LANGUAGES CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "mlx-spatialkit currently requires Apple platforms with Metal support")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
execute_process(
  COMMAND "${XCRUN_EXECUTABLE}" metal -v
  RESULT_VARIABLE MLX_SPATIALKIT_METAL_RESULT
  OUTPUT_VARIABLE MLX_SPATIALKIT_METAL_OUTPUT
  ERROR_VARIABLE MLX_SPATIALKIT_METAL_ERROR
)
if(NOT MLX_SPATIALKIT_METAL_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Metal compiler is unavailable; install the Metal toolchain. "
    "xcrun output: ${MLX_SPATIALKIT_METAL_OUTPUT}${MLX_SPATIALKIT_METAL_ERROR}"
  )
endif()

set(MLX_SPATIALKIT_METAL_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/metal/kernels/texture_bake.metal"
)
set(MLX_SPATIALKIT_METAL_AIR
  "${CMAKE_CURRENT_BINARY_DIR}/texture_bake.air"
)
set(MLX_SPATIALKIT_METAL_LIB
  "${CMAKE_CURRENT_BINARY_DIR}/texture_bake.metallib"
)

add_custom_command(
  OUTPUT "${MLX_SPATIALKIT_METAL_AIR}"
  COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
          -c "${MLX_SPATIALKIT_METAL_SOURCE}"
          -o "${MLX_SPATIALKIT_METAL_AIR}"
  DEPENDS "${MLX_SPATIALKIT_METAL_SOURCE}"
  COMMENT "Compiling mlx-spatialkit Metal probe kernel"
  VERBATIM
)

add_custom_command(
  OUTPUT "${MLX_SPATIALKIT_METAL_LIB}"
  COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib
          "${MLX_SPATIALKIT_METAL_AIR}"
          -o "${MLX_SPATIALKIT_METAL_LIB}"
  DEPENDS "${MLX_SPATIALKIT_METAL_AIR}"
  COMMENT "Linking mlx-spatialkit Metal probe library"
  VERBATIM
)

add_custom_target(mlx_spatialkit_metal_probe
  DEPENDS "${MLX_SPATIALKIT_METAL_LIB}"
)

nanobind_add_module(
  _native
  STABLE_ABI
  cpp/bindings.cpp
  cpp/pixal3d_contracts.cpp
  metal/texture_bake.mm
)

target_include_directories(_native PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
)
target_link_libraries(_native PRIVATE
  "-framework Foundation"
  "-framework Metal"
)
add_dependencies(_native mlx_spatialkit_metal_probe)

install(TARGETS _native LIBRARY DESTINATION mlx_spatialkit)
