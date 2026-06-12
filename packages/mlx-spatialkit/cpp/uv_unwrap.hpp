#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>

namespace mlx_spatialkit {

// Native reference-parity UV unwrap pipeline (CuMesh compute_charts /
// uv_unwrap semantics), built in stages:
//   Stage A (this slice): ConeClusterer — heap-driven, cost-ordered chart
//     agglomeration over the face-adjacency graph bounded by a normal-cone
//     half-angle, exposed through compute_uv_charts below.
//   Stage B..D (later slices): ChartBuilder / Parameterizer / Packer are added
//     alongside ConeClusterer in uv_unwrap.cpp, each with its own entry point
//     declared here.

// Cluster mesh faces into UV charts. Returns a dict with at least:
//   chart_ids ([F] int64), chart_count, largest_chart_faces,
//   cone_rejected_merge_count, merge_count, plus the per-chart cones
//   (chart_cone_axes [C,3] float64, chart_cone_half_angles [C] float64) so
//   callers can machine-check the cone invariant.
nanobind::dict compute_uv_charts(
    nanobind::object vertices,
    nanobind::object faces,
    double threshold_cone_half_angle_rad,
    int64_t refine_iterations,
    int64_t global_iterations,
    double smooth_strength,
    double area_penalty_weight,
    double perimeter_area_ratio_weight);

}  // namespace mlx_spatialkit
