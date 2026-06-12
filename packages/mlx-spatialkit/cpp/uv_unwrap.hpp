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

// Stage B: ChartBuilder — xatlas-equivalent chart growth within stage-A
// clusters (behavior port of the vendored reference ClusteredCharts:
// /tmp/CuMesh/third_party/xatlas/xatlas.cpp), plus the orthographic-projection
// parameterization baseline. Returns a dict with at least:
//   chart_ids ([F] int64, dense), chart_count, corner_uvs ([F*3, 2] float64,
//   chart-local projected UVs in corner order), chart_accepted /
//   chart_needs_lscm ([C] int64 masks), chart_stretch_l2 / chart_stretch_linf
//   ([C] float64), accepted_chart_count, lscm_pending_chart_count,
//   planar_region_count, growth merge/fill/relocation counters.
// cluster_ids may be None (whole mesh treated as one cluster); chart growth
// never crosses a cluster boundary.
nanobind::dict grow_uv_charts(
    nanobind::object vertices,
    nanobind::object faces,
    nanobind::object cluster_ids,
    double max_cost,
    double normal_deviation_weight,
    double roundness_weight,
    double straightness_weight,
    double normal_seam_weight,
    double texture_seam_weight,
    int64_t max_iterations,
    double projection_linf_threshold,
    double max_chart_area,
    double max_boundary_length);

}  // namespace mlx_spatialkit
