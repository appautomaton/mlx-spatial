// Stage A of the native UV unwrap pipeline: ConeClusterer.
//
// Behavioral reference: CuMesh src/atlas.cu (compute_charts and its kernels:
// init_chart_adj, compute_chart_adjacency_cost, collapse_edges,
// compute_chart_normal_cones, refine_charts, hook_edges_if_same_chart) and
// cumesh/cumesh.py compute_charts knob documentation. This is a behavior port,
// not a line port: the reference runs rounds of parallel locally-minimal edge
// collapses on the GPU; here charts agglomerate sequentially in ascending cost
// order from a lazy-invalidation heap (the QemSimplifier discipline from
// simplify.cpp), with adjacency rebuilt in rounds until a round makes no merge.
//
// Intentional deviations from atlas.cu (each preserves a hard guarantee the
// reference only approximates):
//   1. Normal cones are maintained incrementally with the minimal enclosing
//      cone of two cones (the reference's collapse_edges update) and are NEVER
//      re-derived from chart mean normals between rounds. The reference's
//      per-round re-derivation (compute_chart_normal_cones: axis = normalized
//      mean normal, half-angle = max deviation) can transiently exceed the
//      threshold for already-formed charts; the monotone enclosing cone makes
//      "every member face normal lies within threshold of the chart cone axis"
//      a hard invariant of the output.
//   2. refine_charts admits a face into a different chart only when the face
//      normal also lies within the threshold cone of that chart's axis (the
//      reference only requires a positive dot product, which can break the
//      cone invariant for thresholds below 90 degrees). The target cone
//      half-angle widens to cover the admitted face; its axis is unchanged.
//   3. The perimeter/area penalty divides by max(area, 1e-30) and any
//      non-finite cost becomes +infinity (robustness only; the reference
//      divides unguarded).
// Faithfully replicated: the cost composition (atlas.cu
// compute_chart_adjacency_cost: cost = merged_half_angle
// + area_penalty_weight * merged_area
// + perimeter_area_ratio_weight * merged_perimeter^2 / merged_area — note the
// kernel squares the perimeter even though the cumesh.py docstring says
// "Perimeter / Area"; the kernel wins), the enclosing-cone merge formula, the
// merged perimeter (p0 + p1 - 2 * shared boundary length, counting manifold
// inter-chart edges only), the refine candidate cap of 4 and its
// epsilon/smaller-id tie-break, and the per-global-iteration connected
// component reassignment after refinement.

#include "uv_unwrap.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <vector>

#include "mesh_common.hpp"

namespace nb = nanobind;

namespace mlx_spatialkit {
namespace uv_unwrap {
namespace {

using Vec3 = std::array<double, 3>;

constexpr double kRefineScoreEpsilon = 1e-5;     // atlas.cu refine_charts_kernel
constexpr double kRefineScoreSentinel = -1e9;    // atlas.cu refine_charts_kernel
constexpr double kParallelAxisAngle = 1e-3;      // atlas.cu collapse_edges_kernel

double dot(const Vec3 &a, const Vec3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double norm(const Vec3 &a) {
  return std::sqrt(dot(a, a));
}

Vec3 normalized_or_zero(const Vec3 &a) {
  const double length = norm(a);
  if (!std::isfinite(length) || length <= 1e-30) {
    return {0.0, 0.0, 0.0};
  }
  return {a[0] / length, a[1] / length, a[2] / length};
}

double clamped_angle(double cos_angle) {
  return std::acos(std::max(-1.0, std::min(1.0, cos_angle)));
}

Vec3 vertex_position(const mesh_common::MeshData &mesh, int64_t vertex) {
  const auto &v = mesh.vertices[static_cast<size_t>(vertex)];
  return {static_cast<double>(v[0]), static_cast<double>(v[1]), static_cast<double>(v[2])};
}

double edge_length(const mesh_common::MeshData &mesh, int64_t a, int64_t b) {
  const Vec3 pa = vertex_position(mesh, a);
  const Vec3 pb = vertex_position(mesh, b);
  const Vec3 d{pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]};
  return norm(d);
}

// Normal cone: every member face normal lies within `half_angle` of `axis`.
struct NormalCone {
  Vec3 axis{0.0, 0.0, 0.0};
  double half_angle = 0.0;
};

// Minimal enclosing cone of two cones (atlas.cu collapse_edges_kernel /
// compute_chart_adjacency_cost composition): work on the 1D arc through both
// axes; the merged axis is the arc midpoint of the covering interval.
NormalCone merge_cones(const NormalCone &a, const NormalCone &b) {
  const double cos_angle = dot(a.axis, b.axis);
  const double axis_angle = clamped_angle(cos_angle);
  const double low = std::min(-a.half_angle, axis_angle - b.half_angle);
  const double high = std::max(a.half_angle, axis_angle + b.half_angle);
  NormalCone merged;
  merged.half_angle = 0.5 * (high - low);
  if (axis_angle < kParallelAxisAngle) {
    merged.axis = a.axis;
    return merged;
  }
  // Rotate a.axis toward b.axis by the covering-interval midpoint angle.
  const Vec3 perpendicular = normalized_or_zero({
      b.axis[0] - a.axis[0] * cos_angle,
      b.axis[1] - a.axis[1] * cos_angle,
      b.axis[2] - a.axis[2] * cos_angle,
  });
  const double mid = 0.5 * (high + low);
  const Vec3 rotated{
      a.axis[0] * std::cos(mid) + perpendicular[0] * std::sin(mid),
      a.axis[1] * std::cos(mid) + perpendicular[1] * std::sin(mid),
      a.axis[2] * std::cos(mid) + perpendicular[2] * std::sin(mid),
  };
  merged.axis = normalized_or_zero(rotated);
  if (norm(merged.axis) <= 1e-30) {
    merged.axis = a.axis;  // degenerate inputs: keep a deterministic axis
  }
  return merged;
}

// Ordered comparator for EdgeKey so it can key std::map / std::set (EdgeKey
// has no operator<). Lexicographic (a, b) — deterministic, ASLR-independent.
struct EdgeKeyLess {
  bool operator()(const mesh_common::EdgeKey &left, const mesh_common::EdgeKey &right) const {
    if (left.a != right.a) {
      return left.a < right.a;
    }
    return left.b < right.b;
  }
};

struct ChartEdgeCost {
  double cost = 0.0;
  int64_t edge_id = 0;
  int64_t version = 0;
};

// Min-heap ordering: cost ASC, edge_id ASC, version DESC.  std::priority_queue
// is a MAX-heap on operator<, so this comparator returns true when `left`
// should pop AFTER `right` (i.e. when left is the lower-priority element).
struct ChartEdgeCostGreater {
  bool operator()(const ChartEdgeCost &left, const ChartEdgeCost &right) const {
    if (left.cost != right.cost) {
      return left.cost > right.cost;  // smaller cost pops first
    }
    if (left.edge_id != right.edge_id) {
      return left.edge_id > right.edge_id;  // smaller edge_id pops first
    }
    return left.version < right.version;  // newer version pops first
  }
};

// Chart-adjacency edge for one merge round. Endpoints are current chart ids
// with a < b; shared_length aggregates every manifold mesh edge on the
// boundary between the two charts (atlas.cu get_chart_connectivity).
struct ChartEdge {
  int64_t a = 0;
  int64_t b = 0;
  double shared_length = 0.0;
  int64_t version = 0;
  bool alive = false;
};

struct ConeClusterResult {
  std::vector<int64_t> chart_ids;  // dense ids in [0, chart_count)
  std::vector<NormalCone> chart_cones;
  int64_t chart_count = 0;
  int64_t largest_chart_faces = 0;
  int64_t merge_count = 0;
  int64_t cone_rejected_merge_count = 0;
  int64_t cost_rejected_merge_count = 0;
};

class ConeClusterer {
 public:
  ConeClusterer(
      const mesh_common::MeshData &mesh,
      double threshold_cone_half_angle_rad,
      int64_t refine_iterations,
      int64_t global_iterations,
      double smooth_strength,
      double area_penalty_weight,
      double perimeter_area_ratio_weight)
      : mesh_(mesh),
        threshold_(threshold_cone_half_angle_rad),
        refine_iterations_(refine_iterations),
        global_iterations_(global_iterations),
        smooth_strength_(smooth_strength),
        area_penalty_weight_(area_penalty_weight),
        perimeter_area_ratio_weight_(perimeter_area_ratio_weight) {
    build_static_topology();
  }

  ConeClusterResult run() {
    const int64_t face_count = static_cast<int64_t>(mesh_.faces.size());
    chart_of_face_.resize(static_cast<size_t>(face_count));
    std::iota(chart_of_face_.begin(), chart_of_face_.end(), 0);
    chart_count_ = face_count;
    chart_cones_.resize(static_cast<size_t>(face_count));
    chart_areas_.resize(static_cast<size_t>(face_count));
    for (int64_t f = 0; f < face_count; ++f) {
      chart_cones_[static_cast<size_t>(f)] = NormalCone{face_normals_[static_cast<size_t>(f)], 0.0};
      chart_areas_[static_cast<size_t>(f)] = face_areas_[static_cast<size_t>(f)];
    }

    if (face_count > 0) {
      for (int64_t g = 0; g < global_iterations_; ++g) {
        // Merge to fixpoint: a round rebuilds chart adjacency from scratch
        // (restoring pairs dropped by in-round rejections) and drains the
        // heap; a round with zero merges means no admissible merge remains.
        while (run_merge_round() > 0) {
        }
        for (int64_t r = 0; r < refine_iterations_; ++r) {
          refine_pass();
        }
        split_disconnected_charts();
      }
    }

    ConeClusterResult result;
    result.chart_ids = chart_of_face_;
    result.chart_cones = chart_cones_;
    result.chart_count = chart_count_;
    std::vector<int64_t> sizes(static_cast<size_t>(chart_count_), 0);
    for (const int64_t chart : chart_of_face_) {
      sizes[static_cast<size_t>(chart)] += 1;
    }
    for (const int64_t size : sizes) {
      result.largest_chart_faces = std::max(result.largest_chart_faces, size);
    }
    result.merge_count = merge_count_;
    result.cone_rejected_merge_count = cone_rejected_merge_count_;
    result.cost_rejected_merge_count = cost_rejected_merge_count_;
    return result;
  }

 private:
  struct AdjacencyPair {
    int64_t f0 = 0;  // f0 < f1
    int64_t f1 = 0;
    double length = 0.0;
  };

  const mesh_common::MeshData &mesh_;
  const double threshold_;
  const int64_t refine_iterations_;
  const int64_t global_iterations_;
  const double smooth_strength_;
  const double area_penalty_weight_;
  const double perimeter_area_ratio_weight_;

  // Immutable mesh topology.
  std::vector<Vec3> face_normals_;
  std::vector<double> face_areas_;
  // Manifold face adjacency (mesh edges shared by exactly two faces), in
  // ascending edge-key order — the merge graph and component-split graph.
  std::vector<AdjacencyPair> face_adjacency_;
  // All faces incident to each mesh edge (ascending face ids) — the refine
  // candidate source (atlas.cu edge2face).
  std::map<mesh_common::EdgeKey, std::vector<int64_t>, EdgeKeyLess> edge_faces_;

  // Chart state with dense ids in [0, chart_count_).
  std::vector<int64_t> chart_of_face_;
  std::vector<NormalCone> chart_cones_;
  std::vector<double> chart_areas_;
  int64_t chart_count_ = 0;

  // Round-local merge state (rebuilt by run_merge_round).
  std::vector<ChartEdge> edges_;
  std::map<mesh_common::EdgeKey, int64_t, EdgeKeyLess> edge_index_;
  std::vector<std::set<int64_t>> chart_edge_sets_;
  std::vector<double> chart_perimeters_;
  std::vector<int64_t> merge_parent_;  // round DSU; the kept chart is min(a, b)
  std::priority_queue<ChartEdgeCost, std::vector<ChartEdgeCost>, ChartEdgeCostGreater> heap_;
  int64_t live_edge_count_ = 0;

  int64_t merge_count_ = 0;
  int64_t cone_rejected_merge_count_ = 0;
  int64_t cost_rejected_merge_count_ = 0;

  void build_static_topology() {
    face_normals_.reserve(mesh_.faces.size());
    face_areas_.reserve(mesh_.faces.size());
    for (const auto &face : mesh_.faces) {
      const Vec3 a = vertex_position(mesh_, face[0]);
      const Vec3 b = vertex_position(mesh_, face[1]);
      const Vec3 c = vertex_position(mesh_, face[2]);
      const Vec3 ab{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
      const Vec3 ac{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
      const Vec3 cross{
          ab[1] * ac[2] - ab[2] * ac[1],
          ab[2] * ac[0] - ab[0] * ac[2],
          ab[0] * ac[1] - ab[1] * ac[0],
      };
      face_normals_.push_back(normalized_or_zero(cross));
      const double cross_norm = norm(cross);
      face_areas_.push_back(std::isfinite(cross_norm) ? 0.5 * cross_norm : 0.0);
    }
    for (int64_t fi = 0; fi < static_cast<int64_t>(mesh_.faces.size()); ++fi) {
      const auto &face = mesh_.faces[static_cast<size_t>(fi)];
      edge_faces_[mesh_common::edge_key(face[0], face[1])].push_back(fi);
      edge_faces_[mesh_common::edge_key(face[1], face[2])].push_back(fi);
      edge_faces_[mesh_common::edge_key(face[2], face[0])].push_back(fi);
    }
    // Manifold adjacency only (exactly two incident faces), mirroring the
    // reference manifold_face_adj; non-manifold edges never drive merges.
    for (const auto &[key, incident] : edge_faces_) {
      if (incident.size() != 2) {
        continue;
      }
      face_adjacency_.push_back(AdjacencyPair{
          incident[0], incident[1], edge_length(mesh_, key.a, key.b)});
    }
  }

  // ---- merge phase -------------------------------------------------------

  double edge_cost(const ChartEdge &edge, const NormalCone &merged) const {
    // atlas.cu compute_chart_adjacency_cost_kernel composition.
    const double merged_area =
        chart_areas_[static_cast<size_t>(edge.a)] + chart_areas_[static_cast<size_t>(edge.b)];
    const double merged_perimeter = chart_perimeters_[static_cast<size_t>(edge.a)] +
        chart_perimeters_[static_cast<size_t>(edge.b)] - 2.0 * edge.shared_length;
    double cost = merged.half_angle;
    cost += area_penalty_weight_ * merged_area;
    cost += perimeter_area_ratio_weight_ *
        (merged_perimeter * merged_perimeter / std::max(merged_area, 1e-30));
    if (!std::isfinite(cost)) {
      cost = std::numeric_limits<double>::infinity();
    }
    return cost;
  }

  void push_edge_cost(int64_t edge_id) {
    const ChartEdge &edge = edges_[static_cast<size_t>(edge_id)];
    if (!edge.alive) {
      return;
    }
    const NormalCone merged = merge_cones(
        chart_cones_[static_cast<size_t>(edge.a)], chart_cones_[static_cast<size_t>(edge.b)]);
    heap_.push(ChartEdgeCost{edge_cost(edge, merged), edge_id, edge.version});
  }

  bool pop_valid_edge(ChartEdgeCost &out) {
    while (!heap_.empty()) {
      const ChartEdgeCost candidate = heap_.top();
      heap_.pop();
      const ChartEdge &edge = edges_[static_cast<size_t>(candidate.edge_id)];
      if (!edge.alive || edge.version != candidate.version) {
        continue;  // stale lazy entry
      }
      out = candidate;
      return true;
    }
    return false;
  }

  void kill_edge(int64_t edge_id) {
    ChartEdge &edge = edges_[static_cast<size_t>(edge_id)];
    if (!edge.alive) {
      return;
    }
    edge.alive = false;
    edge.version += 1;
    edge_index_.erase(mesh_common::edge_key(edge.a, edge.b));
    chart_edge_sets_[static_cast<size_t>(edge.a)].erase(edge_id);
    chart_edge_sets_[static_cast<size_t>(edge.b)].erase(edge_id);
    live_edge_count_ -= 1;
  }

  void maybe_compact_heap() {
    // Lazy invalidation grows the heap with stale entries; rebuild when stale
    // entries dominate (same 3x live threshold as QemSimplifier). Compaction
    // only bounds memory — pop order is unchanged (determinism preserved).
    if (static_cast<int64_t>(heap_.size()) <= 3 * std::max<int64_t>(1, live_edge_count_)) {
      return;
    }
    decltype(heap_) rebuilt;
    heap_ = std::move(rebuilt);
    for (int64_t edge_id = 0; edge_id < static_cast<int64_t>(edges_.size()); ++edge_id) {
      push_edge_cost(edge_id);  // no-op for dead edges
    }
  }

  int64_t find_merge_root(int64_t chart) {
    int64_t root = chart;
    while (merge_parent_[static_cast<size_t>(root)] != root) {
      root = merge_parent_[static_cast<size_t>(root)];
    }
    while (merge_parent_[static_cast<size_t>(chart)] != chart) {
      const int64_t next = merge_parent_[static_cast<size_t>(chart)];
      merge_parent_[static_cast<size_t>(chart)] = root;
      chart = next;
    }
    return root;
  }

  // One adjacency-rebuild + heap-drain round. Returns the number of merges.
  int64_t run_merge_round() {
    edges_.clear();
    edge_index_.clear();
    chart_edge_sets_.assign(static_cast<size_t>(chart_count_), {});
    chart_perimeters_.assign(static_cast<size_t>(chart_count_), 0.0);
    merge_parent_.resize(static_cast<size_t>(chart_count_));
    std::iota(merge_parent_.begin(), merge_parent_.end(), 0);
    heap_ = {};
    live_edge_count_ = 0;

    // Aggregate shared boundary length per adjacent chart pair (ordered map:
    // edge ids are assigned in ascending (a, b) order — deterministic).
    std::map<mesh_common::EdgeKey, double, EdgeKeyLess> pair_lengths;
    for (const AdjacencyPair &pair : face_adjacency_) {
      const int64_t c0 = chart_of_face_[static_cast<size_t>(pair.f0)];
      const int64_t c1 = chart_of_face_[static_cast<size_t>(pair.f1)];
      if (c0 == c1) {
        continue;
      }
      pair_lengths[mesh_common::edge_key(c0, c1)] += pair.length;
    }
    if (pair_lengths.empty()) {
      return 0;
    }
    for (const auto &[key, length] : pair_lengths) {
      const int64_t edge_id = static_cast<int64_t>(edges_.size());
      edges_.push_back(ChartEdge{key.a, key.b, length, 0, true});
      edge_index_.emplace(key, edge_id);
      chart_edge_sets_[static_cast<size_t>(key.a)].insert(edge_id);
      chart_edge_sets_[static_cast<size_t>(key.b)].insert(edge_id);
      chart_perimeters_[static_cast<size_t>(key.a)] += length;
      chart_perimeters_[static_cast<size_t>(key.b)] += length;
      live_edge_count_ += 1;
    }
    for (int64_t edge_id = 0; edge_id < static_cast<int64_t>(edges_.size()); ++edge_id) {
      push_edge_cost(edge_id);
    }

    int64_t merges = 0;
    ChartEdgeCost top;
    while (pop_valid_edge(top)) {
      const ChartEdge edge = edges_[static_cast<size_t>(top.edge_id)];  // copy
      const NormalCone merged = merge_cones(
          chart_cones_[static_cast<size_t>(edge.a)], chart_cones_[static_cast<size_t>(edge.b)]);
      // Admissibility: the merged cone must respect the half-angle bound, and
      // (reference collapse_edges_kernel) the penalty-inflated cost must not
      // exceed the threshold either. Rejected pairs are dropped for this
      // round; the next round's rebuild re-offers cost-rejected pairs (cone
      // growth is monotone, so cone-rejected pairs can never become valid).
      if (merged.half_angle > threshold_) {
        cone_rejected_merge_count_ += 1;
        kill_edge(top.edge_id);
        continue;
      }
      if (top.cost > threshold_) {
        cost_rejected_merge_count_ += 1;
        kill_edge(top.edge_id);
        continue;
      }
      merge_charts(top.edge_id, edge, merged);
      merges += 1;
      maybe_compact_heap();
    }
    if (merges == 0) {
      return 0;
    }

    // Relabel surviving charts densely, preserving ascending id order.
    std::vector<int64_t> dense(static_cast<size_t>(chart_count_), -1);
    int64_t next_id = 0;
    for (int64_t chart = 0; chart < chart_count_; ++chart) {
      if (find_merge_root(chart) == chart) {
        dense[static_cast<size_t>(chart)] = next_id;
        next_id += 1;
      }
    }
    std::vector<NormalCone> next_cones(static_cast<size_t>(next_id));
    std::vector<double> next_areas(static_cast<size_t>(next_id), 0.0);
    for (int64_t chart = 0; chart < chart_count_; ++chart) {
      if (dense[static_cast<size_t>(chart)] >= 0) {
        next_cones[static_cast<size_t>(dense[static_cast<size_t>(chart)])] =
            chart_cones_[static_cast<size_t>(chart)];
        next_areas[static_cast<size_t>(dense[static_cast<size_t>(chart)])] =
            chart_areas_[static_cast<size_t>(chart)];
      }
    }
    for (int64_t &chart : chart_of_face_) {
      chart = dense[static_cast<size_t>(find_merge_root(chart))];
    }
    chart_cones_ = std::move(next_cones);
    chart_areas_ = std::move(next_areas);
    chart_count_ = next_id;
    return merges;
  }

  void merge_charts(int64_t merge_edge_id, const ChartEdge &edge, const NormalCone &merged) {
    const int64_t keep = edge.a;  // a < b by EdgeKey construction
    const int64_t gone = edge.b;
    chart_cones_[static_cast<size_t>(keep)] = merged;
    chart_areas_[static_cast<size_t>(keep)] += chart_areas_[static_cast<size_t>(gone)];
    chart_perimeters_[static_cast<size_t>(keep)] +=
        chart_perimeters_[static_cast<size_t>(gone)] - 2.0 * edge.shared_length;
    merge_parent_[static_cast<size_t>(gone)] = keep;
    kill_edge(merge_edge_id);

    // Re-target `gone`'s surviving adjacency onto `keep`, summing shared
    // lengths where both charts already bordered the same neighbor.
    const std::vector<int64_t> gone_edges(
        chart_edge_sets_[static_cast<size_t>(gone)].begin(),
        chart_edge_sets_[static_cast<size_t>(gone)].end());
    for (const int64_t moved_id : gone_edges) {
      const ChartEdge moved = edges_[static_cast<size_t>(moved_id)];  // copy before kill
      const int64_t other = moved.a == gone ? moved.b : moved.a;
      kill_edge(moved_id);
      const mesh_common::EdgeKey key = mesh_common::edge_key(keep, other);
      const auto found = edge_index_.find(key);
      if (found != edge_index_.end()) {
        edges_[static_cast<size_t>(found->second)].shared_length += moved.shared_length;
      } else {
        const int64_t new_id = static_cast<int64_t>(edges_.size());
        edges_.push_back(ChartEdge{key.a, key.b, moved.shared_length, 0, true});
        edge_index_.emplace(key, new_id);
        chart_edge_sets_[static_cast<size_t>(key.a)].insert(new_id);
        chart_edge_sets_[static_cast<size_t>(key.b)].insert(new_id);
        live_edge_count_ += 1;
      }
    }
    // Every edge incident to `keep` now has a different merged cost (cone,
    // area, and perimeter all changed): bump versions (invalidating heap
    // entries lazily) and push fresh costs. Set iteration is ascending by
    // edge id — deterministic.
    for (const int64_t incident_id : chart_edge_sets_[static_cast<size_t>(keep)]) {
      edges_[static_cast<size_t>(incident_id)].version += 1;
      push_edge_cost(incident_id);
    }
    merge_count_ += 1;
  }

  // ---- refine phase ------------------------------------------------------

  void refine_pass() {
    const int64_t face_count = static_cast<int64_t>(mesh_.faces.size());
    std::vector<int64_t> next_ids(static_cast<size_t>(face_count));
    for (int64_t f = 0; f < face_count; ++f) {
      next_ids[static_cast<size_t>(f)] = refine_face(f);
    }
    // Ping-pong apply (reference double-buffers), widening target cones so the
    // cone invariant survives reassignment (deviation 2 in the header note).
    for (int64_t f = 0; f < face_count; ++f) {
      const int64_t next = next_ids[static_cast<size_t>(f)];
      if (next == chart_of_face_[static_cast<size_t>(f)]) {
        continue;
      }
      NormalCone &cone = chart_cones_[static_cast<size_t>(next)];
      cone.half_angle = std::max(
          cone.half_angle,
          clamped_angle(dot(face_normals_[static_cast<size_t>(f)], cone.axis)));
    }
    chart_of_face_ = std::move(next_ids);
    compress_chart_labels();
  }

  int64_t refine_face(int64_t f) const {
    const int64_t current = chart_of_face_[static_cast<size_t>(f)];
    const Vec3 &normal = face_normals_[static_cast<size_t>(f)];
    // Register cache semantics from refine_charts_kernel: self plus at most
    // three neighbor charts, first-come in edge order.
    std::array<int64_t, 4> candidates{current, 0, 0, 0};
    std::array<double, 4> smooth_scores{0.0, 0.0, 0.0, 0.0};
    int candidate_count = 1;

    const auto &face = mesh_.faces[static_cast<size_t>(f)];
    for (int corner = 0; corner < 3; ++corner) {
      const int64_t v0 = face[static_cast<size_t>(corner)];
      const int64_t v1 = face[static_cast<size_t>((corner + 1) % 3)];
      const double length = edge_length(mesh_, v0, v1);
      const auto &incident = edge_faces_.at(mesh_common::edge_key(v0, v1));
      for (const int64_t neighbor : incident) {
        if (neighbor == f) {
          continue;
        }
        const int64_t chart = chart_of_face_[static_cast<size_t>(neighbor)];
        int index = -1;
        for (int k = 0; k < candidate_count; ++k) {
          if (candidates[static_cast<size_t>(k)] == chart) {
            index = k;
            break;
          }
        }
        if (index < 0 && candidate_count < 4) {
          index = candidate_count;
          candidate_count += 1;
          candidates[static_cast<size_t>(index)] = chart;
        }
        if (index >= 0) {
          smooth_scores[static_cast<size_t>(index)] += length;
        }
      }
    }

    int64_t best_chart = current;
    double best_score = kRefineScoreSentinel;
    for (int i = 0; i < candidate_count; ++i) {
      const int64_t chart = candidates[static_cast<size_t>(i)];
      const NormalCone &cone = chart_cones_[static_cast<size_t>(chart)];
      const double geometric = dot(normal, cone.axis);
      if (geometric <= 0.0) {
        continue;
      }
      // Cone admission (deviation 2): only adopt a chart whose cone axis
      // already lies within the threshold of this face normal.
      if (chart != current && clamped_angle(geometric) > threshold_) {
        continue;
      }
      const double total = geometric + smooth_strength_ * smooth_scores[static_cast<size_t>(i)];
      if (chart == current && best_score == kRefineScoreSentinel) {
        best_score = total;
        best_chart = chart;
      }
      const double diff = total - best_score;
      if (diff > kRefineScoreEpsilon) {
        best_score = total;
        best_chart = chart;
      } else if (std::fabs(diff) <= kRefineScoreEpsilon && chart < best_chart) {
        best_score = total;
        best_chart = chart;
      }
    }
    return best_chart;
  }

  // Drop empty chart labels (after refinement) preserving ascending id order;
  // cones carry over, areas are re-accumulated from member faces.
  void compress_chart_labels() {
    std::vector<int64_t> sizes(static_cast<size_t>(chart_count_), 0);
    for (const int64_t chart : chart_of_face_) {
      sizes[static_cast<size_t>(chart)] += 1;
    }
    std::vector<int64_t> dense(static_cast<size_t>(chart_count_), -1);
    int64_t next_id = 0;
    for (int64_t chart = 0; chart < chart_count_; ++chart) {
      if (sizes[static_cast<size_t>(chart)] > 0) {
        dense[static_cast<size_t>(chart)] = next_id;
        next_id += 1;
      }
    }
    if (next_id == chart_count_) {
      recompute_chart_areas();
      return;
    }
    std::vector<NormalCone> next_cones(static_cast<size_t>(next_id));
    for (int64_t chart = 0; chart < chart_count_; ++chart) {
      if (dense[static_cast<size_t>(chart)] >= 0) {
        next_cones[static_cast<size_t>(dense[static_cast<size_t>(chart)])] =
            chart_cones_[static_cast<size_t>(chart)];
      }
    }
    for (int64_t &chart : chart_of_face_) {
      chart = dense[static_cast<size_t>(chart)];
    }
    chart_cones_ = std::move(next_cones);
    chart_count_ = next_id;
    recompute_chart_areas();
  }

  void recompute_chart_areas() {
    chart_areas_.assign(static_cast<size_t>(chart_count_), 0.0);
    for (int64_t f = 0; f < static_cast<int64_t>(mesh_.faces.size()); ++f) {
      chart_areas_[static_cast<size_t>(chart_of_face_[static_cast<size_t>(f)])] +=
          face_areas_[static_cast<size_t>(f)];
    }
  }

  // Refinement can disconnect a chart; split every chart into face-connected
  // components (reference reassign_chart_ids / hook_edges_if_same_chart).
  // Components inherit the parent chart's cone — still a superset of their
  // member normals, so the cone invariant is preserved.
  void split_disconnected_charts() {
    const int64_t face_count = static_cast<int64_t>(mesh_.faces.size());
    mesh_common::UnionFind components(static_cast<size_t>(face_count));
    for (const AdjacencyPair &pair : face_adjacency_) {
      if (chart_of_face_[static_cast<size_t>(pair.f0)] ==
          chart_of_face_[static_cast<size_t>(pair.f1)]) {
        components.unite(static_cast<size_t>(pair.f0), static_cast<size_t>(pair.f1));
      }
    }
    // New ids by first occurrence in face order — deterministic.
    std::map<int64_t, int64_t> root_to_id;
    std::vector<int64_t> next_ids(static_cast<size_t>(face_count));
    std::vector<NormalCone> next_cones;
    for (int64_t f = 0; f < face_count; ++f) {
      const int64_t root = static_cast<int64_t>(components.find(static_cast<size_t>(f)));
      const auto found = root_to_id.find(root);
      if (found != root_to_id.end()) {
        next_ids[static_cast<size_t>(f)] = found->second;
        continue;
      }
      const int64_t id = static_cast<int64_t>(next_cones.size());
      root_to_id.emplace(root, id);
      next_cones.push_back(chart_cones_[static_cast<size_t>(chart_of_face_[static_cast<size_t>(f)])]);
      next_ids[static_cast<size_t>(f)] = id;
    }
    chart_of_face_ = std::move(next_ids);
    chart_cones_ = std::move(next_cones);
    chart_count_ = static_cast<int64_t>(root_to_id.size());
    recompute_chart_areas();
  }
};

nb::object make_int64_vector(std::vector<int64_t> values) {
  auto owner = new std::vector<int64_t>(std::move(values));
  nb::capsule capsule(owner, [](void *ptr) noexcept {
    delete static_cast<std::vector<int64_t> *>(ptr);
  });
  return nb::ndarray<nb::numpy, int64_t>(owner->data(), {owner->size()}, capsule).cast();
}

nb::object make_float64_vector(std::vector<double> values) {
  auto owner = new std::vector<double>(std::move(values));
  nb::capsule capsule(owner, [](void *ptr) noexcept {
    delete static_cast<std::vector<double> *>(ptr);
  });
  return nb::ndarray<nb::numpy, double>(owner->data(), {owner->size()}, capsule).cast();
}

nb::object make_float64_matrix(std::vector<double> values, size_t rows, size_t cols) {
  auto owner = new std::vector<double>(std::move(values));
  nb::capsule capsule(owner, [](void *ptr) noexcept {
    delete static_cast<std::vector<double> *>(ptr);
  });
  return nb::ndarray<nb::numpy, double>(owner->data(), {rows, cols}, capsule).cast();
}

}  // namespace
}  // namespace uv_unwrap

nb::dict compute_uv_charts(
    nb::object vertices,
    nb::object faces,
    double threshold_cone_half_angle_rad,
    int64_t refine_iterations,
    int64_t global_iterations,
    double smooth_strength,
    double area_penalty_weight,
    double perimeter_area_ratio_weight) {
  const mesh_common::MeshData mesh = mesh_common::load_mesh(vertices, faces);
  if (!std::isfinite(threshold_cone_half_angle_rad) || threshold_cone_half_angle_rad <= 0.0) {
    throw nb::value_error("threshold_cone_half_angle_rad must be finite and positive");
  }
  if (refine_iterations < 0) {
    throw nb::value_error("refine_iterations must be non-negative");
  }
  if (global_iterations < 0) {
    throw nb::value_error("global_iterations must be non-negative");
  }
  if (!std::isfinite(smooth_strength) || !std::isfinite(area_penalty_weight) ||
      !std::isfinite(perimeter_area_ratio_weight)) {
    throw nb::value_error(
        "smooth_strength, area_penalty_weight, and perimeter_area_ratio_weight must be finite");
  }

  uv_unwrap::ConeClusterResult clusters =
      uv_unwrap::ConeClusterer(
          mesh,
          threshold_cone_half_angle_rad,
          refine_iterations,
          global_iterations,
          smooth_strength,
          area_penalty_weight,
          perimeter_area_ratio_weight)
          .run();

  std::vector<double> axes;
  axes.reserve(clusters.chart_cones.size() * 3);
  std::vector<double> half_angles;
  half_angles.reserve(clusters.chart_cones.size());
  for (const uv_unwrap::NormalCone &cone : clusters.chart_cones) {
    axes.push_back(cone.axis[0]);
    axes.push_back(cone.axis[1]);
    axes.push_back(cone.axis[2]);
    half_angles.push_back(cone.half_angle);
  }

  nb::dict result;
  result["chart_ids"] = uv_unwrap::make_int64_vector(std::move(clusters.chart_ids));
  result["chart_count"] = clusters.chart_count;
  result["largest_chart_faces"] = clusters.largest_chart_faces;
  result["merge_count"] = clusters.merge_count;
  result["cone_rejected_merge_count"] = clusters.cone_rejected_merge_count;
  result["cost_rejected_merge_count"] = clusters.cost_rejected_merge_count;
  result["chart_cone_axes"] = uv_unwrap::make_float64_matrix(
      std::move(axes), clusters.chart_cones.size(), 3);
  result["chart_cone_half_angles"] = uv_unwrap::make_float64_vector(std::move(half_angles));
  return result;
}

}  // namespace mlx_spatialkit
