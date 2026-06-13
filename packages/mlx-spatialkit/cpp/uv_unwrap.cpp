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
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <utility>
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
          // Fixpoint early-exit: a pass that reassigns nothing is a fixpoint
          // (no cone widens, compress/recompute are idempotent), so the
          // remaining passes would be identical no-ops.
          if (!refine_pass()) {
            break;
          }
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
  // Flattened per-face refine neighbors (CSR), precomputed once: entries in
  // corner order then ascending incident-face order — the exact iteration
  // order refine_face previously derived from edge_faces_ lookups per pass.
  struct RefineNeighbor {
    int64_t face = 0;
    double edge_length = 0.0;
  };
  std::vector<RefineNeighbor> face_neighbors_;
  std::vector<int64_t> face_neighbor_offsets_;

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
    // Precompute the refine neighbor table (was three edge_faces_ map lookups
    // plus three sqrt edge lengths per face per refine pass).
    const int64_t face_count = static_cast<int64_t>(mesh_.faces.size());
    face_neighbor_offsets_.assign(static_cast<size_t>(face_count) + 1, 0);
    for (int64_t fi = 0; fi < face_count; ++fi) {
      const auto &face = mesh_.faces[static_cast<size_t>(fi)];
      for (int corner = 0; corner < 3; ++corner) {
        const int64_t v0 = face[static_cast<size_t>(corner)];
        const int64_t v1 = face[static_cast<size_t>((corner + 1) % 3)];
        const double length = edge_length(mesh_, v0, v1);
        const auto &incident = edge_faces_.at(mesh_common::edge_key(v0, v1));
        for (const int64_t neighbor : incident) {
          if (neighbor == fi) {
            continue;
          }
          face_neighbors_.push_back(RefineNeighbor{neighbor, length});
        }
      }
      face_neighbor_offsets_[static_cast<size_t>(fi) + 1] =
          static_cast<int64_t>(face_neighbors_.size());
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

  // Returns false when the pass reassigned nothing (fixpoint).
  bool refine_pass() {
    const int64_t face_count = static_cast<int64_t>(mesh_.faces.size());
    std::vector<int64_t> next_ids(static_cast<size_t>(face_count));
    for (int64_t f = 0; f < face_count; ++f) {
      next_ids[static_cast<size_t>(f)] = refine_face(f);
    }
    if (next_ids == chart_of_face_) {
      return false;
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
    return true;
  }

  int64_t refine_face(int64_t f) const {
    const int64_t current = chart_of_face_[static_cast<size_t>(f)];
    const Vec3 &normal = face_normals_[static_cast<size_t>(f)];
    // Register cache semantics from refine_charts_kernel: self plus at most
    // three neighbor charts, first-come in edge order.
    std::array<int64_t, 4> candidates{current, 0, 0, 0};
    std::array<double, 4> smooth_scores{0.0, 0.0, 0.0, 0.0};
    int candidate_count = 1;

    const int64_t begin = face_neighbor_offsets_[static_cast<size_t>(f)];
    const int64_t end = face_neighbor_offsets_[static_cast<size_t>(f) + 1];
    for (int64_t n = begin; n < end; ++n) {
      const RefineNeighbor &entry = face_neighbors_[static_cast<size_t>(n)];
      const int64_t neighbor = entry.face;
      const double length = entry.edge_length;
      {
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

// ---------------------------------------------------------------------------
// Stage B: ChartBuilder — behavior port of the vendored reference xatlas
// ClusteredCharts (/tmp/CuMesh/third_party/xatlas/xatlas.cpp:5320-6300), the
// engine CuMesh.uv_unwrap runs per stage-A cluster. Running it once over the
// whole mesh with cluster-filtered adjacency is equivalent to per-cluster
// runs: growth, holes, and merges never cross a cluster boundary, and phases
// only interleave clusters without changing within-cluster decisions.
//
// Documented deviations from the reference (behavior-only port):
//  1. Welded indexed input means xatlas isSeam() is always false, so the
//     normal-seam and texture-seam metrics contribute exactly 0; the weights
//     are accepted and echoed for contract completeness.
//  2. Basis fitting implements the reference least-squares path only; the
//     eigen fallback (degenerate covariance) fails the face add instead.
//  3. Boundary self-intersection uses exact pairwise segment tests with a
//     bbox cull instead of the reference's uniform grid — same predicate,
//     different acceleration; chart boundaries are small.
//  4. Equal-cost candidate ordering matches the reference's insertion-order
//     stable queue; the global growth scan breaks ties toward the
//     lowest-indexed chart (the reference's strict '<' scan does the same).
//  5. CostQueue per-merge stats and the merged chart's boundary bookkeeping
//     follow the reference exactly, including its one-shared-length merge
//     update (mergeChart: bl_owner += bl_2 - shared, not the geometric 2x).
// ---------------------------------------------------------------------------

constexpr double kPlanarNormalEpsilon = 0.001;       // xatlas kNormalEpsilon
constexpr double kMergeMinNormalDot = 0.5;           // XA_MERGE_CHARTS_MIN_NORMAL_DEVIATION
constexpr double kXatlasEpsilon = 0.0001;            // xatlas kEpsilon
constexpr double kNormalDeviationCutoff = 0.707;     // computeCost hard reject (~75 deg)
constexpr double kBoundaryIntersectEpsilon = 1.192092896e-07;  // xatlas mesh epsilon
constexpr double kUvAreaEpsilon = 1e-12;

struct Vec2d {
  double u = 0.0;
  double v = 0.0;
};

struct Basis {
  Vec3 normal{0.0, 0.0, 0.0};
  Vec3 tangent{0.0, 0.0, 0.0};
  Vec3 bitangent{0.0, 0.0, 0.0};
};

Vec3 vec_sub(const Vec3 &a, const Vec3 &b) {
  return Vec3{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 vec_cross(const Vec3 &a, const Vec3 &b) {
  return Vec3{
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
  };
}

// xatlas Basis::computeTangent: minimum axis, orthogonalized.
Vec3 compute_tangent(const Vec3 &normal) {
  Vec3 tangent{0.0, 0.0, 0.0};
  if (std::fabs(normal[0]) < std::fabs(normal[1]) &&
      std::fabs(normal[0]) < std::fabs(normal[2])) {
    tangent = Vec3{1.0, 0.0, 0.0};
  } else if (std::fabs(normal[1]) < std::fabs(normal[2])) {
    tangent = Vec3{0.0, 1.0, 0.0};
  } else {
    tangent = Vec3{0.0, 0.0, 1.0};
  }
  const double d = dot(normal, tangent);
  tangent = Vec3{tangent[0] - normal[0] * d, tangent[1] - normal[1] * d,
                 tangent[2] - normal[2] * d};
  return normalized_or_zero(tangent);
}

// xatlas Fit::computeLeastSquaresNormal + tangent/bitangent (deviation 2: no
// eigen fallback).
bool least_squares_basis(const std::vector<Vec3> &points, Basis *basis) {
  if (points.size() < 3) {
    return false;
  }
  Vec3 normal{0.0, 0.0, 0.0};
  if (points.size() == 3) {
    normal = normalized_or_zero(
        vec_cross(vec_sub(points[2], points[0]), vec_sub(points[1], points[0])));
  } else {
    const double inv_n = 1.0 / static_cast<double>(points.size());
    Vec3 centroid{0.0, 0.0, 0.0};
    for (const Vec3 &p : points) {
      centroid[0] += p[0];
      centroid[1] += p[1];
      centroid[2] += p[2];
    }
    centroid = Vec3{centroid[0] * inv_n, centroid[1] * inv_n, centroid[2] * inv_n};
    double xx = 0.0, xy = 0.0, xz = 0.0, yy = 0.0, yz = 0.0, zz = 0.0;
    for (const Vec3 &p : points) {
      const Vec3 r = vec_sub(p, centroid);
      xx += r[0] * r[0];
      xy += r[0] * r[1];
      xz += r[0] * r[2];
      yy += r[1] * r[1];
      yz += r[1] * r[2];
      zz += r[2] * r[2];
    }
    const double det_x = yy * zz - yz * yz;
    const double det_y = xx * zz - xz * xz;
    const double det_z = xx * yy - xy * xy;
    const double det_max = std::max(det_x, std::max(det_y, det_z));
    if (det_max <= 0.0) {
      return false;
    }
    Vec3 dir{0.0, 0.0, 0.0};
    if (det_max == det_x) {
      dir = Vec3{det_x, xz * yz - xy * zz, xy * yz - xz * yy};
    } else if (det_max == det_y) {
      dir = Vec3{xz * yz - xy * zz, det_y, xy * xz - yz * xx};
    } else {
      dir = Vec3{xy * yz - xz * yy, xy * xz - yz * xx, det_z};
    }
    normal = normalized_or_zero(dir);
  }
  if (norm(normal) < 0.5) {
    return false;
  }
  basis->normal = normal;
  basis->tangent = compute_tangent(normal);
  if (norm(basis->tangent) < 0.5) {
    return false;
  }
  basis->bitangent = vec_cross(normal, basis->tangent);
  return true;
}

struct GrowthOptions {
  double max_cost = 2.0;
  double normal_deviation_weight = 2.0;
  double roundness_weight = 0.01;
  double straightness_weight = 6.0;
  double normal_seam_weight = 4.0;   // deviation 1: zero contribution
  double texture_seam_weight = 0.5;  // deviation 1: zero contribution
  int64_t max_iterations = 1;
  double projection_linf_threshold = 1.25;
  double max_chart_area = 0.0;
  double max_boundary_length = 0.0;
};

// xatlas CostQueue: array sorted descending by cost (back = best/lowest);
// insertion-order stable for equal costs; bounded queues drop the worst.
class GrowthCostQueue {
 public:
  explicit GrowthCostQueue(size_t max_size = std::numeric_limits<size_t>::max())
      : max_size_(max_size) {}

  bool empty() const { return pairs_.empty(); }
  size_t count() const { return pairs_.size(); }
  double peek_cost() const { return pairs_.back().cost; }
  int64_t peek_face() const { return pairs_.back().face; }
  void clear() { pairs_.clear(); }

  void push(double cost, int64_t face) {
    const Pair pair{cost, face};
    if (pairs_.empty() || cost < peek_cost()) {
      pairs_.push_back(pair);
      return;
    }
    size_t i = 0;
    for (; i < pairs_.size(); ++i) {
      if (pairs_[i].cost < cost) {
        break;
      }
    }
    pairs_.insert(pairs_.begin() + static_cast<std::ptrdiff_t>(i), pair);
    if (pairs_.size() > max_size_) {
      pairs_.erase(pairs_.begin());
    }
  }

  int64_t pop() {
    const int64_t face = pairs_.back().face;
    pairs_.pop_back();
    return face;
  }

 private:
  struct Pair {
    double cost = 0.0;
    int64_t face = 0;
  };
  std::vector<Pair> pairs_;
  size_t max_size_;
};

struct ChartBuildResult {
  std::vector<int64_t> chart_ids;
  std::vector<double> corner_uvs;  // [F*3, 2] flattened
  int64_t chart_count = 0;
  std::vector<int64_t> chart_face_counts;
  std::vector<double> chart_stretch_l2;
  std::vector<double> chart_stretch_linf;
  std::vector<int64_t> chart_accepted;
  std::vector<int64_t> chart_needs_lscm;
  int64_t accepted_chart_count = 0;
  int64_t lscm_pending_chart_count = 0;
  int64_t planar_region_count = 0;
  int64_t place_seed_chart_count = 0;
  int64_t fill_hole_chart_count = 0;
  int64_t growth_merge_count = 0;
  int64_t seed_relocation_count = 0;
  int64_t failed_add_count = 0;
  int64_t mirrored_chart_normalized_count = 0;
};

class ChartBuilder {
 public:
  ChartBuilder(
      const mesh_common::MeshData &mesh,
      const std::vector<int64_t> &cluster_ids,
      const GrowthOptions &options)
      : mesh_(mesh), cluster_ids_(cluster_ids), options_(options) {
    build_topology();
    build_planar_regions();
  }

  ChartBuildResult run() {
    const int64_t face_count = static_cast<int64_t>(mesh_.faces.size());
    face_chart_.assign(static_cast<size_t>(face_count), -1);
    corner_uvs_.assign(static_cast<size_t>(face_count) * 3, Vec2d{});
    faces_left_ = face_count;

    // Reference ClusteredCharts::compute().
    place_seeds(options_.max_cost * 0.5);
    place_seed_chart_count_ = static_cast<int64_t>(charts_.size());
    if (options_.max_iterations > 0) {
      relocate_seeds();
      reset_charts();
      int64_t iteration = 0;
      for (;;) {
        grow_charts(options_.max_cost);
        const int64_t before_fill = static_cast<int64_t>(charts_.size());
        fill_holes(options_.max_cost * 0.5);
        fill_hole_chart_count_ += static_cast<int64_t>(charts_.size()) - before_fill;
        merge_charts();
        if (++iteration == options_.max_iterations) {
          break;
        }
        if (!relocate_seeds()) {
          break;
        }
        reset_charts();
      }
    }
    return finalize();
  }

 private:
  struct Chart {
    Basis basis;
    double area = 0.0;
    double boundary_length = 0.0;
    Vec3 centroid_sum{0.0, 0.0, 0.0};
    Vec3 centroid{0.0, 0.0, 0.0};
    std::vector<int64_t> faces;
    GrowthCostQueue candidates;
    std::set<int64_t> failed_regions;
    int64_t seed = 0;
    bool alive = true;
  };

  const mesh_common::MeshData &mesh_;
  const std::vector<int64_t> &cluster_ids_;
  const GrowthOptions options_;

  // Static topology (cluster-filtered manifold adjacency).
  std::vector<Vec3> face_normals_;
  std::vector<double> face_areas_;
  std::vector<std::array<int64_t, 3>> opposite_face_;  // -1: boundary/non-manifold/cross-cluster
  std::vector<std::array<double, 3>> corner_edge_lengths_;

  // Planar regions (xatlas PlanarCharts): connected same-cluster faces whose
  // normals are componentwise equal within kPlanarNormalEpsilon.
  std::vector<int64_t> region_of_face_;
  std::vector<std::vector<int64_t>> region_faces_;  // ascending face ids
  std::vector<double> region_areas_;

  // Growth state.
  std::vector<Chart> charts_;
  std::vector<int64_t> face_chart_;
  std::vector<Vec2d> corner_uvs_;
  int64_t faces_left_ = 0;

  int64_t place_seed_chart_count_ = 0;
  int64_t fill_hole_chart_count_ = 0;
  int64_t growth_merge_count_ = 0;
  int64_t seed_relocation_count_ = 0;
  int64_t failed_add_count_ = 0;
  int64_t mirrored_chart_normalized_count_ = 0;

  void build_topology() {
    const int64_t face_count = static_cast<int64_t>(mesh_.faces.size());
    face_normals_.reserve(static_cast<size_t>(face_count));
    face_areas_.reserve(static_cast<size_t>(face_count));
    opposite_face_.assign(static_cast<size_t>(face_count), {-1, -1, -1});
    corner_edge_lengths_.assign(static_cast<size_t>(face_count), {0.0, 0.0, 0.0});
    for (int64_t fi = 0; fi < face_count; ++fi) {
      const auto &face = mesh_.faces[static_cast<size_t>(fi)];
      const Vec3 a = vertex_position(mesh_, face[0]);
      const Vec3 b = vertex_position(mesh_, face[1]);
      const Vec3 c = vertex_position(mesh_, face[2]);
      const Vec3 cross = vec_cross(vec_sub(b, a), vec_sub(c, a));
      face_normals_.push_back(normalized_or_zero(cross));
      const double cross_norm = norm(cross);
      face_areas_.push_back(std::isfinite(cross_norm) ? 0.5 * cross_norm : 0.0);
      for (int corner = 0; corner < 3; ++corner) {
        const int64_t v0 = face[static_cast<size_t>(corner)];
        const int64_t v1 = face[static_cast<size_t>((corner + 1) % 3)];
        corner_edge_lengths_[static_cast<size_t>(fi)][static_cast<size_t>(corner)] =
            edge_length(mesh_, v0, v1);
      }
    }
    // Edge map -> (face, corner) incidences; opposite only for manifold
    // same-cluster pairs.
    std::map<mesh_common::EdgeKey, std::vector<std::pair<int64_t, int>>, EdgeKeyLess> edge_map;
    for (int64_t fi = 0; fi < face_count; ++fi) {
      const auto &face = mesh_.faces[static_cast<size_t>(fi)];
      for (int corner = 0; corner < 3; ++corner) {
        const int64_t v0 = face[static_cast<size_t>(corner)];
        const int64_t v1 = face[static_cast<size_t>((corner + 1) % 3)];
        edge_map[mesh_common::edge_key(v0, v1)].emplace_back(fi, corner);
      }
    }
    for (const auto &[key, incidences] : edge_map) {
      if (incidences.size() != 2) {
        continue;
      }
      const auto &[f0, c0] = incidences[0];
      const auto &[f1, c1] = incidences[1];
      if (cluster_ids_[static_cast<size_t>(f0)] != cluster_ids_[static_cast<size_t>(f1)]) {
        continue;
      }
      opposite_face_[static_cast<size_t>(f0)][static_cast<size_t>(c0)] = f1;
      opposite_face_[static_cast<size_t>(f1)][static_cast<size_t>(c1)] = f0;
    }
  }

  static bool normals_equal(const Vec3 &a, const Vec3 &b) {
    return std::fabs(a[0] - b[0]) <= kPlanarNormalEpsilon &&
        std::fabs(a[1] - b[1]) <= kPlanarNormalEpsilon &&
        std::fabs(a[2] - b[2]) <= kPlanarNormalEpsilon;
  }

  void build_planar_regions() {
    const int64_t face_count = static_cast<int64_t>(mesh_.faces.size());
    mesh_common::UnionFind regions(static_cast<size_t>(face_count));
    for (int64_t fi = 0; fi < face_count; ++fi) {
      for (int corner = 0; corner < 3; ++corner) {
        const int64_t of = opposite_face_[static_cast<size_t>(fi)][static_cast<size_t>(corner)];
        if (of < 0 || of <= fi) {
          continue;
        }
        if (normals_equal(
                face_normals_[static_cast<size_t>(fi)], face_normals_[static_cast<size_t>(of)])) {
          regions.unite(static_cast<size_t>(fi), static_cast<size_t>(of));
        }
      }
    }
    region_of_face_.assign(static_cast<size_t>(face_count), -1);
    std::map<int64_t, int64_t> root_to_region;
    for (int64_t fi = 0; fi < face_count; ++fi) {
      const int64_t root = static_cast<int64_t>(regions.find(static_cast<size_t>(fi)));
      auto found = root_to_region.find(root);
      if (found == root_to_region.end()) {
        found = root_to_region.emplace(root, static_cast<int64_t>(region_faces_.size())).first;
        region_faces_.emplace_back();
        region_areas_.push_back(0.0);
      }
      region_of_face_[static_cast<size_t>(fi)] = found->second;
      region_faces_[static_cast<size_t>(found->second)].push_back(fi);
      region_areas_[static_cast<size_t>(found->second)] += face_areas_[static_cast<size_t>(fi)];
    }
  }

  // ---- metrics (computeCost family) --------------------------------------

  double compute_area(const Chart &chart, int64_t first_face) const {
    double area = chart.area;
    for (const int64_t f : region_faces_[static_cast<size_t>(region_of_face_[static_cast<size_t>(first_face)])]) {
      area += face_areas_[static_cast<size_t>(f)];
    }
    return area;
  }

  double compute_boundary_length(const Chart &chart, int64_t first_face, int64_t chart_index) const {
    double boundary_length = chart.boundary_length;
    const int64_t region = region_of_face_[static_cast<size_t>(first_face)];
    for (const int64_t f : region_faces_[static_cast<size_t>(region)]) {
      for (int corner = 0; corner < 3; ++corner) {
        const double l = corner_edge_lengths_[static_cast<size_t>(f)][static_cast<size_t>(corner)];
        const int64_t of = opposite_face_[static_cast<size_t>(f)][static_cast<size_t>(corner)];
        if (of < 0) {
          boundary_length += l;
        } else if (region_of_face_[static_cast<size_t>(of)] != region) {
          if (face_chart_[static_cast<size_t>(of)] != chart_index) {
            boundary_length += l;
          } else {
            boundary_length -= l;
          }
        }
      }
    }
    return std::max(0.0, boundary_length);
  }

  double normal_deviation_metric(const Chart &chart, int64_t face) const {
    return std::min(1.0 - dot(face_normals_[static_cast<size_t>(face)], chart.basis.normal), 1.0);
  }

  double straightness_metric(const Chart &chart, int64_t first_face, int64_t chart_index) const {
    double l_out = 0.0;
    double l_in = 0.0;
    const int64_t region = region_of_face_[static_cast<size_t>(first_face)];
    for (const int64_t f : region_faces_[static_cast<size_t>(region)]) {
      for (int corner = 0; corner < 3; ++corner) {
        const double l = corner_edge_lengths_[static_cast<size_t>(f)][static_cast<size_t>(corner)];
        const int64_t of = opposite_face_[static_cast<size_t>(f)][static_cast<size_t>(corner)];
        if (of < 0) {
          l_out += l;
        } else if (region_of_face_[static_cast<size_t>(of)] != region) {
          if (face_chart_[static_cast<size_t>(of)] != chart_index) {
            l_out += l;
          } else {
            l_in += l;
          }
        }
      }
    }
    const double total = l_out + l_in;
    if (total <= 0.0) {
      return 0.0;
    }
    return std::min((l_out - l_in) / total, 0.0);
  }

  double compute_cost(const Chart &chart, int64_t chart_index, int64_t face) const {
    const double new_area = compute_area(chart, face);
    const double new_boundary = compute_boundary_length(chart, face, chart_index);
    if (options_.max_chart_area > 0.0 && new_area > options_.max_chart_area) {
      return std::numeric_limits<double>::infinity();
    }
    if (options_.max_boundary_length > 0.0 && new_boundary > options_.max_boundary_length) {
      return std::numeric_limits<double>::infinity();
    }
    const double normal_deviation = normal_deviation_metric(chart, face);
    if (normal_deviation >= kNormalDeviationCutoff) {
      return std::numeric_limits<double>::infinity();
    }
    double cost = options_.normal_deviation_weight * normal_deviation;
    // Deviation 1: seam metrics are exactly 0 on welded indexed input.
    const double old_roundness =
        chart.boundary_length * chart.boundary_length / std::max(chart.area, 1e-300);
    const double new_roundness = new_boundary * new_boundary / std::max(new_area, 1e-300);
    cost += options_.roundness_weight * (1.0 - old_roundness / std::max(new_roundness, 1e-300));
    cost += options_.straightness_weight * straightness_metric(chart, face, chart_index);
    if (!std::isfinite(cost)) {
      return std::numeric_limits<double>::infinity();
    }
    return cost;
  }

  // ---- parameterization + validity ----------------------------------------

  void parameterize_chart(const Chart &chart) {
    for (const int64_t f : chart.faces) {
      const auto &face = mesh_.faces[static_cast<size_t>(f)];
      for (int corner = 0; corner < 3; ++corner) {
        const Vec3 p = vertex_position(mesh_, face[static_cast<size_t>(corner)]);
        corner_uvs_[static_cast<size_t>(f) * 3 + static_cast<size_t>(corner)] =
            Vec2d{dot(chart.basis.tangent, p), dot(chart.basis.bitangent, p)};
      }
    }
  }

  double signed_uv_area(int64_t face) const {
    const Vec2d &a = corner_uvs_[static_cast<size_t>(face) * 3 + 0];
    const Vec2d &b = corner_uvs_[static_cast<size_t>(face) * 3 + 1];
    const Vec2d &c = corner_uvs_[static_cast<size_t>(face) * 3 + 2];
    return 0.5 * ((b.u - a.u) * (c.v - a.v) - (c.u - a.u) * (b.v - a.v));
  }

  static bool segments_intersect(
      const Vec2d &a0, const Vec2d &a1, const Vec2d &b0, const Vec2d &b1) {
    const double eps = kBoundaryIntersectEpsilon;
    const double min_ax = std::min(a0.u, a1.u), max_ax = std::max(a0.u, a1.u);
    const double min_ay = std::min(a0.v, a1.v), max_ay = std::max(a0.v, a1.v);
    const double min_bx = std::min(b0.u, b1.u), max_bx = std::max(b0.u, b1.u);
    const double min_by = std::min(b0.v, b1.v), max_by = std::max(b0.v, b1.v);
    if (min_ax > max_bx + eps || min_bx > max_ax + eps ||
        min_ay > max_by + eps || min_by > max_ay + eps) {
      return false;
    }
    const auto orient = [](const Vec2d &p, const Vec2d &q, const Vec2d &r) {
      return (q.u - p.u) * (r.v - p.v) - (q.v - p.v) * (r.u - p.u);
    };
    const double o1 = orient(a0, a1, b0);
    const double o2 = orient(a0, a1, b1);
    const double o3 = orient(b0, b1, a0);
    const double o4 = orient(b0, b1, a1);
    // Strict interior crossing; eps-touching does not count.
    return ((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps)) &&
        ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps));
  }

  bool chart_parameterization_valid(const Chart &chart, int64_t chart_index) const {
    // Flips: OK only when none or all faces are flipped (mirrored chart).
    int64_t flipped = 0;
    for (const int64_t f : chart.faces) {
      if (signed_uv_area(f) < 0.0) {
        flipped += 1;
      }
    }
    if (flipped != 0 && flipped != static_cast<int64_t>(chart.faces.size())) {
      return false;
    }
    // Boundary self-intersection (deviation 3: pairwise with bbox cull).
    struct BoundaryEdge {
      Vec2d p0, p1;
      int64_t v0 = 0, v1 = 0;
    };
    std::vector<BoundaryEdge> boundary;
    for (const int64_t f : chart.faces) {
      const auto &face = mesh_.faces[static_cast<size_t>(f)];
      for (int corner = 0; corner < 3; ++corner) {
        const int64_t of = opposite_face_[static_cast<size_t>(f)][static_cast<size_t>(corner)];
        if (of >= 0 && face_chart_[static_cast<size_t>(of)] == chart_index) {
          continue;
        }
        boundary.push_back(BoundaryEdge{
            corner_uvs_[static_cast<size_t>(f) * 3 + static_cast<size_t>(corner)],
            corner_uvs_[static_cast<size_t>(f) * 3 + static_cast<size_t>((corner + 1) % 3)],
            face[static_cast<size_t>(corner)],
            face[static_cast<size_t>((corner + 1) % 3)]});
      }
    }
    for (size_t i = 0; i < boundary.size(); ++i) {
      for (size_t j = i + 1; j < boundary.size(); ++j) {
        const BoundaryEdge &a = boundary[i];
        const BoundaryEdge &b = boundary[j];
        if (a.v0 == b.v0 || a.v0 == b.v1 || a.v1 == b.v0 || a.v1 == b.v1) {
          continue;  // edges sharing a mesh vertex never count
        }
        if (segments_intersect(a.p0, a.p1, b.p0, b.p1)) {
          return false;
        }
      }
    }
    return true;
  }

  // ---- chart growth (reference addFaceToChart) -----------------------------

  bool add_face_to_chart(int64_t chart_index, int64_t face) {
    Chart &chart = charts_[static_cast<size_t>(chart_index)];
    const size_t old_count = chart.faces.size();
    const bool first_face = old_count == 0;
    // Append the face's whole planar region, rotated to start at `face`.
    const auto &region =
        region_faces_[static_cast<size_t>(region_of_face_[static_cast<size_t>(face)])];
    const auto start = std::find(region.begin(), region.end(), face);
    chart.faces.insert(chart.faces.end(), start, region.end());
    chart.faces.insert(chart.faces.end(), region.begin(), start);
    const size_t face_count = chart.faces.size();

    Basis basis;
    if (first_face) {
      basis.normal = face_normals_[static_cast<size_t>(face)];
      const auto &fv = mesh_.faces[static_cast<size_t>(face)];
      basis.tangent = normalized_or_zero(
          vec_sub(vertex_position(mesh_, fv[0]), vertex_position(mesh_, fv[1])));
      basis.bitangent = vec_cross(basis.normal, basis.tangent);
      if (norm(basis.normal) < 0.5 || norm(basis.tangent) < 0.5) {
        chart.faces.resize(old_count);
        return false;
      }
    } else {
      std::vector<Vec3> points;
      points.reserve(face_count * 3);
      for (const int64_t f : chart.faces) {
        const auto &fv = mesh_.faces[static_cast<size_t>(f)];
        points.push_back(vertex_position(mesh_, fv[0]));
        points.push_back(vertex_position(mesh_, fv[1]));
        points.push_back(vertex_position(mesh_, fv[2]));
      }
      if (!least_squares_basis(points, &basis)) {
        chart.faces.resize(old_count);
        return false;
      }
      if (dot(basis.normal, face_normals_[static_cast<size_t>(face)]) < 0.0) {
        basis.normal = Vec3{-basis.normal[0], -basis.normal[1], -basis.normal[2]};
      }
    }
    if (!first_face) {
      const Basis saved = chart.basis;
      chart.basis = basis;
      parameterize_chart(chart);
      for (size_t i = old_count; i < face_count; ++i) {
        face_chart_[static_cast<size_t>(chart.faces[i])] = chart_index;
      }
      if (!chart_parameterization_valid(chart, chart_index)) {
        for (size_t i = old_count; i < face_count; ++i) {
          face_chart_[static_cast<size_t>(chart.faces[i])] = -1;
        }
        chart.faces.resize(old_count);
        chart.basis = saved;
        return false;
      }
    }
    chart.basis = basis;
    chart.area = compute_area_pre_add(chart, old_count);
    chart.boundary_length = compute_boundary_pre_add(chart, old_count, chart_index, first_face);
    for (size_t i = old_count; i < face_count; ++i) {
      const int64_t f = chart.faces[i];
      face_chart_[static_cast<size_t>(f)] = chart_index;
      faces_left_ -= 1;
      const auto &fv = mesh_.faces[static_cast<size_t>(f)];
      const Vec3 center{
          (vertex_position(mesh_, fv[0])[0] + vertex_position(mesh_, fv[1])[0] +
           vertex_position(mesh_, fv[2])[0]) / 3.0,
          (vertex_position(mesh_, fv[0])[1] + vertex_position(mesh_, fv[1])[1] +
           vertex_position(mesh_, fv[2])[1]) / 3.0,
          (vertex_position(mesh_, fv[0])[2] + vertex_position(mesh_, fv[1])[2] +
           vertex_position(mesh_, fv[2])[2]) / 3.0,
      };
      chart.centroid_sum[0] += center[0];
      chart.centroid_sum[1] += center[1];
      chart.centroid_sum[2] += center[2];
    }
    const double inv = 1.0 / static_cast<double>(chart.faces.size());
    chart.centroid = Vec3{
        chart.centroid_sum[0] * inv, chart.centroid_sum[1] * inv, chart.centroid_sum[2] * inv};
    refresh_candidates(chart_index);
    return true;
  }

  // chart.area/boundary updates must mirror computeArea/computeBoundaryLength
  // exactly: both are evaluated as if the region were not yet added, which is
  // why the snapshot happens against old_count state below.
  double compute_area_pre_add(const Chart &chart, size_t old_count) const {
    double area = 0.0;
    for (size_t i = 0; i < old_count; ++i) {
      area += face_areas_[static_cast<size_t>(chart.faces[i])];
    }
    for (size_t i = old_count; i < chart.faces.size(); ++i) {
      area += face_areas_[static_cast<size_t>(chart.faces[i])];
    }
    return area;
  }

  double compute_boundary_pre_add(
      const Chart &chart, size_t old_count, int64_t chart_index, bool first_face) const {
    (void)first_face;
    // Recompute the chart boundary from scratch over current membership
    // (face_chart_ already set for old faces; new faces counted via region
    // walk like the reference incremental form). Chart sizes are small.
    double boundary = 0.0;
    for (const int64_t f : chart.faces) {
      for (int corner = 0; corner < 3; ++corner) {
        const double l = corner_edge_lengths_[static_cast<size_t>(f)][static_cast<size_t>(corner)];
        const int64_t of = opposite_face_[static_cast<size_t>(f)][static_cast<size_t>(corner)];
        bool inside = false;
        if (of >= 0) {
          if (face_chart_[static_cast<size_t>(of)] == chart_index) {
            inside = true;
          } else {
            for (size_t i = old_count; i < chart.faces.size() && !inside; ++i) {
              inside = chart.faces[i] == of;
            }
          }
        }
        if (!inside) {
          boundary += l;
        }
      }
    }
    return boundary;
  }

  void refresh_candidates(int64_t chart_index) {
    Chart &chart = charts_[static_cast<size_t>(chart_index)];
    chart.candidates.clear();
    for (const int64_t f : chart.faces) {
      for (int corner = 0; corner < 3; ++corner) {
        const int64_t of = opposite_face_[static_cast<size_t>(f)][static_cast<size_t>(corner)];
        if (of < 0 || face_chart_[static_cast<size_t>(of)] >= 0) {
          continue;
        }
        if (chart.failed_regions.count(region_of_face_[static_cast<size_t>(of)]) > 0) {
          continue;
        }
        const double cost = compute_cost(chart, chart_index, of);
        if (std::isfinite(cost)) {
          chart.candidates.push(cost, of);
        }
      }
    }
  }

  // ---- growth phases -------------------------------------------------------

  void create_chart(double threshold) {
    const int64_t chart_index = static_cast<int64_t>(charts_.size());
    charts_.emplace_back();
    Chart &chart = charts_.back();
    // Seed: first unassigned face of the largest-area planar region (strict >
    // scan in ascending face order, as the reference).
    chart.seed = 0;
    double largest_area = 0.0;
    for (int64_t f = 0; f < static_cast<int64_t>(mesh_.faces.size()); ++f) {
      if (face_chart_[static_cast<size_t>(f)] >= 0) {
        continue;
      }
      const double area = region_areas_[static_cast<size_t>(region_of_face_[static_cast<size_t>(f)])];
      if (area > largest_area) {
        largest_area = area;
        chart.seed = f;
      }
    }
    if (!add_face_to_chart(chart_index, chart.seed)) {
      // Degenerate seed rescue (no reference analog: xatlas asserts instead).
      // Force-assign the seed's whole region with zero UVs so place_seeds /
      // fill_holes cannot loop forever; the chart fails acceptance and is
      // routed to LSCM. Deterministic and bounded.
      failed_add_count_ += 1;
      Chart &failed = charts_[static_cast<size_t>(chart_index)];
      const auto &region =
          region_faces_[static_cast<size_t>(region_of_face_[static_cast<size_t>(failed.seed)])];
      for (const int64_t f : region) {
        if (face_chart_[static_cast<size_t>(f)] < 0) {
          failed.faces.push_back(f);
          face_chart_[static_cast<size_t>(f)] = chart_index;
          faces_left_ -= 1;
        }
      }
      return;
    }
    for (;;) {
      Chart &current = charts_[static_cast<size_t>(chart_index)];
      if (current.candidates.empty() || current.candidates.peek_cost() > threshold) {
        break;
      }
      const int64_t f = current.candidates.pop();
      if (face_chart_[static_cast<size_t>(f)] >= 0) {
        continue;
      }
      if (!add_face_to_chart(chart_index, f)) {
        failed_add_count_ += 1;
        charts_[static_cast<size_t>(chart_index)].failed_regions.insert(
            region_of_face_[static_cast<size_t>(f)]);
        continue;
      }
    }
  }

  void place_seeds(double threshold) {
    while (faces_left_ > 0) {
      create_chart(threshold);
    }
  }

  void fill_holes(double threshold) {
    while (faces_left_ > 0) {
      create_chart(threshold);
    }
  }

  bool relocate_seed(Chart &chart) {
    GrowthCostQueue best_triangles(10);
    for (const int64_t f : chart.faces) {
      best_triangles.push(normal_deviation_metric(chart, f), f);
    }
    int64_t most_central = chart.faces.empty() ? chart.seed : chart.faces[0];
    double min_distance = std::numeric_limits<double>::max();
    while (best_triangles.count() > 0) {
      const int64_t face = best_triangles.pop();
      const auto &fv = mesh_.faces[static_cast<size_t>(face)];
      const Vec3 center{
          (vertex_position(mesh_, fv[0])[0] + vertex_position(mesh_, fv[1])[0] +
           vertex_position(mesh_, fv[2])[0]) / 3.0,
          (vertex_position(mesh_, fv[0])[1] + vertex_position(mesh_, fv[1])[1] +
           vertex_position(mesh_, fv[2])[1]) / 3.0,
          (vertex_position(mesh_, fv[0])[2] + vertex_position(mesh_, fv[1])[2] +
           vertex_position(mesh_, fv[2])[2]) / 3.0,
      };
      const Vec3 diff = vec_sub(chart.centroid, center);
      const double distance = norm(diff);
      if (distance < min_distance) {
        min_distance = distance;
        most_central = face;
      }
    }
    if (most_central == chart.seed) {
      return false;
    }
    chart.seed = most_central;
    return true;
  }

  bool relocate_seeds() {
    bool any = false;
    for (Chart &chart : charts_) {
      if (chart.alive && relocate_seed(chart)) {
        any = true;
        seed_relocation_count_ += 1;
      }
    }
    return any;
  }

  void reset_charts() {
    const int64_t face_count = static_cast<int64_t>(mesh_.faces.size());
    face_chart_.assign(static_cast<size_t>(face_count), -1);
    faces_left_ = face_count;
    for (int64_t i = 0; i < static_cast<int64_t>(charts_.size()); ++i) {
      Chart &chart = charts_[static_cast<size_t>(i)];
      if (!chart.alive) {
        continue;  // merged away (reference removes these before any reset)
      }
      chart.area = 0.0;
      chart.boundary_length = 0.0;
      chart.basis = Basis{};
      chart.centroid_sum = Vec3{0.0, 0.0, 0.0};
      chart.centroid = Vec3{0.0, 0.0, 0.0};
      chart.faces.clear();
      chart.candidates.clear();
      chart.failed_regions.clear();
      if (!add_face_to_chart(i, chart.seed)) {
        failed_add_count_ += 1;
      }
    }
  }

  void grow_charts(double threshold) {
    for (;;) {
      if (faces_left_ == 0) {
        break;
      }
      int64_t best_face = -1;
      int64_t best_chart = -1;
      double lowest_cost = std::numeric_limits<double>::max();
      for (int64_t i = 0; i < static_cast<int64_t>(charts_.size()); ++i) {
        Chart &chart = charts_[static_cast<size_t>(i)];
        int64_t face = -1;
        double cost = std::numeric_limits<double>::max();
        for (;;) {
          if (chart.candidates.count() == 0) {
            break;
          }
          cost = chart.candidates.peek_cost();
          face = chart.candidates.peek_face();
          if (face_chart_[static_cast<size_t>(face)] < 0) {
            break;
          }
          chart.candidates.pop();  // claimed by another chart
          face = -1;
        }
        if (face < 0) {
          continue;
        }
        if (cost < lowest_cost) {
          lowest_cost = cost;
          best_face = face;
          best_chart = i;
        }
      }
      if (best_face < 0 || lowest_cost > threshold) {
        break;
      }
      charts_[static_cast<size_t>(best_chart)].candidates.pop();
      if (!add_face_to_chart(best_chart, best_face)) {
        failed_add_count_ += 1;
        charts_[static_cast<size_t>(best_chart)].failed_regions.insert(
            region_of_face_[static_cast<size_t>(best_face)]);
      }
    }
  }

  // ---- merge phase (reference mergeCharts) ---------------------------------

  bool merge_chart(int64_t owner_index, int64_t other_index, double shared_length) {
    Chart &owner = charts_[static_cast<size_t>(owner_index)];
    Chart &other = charts_[static_cast<size_t>(other_index)];
    const size_t old_count = owner.faces.size();
    owner.faces.insert(owner.faces.end(), other.faces.begin(), other.faces.end());
    for (const int64_t f : other.faces) {
      face_chart_[static_cast<size_t>(f)] = owner_index;
    }
    Basis basis;
    std::vector<Vec3> points;
    points.reserve(owner.faces.size() * 3);
    for (const int64_t f : owner.faces) {
      const auto &fv = mesh_.faces[static_cast<size_t>(f)];
      points.push_back(vertex_position(mesh_, fv[0]));
      points.push_back(vertex_position(mesh_, fv[1]));
      points.push_back(vertex_position(mesh_, fv[2]));
    }
    const auto revert = [&]() {
      owner.faces.resize(old_count);
      for (const int64_t f : other.faces) {
        face_chart_[static_cast<size_t>(f)] = other_index;
      }
    };
    if (!least_squares_basis(points, &basis)) {
      revert();
      return false;
    }
    if (dot(basis.normal, face_normals_[static_cast<size_t>(owner.faces[0])]) < 0.0) {
      basis.normal = Vec3{-basis.normal[0], -basis.normal[1], -basis.normal[2]};
    }
    const Basis saved = owner.basis;
    owner.basis = basis;
    parameterize_chart(owner);
    if (!chart_parameterization_valid(owner, owner_index)) {
      owner.basis = saved;
      revert();
      return false;
    }
    for (const int64_t region : other.failed_regions) {
      owner.failed_regions.insert(region);
    }
    owner.area += other.area;
    // Reference mergeChart bookkeeping (deviation 5): minus ONE shared length.
    owner.boundary_length += other.boundary_length - shared_length;
    owner.centroid_sum[0] += other.centroid_sum[0];
    owner.centroid_sum[1] += other.centroid_sum[1];
    owner.centroid_sum[2] += other.centroid_sum[2];
    const double inv = 1.0 / static_cast<double>(owner.faces.size());
    owner.centroid = Vec3{
        owner.centroid_sum[0] * inv, owner.centroid_sum[1] * inv, owner.centroid_sum[2] * inv};
    other.alive = false;
    other.faces.clear();
    growth_merge_count_ += 1;
    return true;
  }

  void merge_charts() {
    const int64_t chart_count = static_cast<int64_t>(charts_.size());
    for (;;) {
      bool merged = false;
      for (int64_t c = chart_count - 1; c >= 0; --c) {
        Chart &chart = charts_[static_cast<size_t>(c)];
        if (!chart.alive) {
          continue;
        }
        double external_boundary = 0.0;
        std::vector<double> shared(static_cast<size_t>(chart_count), 0.0);
        std::vector<int64_t> shared_edge_count(static_cast<size_t>(chart_count), 0);
        for (const int64_t f : chart.faces) {
          for (int corner = 0; corner < 3; ++corner) {
            const double l =
                corner_edge_lengths_[static_cast<size_t>(f)][static_cast<size_t>(corner)];
            const int64_t of =
                opposite_face_[static_cast<size_t>(f)][static_cast<size_t>(corner)];
            if (of < 0) {
              external_boundary += l;
              continue;
            }
            const int64_t neighbor = face_chart_[static_cast<size_t>(of)];
            if (neighbor < 0) {
              external_boundary += l;
            } else if (neighbor != c) {
              // Welded input: no seams, so shared == sharedNoSeams.
              shared[static_cast<size_t>(neighbor)] += l;
              shared_edge_count[static_cast<size_t>(neighbor)] += 1;
            }
          }
        }
        for (int64_t cc = chart_count - 1; cc >= 0; --cc) {
          if (cc == c) {
            continue;
          }
          Chart &other = charts_[static_cast<size_t>(cc)];
          if (!other.alive || shared[static_cast<size_t>(cc)] <= 0.0) {
            continue;
          }
          if (dot(other.basis.normal, chart.basis.normal) < kMergeMinNormalDot) {
            continue;
          }
          if (options_.max_chart_area > 0.0 &&
              chart.area + other.area > options_.max_chart_area) {
            continue;
          }
          if (options_.max_boundary_length > 0.0 &&
              chart.boundary_length + other.boundary_length - shared[static_cast<size_t>(cc)] >
                  options_.max_boundary_length) {
            continue;
          }
          const double s = shared[static_cast<size_t>(cc)];
          const bool rule_single = chart.faces.size() > 1 && other.faces.size() == 1 &&
              other.area <= chart.area * 0.1;
          const bool rule_quad =
              other.faces.size() == 2 && shared_edge_count[static_cast<size_t>(cc)] >= 2;
          const bool rule_inside = std::fabs(s - other.boundary_length) <= kXatlasEpsilon;
          const bool rule_fraction =
              s > 0.2 * std::max(0.0, chart.boundary_length - external_boundary) ||
              s > 0.75 * other.boundary_length;
          if (!(rule_single || rule_quad || rule_inside || rule_fraction)) {
            continue;
          }
          if (!merge_chart(c, cc, s)) {
            continue;
          }
          merged = true;
          break;
        }
        if (merged) {
          break;
        }
      }
      if (!merged) {
        break;
      }
    }
  }

  // ---- finalize ------------------------------------------------------------

  ChartBuildResult finalize() {
    // Compact alive charts in ascending index order (reference removeAt).
    std::vector<int64_t> dense(charts_.size(), -1);
    std::vector<int64_t> alive_indices;
    for (size_t i = 0; i < charts_.size(); ++i) {
      if (charts_[i].alive && !charts_[i].faces.empty()) {
        dense[i] = static_cast<int64_t>(alive_indices.size());
        alive_indices.push_back(static_cast<int64_t>(i));
      }
    }
    ChartBuildResult result;
    result.chart_count = static_cast<int64_t>(alive_indices.size());
    result.chart_ids.assign(mesh_.faces.size(), -1);
    for (size_t f = 0; f < mesh_.faces.size(); ++f) {
      const int64_t raw = face_chart_[f];
      result.chart_ids[f] = raw >= 0 ? dense[static_cast<size_t>(raw)] : -1;
    }

    result.chart_face_counts.assign(static_cast<size_t>(result.chart_count), 0);
    result.chart_stretch_l2.assign(static_cast<size_t>(result.chart_count), 0.0);
    result.chart_stretch_linf.assign(static_cast<size_t>(result.chart_count), 0.0);
    result.chart_accepted.assign(static_cast<size_t>(result.chart_count), 0);
    result.chart_needs_lscm.assign(static_cast<size_t>(result.chart_count), 0);

    for (size_t out = 0; out < alive_indices.size(); ++out) {
      Chart &chart = charts_[static_cast<size_t>(alive_indices[out])];
      // Final projection with the final basis (growth texcoords are scratch).
      parameterize_chart(chart);
      // Orientation normalization: a fully mirrored chart (xatlas validity
      // allows all-flipped) is unmirrored by negating u — our zero-flip
      // output invariant, unlike the reference (anchors record ~50% mirrored
      // reference charts; mirroring carries no parity signal).
      int64_t flipped = 0;
      int64_t measurable = 0;
      for (const int64_t f : chart.faces) {
        const double area = signed_uv_area(f);
        if (std::fabs(area) > kUvAreaEpsilon) {
          measurable += 1;
          if (area < 0.0) {
            flipped += 1;
          }
        }
      }
      if (measurable > 0 && flipped == measurable) {
        for (const int64_t f : chart.faces) {
          for (int corner = 0; corner < 3; ++corner) {
            corner_uvs_[static_cast<size_t>(f) * 3 + static_cast<size_t>(corner)].u *= -1.0;
          }
        }
        flipped = 0;
        mirrored_chart_normalized_count_ += 1;
      }
      // Sander stretch (uv -> 3D), 3D-area weighted, skipping degenerates.
      double weighted_l2_sq = 0.0;
      double total_area3d = 0.0;
      double linf = 0.0;
      for (const int64_t f : chart.faces) {
        const double uv_area = signed_uv_area(f);
        const double area3d = face_areas_[static_cast<size_t>(f)];
        if (std::fabs(uv_area) <= kUvAreaEpsilon || area3d <= 0.0 || uv_area < 0.0) {
          continue;
        }
        const auto &fv = mesh_.faces[static_cast<size_t>(f)];
        const Vec3 q0 = vertex_position(mesh_, fv[0]);
        const Vec3 q1 = vertex_position(mesh_, fv[1]);
        const Vec3 q2 = vertex_position(mesh_, fv[2]);
        const Vec2d &t0 = corner_uvs_[static_cast<size_t>(f) * 3 + 0];
        const Vec2d &t1 = corner_uvs_[static_cast<size_t>(f) * 3 + 1];
        const Vec2d &t2 = corner_uvs_[static_cast<size_t>(f) * 3 + 2];
        const double inv2a = 1.0 / (2.0 * uv_area);
        Vec3 ss{0.0, 0.0, 0.0};
        Vec3 st{0.0, 0.0, 0.0};
        for (int axis = 0; axis < 3; ++axis) {
          ss[static_cast<size_t>(axis)] = (q0[static_cast<size_t>(axis)] * (t1.v - t2.v) +
                                           q1[static_cast<size_t>(axis)] * (t2.v - t0.v) +
                                           q2[static_cast<size_t>(axis)] * (t0.v - t1.v)) * inv2a;
          st[static_cast<size_t>(axis)] = (q0[static_cast<size_t>(axis)] * (t2.u - t1.u) +
                                           q1[static_cast<size_t>(axis)] * (t0.u - t2.u) +
                                           q2[static_cast<size_t>(axis)] * (t1.u - t0.u)) * inv2a;
        }
        const double a = dot(ss, ss);
        const double b = dot(ss, st);
        const double c2 = dot(st, st);
        const double disc = std::sqrt(std::max(0.0, (a - c2) * (a - c2) + 4.0 * b * b));
        const double gamma = std::sqrt(std::max(0.0, ((a + c2) + disc) * 0.5));
        weighted_l2_sq += ((a + c2) * 0.5) * area3d;
        total_area3d += area3d;
        linf = std::max(linf, gamma);
      }
      const double l2 = total_area3d > 0.0 ? std::sqrt(weighted_l2_sq / total_area3d) : 0.0;
      result.chart_face_counts[out] = static_cast<int64_t>(chart.faces.size());
      result.chart_stretch_l2[out] = l2;
      result.chart_stretch_linf[out] = linf;
      const bool accepted = total_area3d > 0.0 && flipped == 0 && std::isfinite(linf) &&
          linf <= options_.projection_linf_threshold;
      result.chart_accepted[out] = accepted ? 1 : 0;
      result.chart_needs_lscm[out] = accepted ? 0 : 1;
      if (accepted) {
        result.accepted_chart_count += 1;
      } else {
        result.lscm_pending_chart_count += 1;
      }
    }

    result.corner_uvs.reserve(corner_uvs_.size() * 2);
    for (const Vec2d &uv : corner_uvs_) {
      result.corner_uvs.push_back(uv.u);
      result.corner_uvs.push_back(uv.v);
    }
    result.planar_region_count = static_cast<int64_t>(region_faces_.size());
    result.place_seed_chart_count = place_seed_chart_count_;
    result.fill_hole_chart_count = fill_hole_chart_count_;
    result.growth_merge_count = growth_merge_count_;
    result.seed_relocation_count = seed_relocation_count_;
    result.failed_add_count = failed_add_count_;
    result.mirrored_chart_normalized_count = mirrored_chart_normalized_count_;
    return result;
  }
};

// ---------------------------------------------------------------------------
// Stage B parameterization (slice 5): LSCM for charts whose orthographic
// projection fails the flip/overlap/stretch acceptance, plus bounded
// split/shatter repair establishing the zero-overlap invariant per chart.
// Behavior reference: xatlas computeLeastSquaresConformalMap
// (/tmp/CuMesh/third_party/xatlas/xatlas.cpp:6482) — ABF conformal relations
// (setup_abf_relations :6424) with the projected-triangle LSCM fallback
// (:6391), pins from findApproximateDiameterVertices (:6333), solved by
// OpenNL's Jacobi-preconditioned CG on the normal equations warm-started from
// the projection (NL_MAX_ITERATIONS = 5*V). Documented deviations:
//  6. The solver is a matrix-free Jacobi-CG port of OpenNL's path (same
//     system, same pins, same warm start, tolerance 1e-6), not a line port.
//  7. Charts that stay invalid after LSCM are repaired by deterministic
//     longest-axis bisection (depth-capped), then shattered to single-face
//     charts — the reference has no equivalent (its invalid charts ship);
//     this is what buys the zero-overlap invariant the SPEC requires.
//  8. Validity here adds a full interior triangle-overlap SAT check (the
//     growth phase, like the reference, only tests boundary
//     self-intersection); eps-touching does not count as overlap.
// ---------------------------------------------------------------------------

constexpr double kLscmTolerance = 1e-6;

struct ParamOptions {
  double projection_linf_threshold = 1.25;
  int64_t max_split_depth = 3;
  int64_t lscm_iteration_factor = 5;
  // A valid-but-stretchy projection is kept as the fallback when LSCM fails
  // validity — but only below this Linf cap; beyond it the chart is split
  // (unbounded fallback stretch produced fixture charts with Linf > 1000,
  // wrecking the mean stretch parity while p95 stayed ~1.1).
  double fallback_linf_cap = 12.5;
};

// Exact 2D triangle-triangle interior overlap (SAT over the 6 edge normals,
// eps slack so touching does not count). Triangles given as corner arrays.
bool triangles_overlap_2d(const std::array<Vec2d, 3> &a, const std::array<Vec2d, 3> &b) {
  const double eps = 1e-12;
  const auto separated_by_edges_of = [&](const std::array<Vec2d, 3> &p,
                                         const std::array<Vec2d, 3> &q) {
    for (int i = 0; i < 3; ++i) {
      const Vec2d &e0 = p[static_cast<size_t>(i)];
      const Vec2d &e1 = p[static_cast<size_t>((i + 1) % 3)];
      const double nx = -(e1.v - e0.v);
      const double ny = e1.u - e0.u;
      double pmin = std::numeric_limits<double>::max();
      double pmax = std::numeric_limits<double>::lowest();
      double qmin = std::numeric_limits<double>::max();
      double qmax = std::numeric_limits<double>::lowest();
      for (int k = 0; k < 3; ++k) {
        const double pp = nx * p[static_cast<size_t>(k)].u + ny * p[static_cast<size_t>(k)].v;
        const double qq = nx * q[static_cast<size_t>(k)].u + ny * q[static_cast<size_t>(k)].v;
        pmin = std::min(pmin, pp);
        pmax = std::max(pmax, pp);
        qmin = std::min(qmin, qq);
        qmax = std::max(qmax, qq);
      }
      const double scale = std::max({std::fabs(pmin), std::fabs(pmax), std::fabs(qmin),
                                     std::fabs(qmax), 1e-300});
      if (pmax - qmin <= eps * scale || qmax - pmin <= eps * scale) {
        return true;  // separating axis (touching counts as separated)
      }
    }
    return false;
  };
  return !separated_by_edges_of(a, b) && !separated_by_edges_of(b, a);
}

// Sparse least-squares row set with locked variables, solved by
// Jacobi-preconditioned CG on the normal equations (deviation 6).
class LscmSolver {
 public:
  explicit LscmSolver(int64_t variable_count) : variable_count_(variable_count) {}

  void begin_row() { row_starts_.push_back(static_cast<int64_t>(coefficients_.size())); }
  void coefficient(int64_t variable, double value) {
    coefficients_.push_back({variable, value});
  }

  // x: initial values (warm start); locked: mask of locked variables.
  // Returns false when CG failed to reach the tolerance.
  bool solve(std::vector<double> &x, const std::vector<bool> &locked, int64_t max_iterations) {
    row_starts_.push_back(static_cast<int64_t>(coefficients_.size()));
    const int64_t rows = static_cast<int64_t>(row_starts_.size()) - 1;
    // b = -A_locked * x_locked per row; unknowns are the free variables.
    std::vector<double> row_rhs(static_cast<size_t>(rows), 0.0);
    for (int64_t r = 0; r < rows; ++r) {
      double rhs = 0.0;
      for (int64_t c = row_starts_[static_cast<size_t>(r)];
           c < row_starts_[static_cast<size_t>(r) + 1]; ++c) {
        const auto &[variable, value] = coefficients_[static_cast<size_t>(c)];
        if (locked[static_cast<size_t>(variable)]) {
          rhs -= value * x[static_cast<size_t>(variable)];
        }
      }
      row_rhs[static_cast<size_t>(r)] = rhs;
    }
    // Normal equations: M = A_f^T A_f, rhs_n = A_f^T b. Matrix-free apply.
    const auto apply = [&](const std::vector<double> &in, std::vector<double> &out) {
      std::fill(out.begin(), out.end(), 0.0);
      for (int64_t r = 0; r < rows; ++r) {
        double dot_row = 0.0;
        for (int64_t c = row_starts_[static_cast<size_t>(r)];
             c < row_starts_[static_cast<size_t>(r) + 1]; ++c) {
          const auto &[variable, value] = coefficients_[static_cast<size_t>(c)];
          if (!locked[static_cast<size_t>(variable)]) {
            dot_row += value * in[static_cast<size_t>(variable)];
          }
        }
        for (int64_t c = row_starts_[static_cast<size_t>(r)];
             c < row_starts_[static_cast<size_t>(r) + 1]; ++c) {
          const auto &[variable, value] = coefficients_[static_cast<size_t>(c)];
          if (!locked[static_cast<size_t>(variable)]) {
            out[static_cast<size_t>(variable)] += value * dot_row;
          }
        }
      }
    };
    std::vector<double> rhs_n(static_cast<size_t>(variable_count_), 0.0);
    std::vector<double> diag(static_cast<size_t>(variable_count_), 0.0);
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = row_starts_[static_cast<size_t>(r)];
           c < row_starts_[static_cast<size_t>(r) + 1]; ++c) {
        const auto &[variable, value] = coefficients_[static_cast<size_t>(c)];
        if (!locked[static_cast<size_t>(variable)]) {
          rhs_n[static_cast<size_t>(variable)] += value * row_rhs[static_cast<size_t>(r)];
          diag[static_cast<size_t>(variable)] += value * value;
        }
      }
    }
    const auto precondition = [&](const std::vector<double> &in, std::vector<double> &out) {
      for (int64_t i = 0; i < variable_count_; ++i) {
        out[static_cast<size_t>(i)] = diag[static_cast<size_t>(i)] > 0.0
            ? in[static_cast<size_t>(i)] / diag[static_cast<size_t>(i)]
            : 0.0;
      }
    };
    std::vector<double> residual(static_cast<size_t>(variable_count_), 0.0);
    std::vector<double> z(static_cast<size_t>(variable_count_), 0.0);
    std::vector<double> direction(static_cast<size_t>(variable_count_), 0.0);
    std::vector<double> applied(static_cast<size_t>(variable_count_), 0.0);
    apply(x, applied);
    double rhs_norm_sq = 0.0;
    for (int64_t i = 0; i < variable_count_; ++i) {
      if (locked[static_cast<size_t>(i)]) {
        continue;
      }
      residual[static_cast<size_t>(i)] =
          rhs_n[static_cast<size_t>(i)] - applied[static_cast<size_t>(i)];
      rhs_norm_sq += rhs_n[static_cast<size_t>(i)] * rhs_n[static_cast<size_t>(i)];
    }
    const double stop_sq = std::max(rhs_norm_sq, 1e-300) * kLscmTolerance * kLscmTolerance;
    precondition(residual, z);
    direction = z;
    double rz = 0.0;
    double residual_sq = 0.0;
    for (int64_t i = 0; i < variable_count_; ++i) {
      rz += residual[static_cast<size_t>(i)] * z[static_cast<size_t>(i)];
      residual_sq += residual[static_cast<size_t>(i)] * residual[static_cast<size_t>(i)];
    }
    for (int64_t iteration = 0; iteration < max_iterations; ++iteration) {
      if (residual_sq <= stop_sq) {
        return true;
      }
      apply(direction, applied);
      double d_ad = 0.0;
      for (int64_t i = 0; i < variable_count_; ++i) {
        d_ad += direction[static_cast<size_t>(i)] * applied[static_cast<size_t>(i)];
      }
      if (!(d_ad > 0.0) || !std::isfinite(d_ad)) {
        return residual_sq <= stop_sq;
      }
      const double alpha = rz / d_ad;
      residual_sq = 0.0;
      for (int64_t i = 0; i < variable_count_; ++i) {
        if (locked[static_cast<size_t>(i)]) {
          continue;
        }
        x[static_cast<size_t>(i)] += alpha * direction[static_cast<size_t>(i)];
        residual[static_cast<size_t>(i)] -= alpha * applied[static_cast<size_t>(i)];
        residual_sq +=
            residual[static_cast<size_t>(i)] * residual[static_cast<size_t>(i)];
      }
      precondition(residual, z);
      double rz_next = 0.0;
      for (int64_t i = 0; i < variable_count_; ++i) {
        rz_next += residual[static_cast<size_t>(i)] * z[static_cast<size_t>(i)];
      }
      const double beta = rz > 0.0 ? rz_next / rz : 0.0;
      rz = rz_next;
      for (int64_t i = 0; i < variable_count_; ++i) {
        direction[static_cast<size_t>(i)] =
            z[static_cast<size_t>(i)] + beta * direction[static_cast<size_t>(i)];
      }
    }
    return residual_sq <= stop_sq;
  }

 private:
  int64_t variable_count_;
  std::vector<std::pair<int64_t, double>> coefficients_;
  std::vector<int64_t> row_starts_;
};

// xatlas setup_abf_relations (:6424): conformal relations from triangle
// angles; returns false for degenerate angles (caller falls back to the
// projected-triangle LSCM rows).
bool add_abf_rows(LscmSolver &solver, std::array<int64_t, 3> ids, const std::array<Vec3, 3> &p) {
  const auto angle = [](const Vec3 &v1, const Vec3 &v2, const Vec3 &v3) {
    const Vec3 d1 = vec_sub(v1, v2);
    const Vec3 d2 = vec_sub(v3, v2);
    const double denom = norm(d1) * norm(d2);
    if (denom <= 0.0) {
      return 0.0;
    }
    return std::acos(std::clamp(dot(d1, d2) / denom, -1.0, 1.0));
  };
  double a0 = angle(p[2], p[0], p[1]);
  double a1 = angle(p[0], p[1], p[2]);
  double a2 = 3.14159265358979323846 - a1 - a0;
  if (a0 == 0.0 || a1 == 0.0 || a2 == 0.0) {
    return false;
  }
  double s0 = std::sin(a0);
  double s1 = std::sin(a1);
  double s2 = std::sin(a2);
  int64_t id0 = ids[0], id1 = ids[1], id2 = ids[2];
  if (s1 > s0 && s1 > s2) {
    std::swap(s1, s2);
    std::swap(s0, s1);
    std::swap(a1, a2);
    std::swap(a0, a1);
    std::swap(id1, id2);
    std::swap(id0, id1);
  } else if (s0 > s1 && s0 > s2) {
    std::swap(s0, s2);
    std::swap(s0, s1);
    std::swap(a0, a2);
    std::swap(a0, a1);
    std::swap(id0, id2);
    std::swap(id0, id1);
  }
  const double c0 = std::cos(a0);
  const double ratio = s2 == 0.0 ? 1.0 : s1 / s2;
  const double cosine = c0 * ratio;
  const double sine = s0 * ratio;
  solver.begin_row();
  solver.coefficient(2 * id0, cosine - 1.0);
  solver.coefficient(2 * id0 + 1, -sine);
  solver.coefficient(2 * id1, -cosine);
  solver.coefficient(2 * id1 + 1, sine);
  solver.coefficient(2 * id2, 1.0);
  solver.begin_row();
  solver.coefficient(2 * id0, sine);
  solver.coefficient(2 * id0 + 1, cosine - 1.0);
  solver.coefficient(2 * id1, -sine);
  solver.coefficient(2 * id1 + 1, -cosine);
  solver.coefficient(2 * id2 + 1, 1.0);
  return true;
}

// xatlas projectTriangle fallback rows (:6391, b == 0 form).
void add_projected_rows(LscmSolver &solver, const std::array<int64_t, 3> &ids,
                        const std::array<Vec3, 3> &p) {
  const Vec3 x_axis = normalized_or_zero(vec_sub(p[1], p[0]));
  const Vec3 z_axis = normalized_or_zero(vec_cross(x_axis, vec_sub(p[2], p[0])));
  const Vec3 y_axis = vec_cross(z_axis, x_axis);
  const double a = norm(vec_sub(p[1], p[0]));
  const double c = dot(vec_sub(p[2], p[0]), x_axis);
  const double d = dot(vec_sub(p[2], p[0]), y_axis);
  solver.begin_row();
  solver.coefficient(2 * ids[0], -a + c);
  solver.coefficient(2 * ids[0] + 1, -d);
  solver.coefficient(2 * ids[1], -c);
  solver.coefficient(2 * ids[1] + 1, d);
  solver.coefficient(2 * ids[2], a);
  solver.begin_row();
  solver.coefficient(2 * ids[0], d);
  solver.coefficient(2 * ids[0] + 1, -a + c);
  solver.coefficient(2 * ids[1], -d);
  solver.coefficient(2 * ids[1] + 1, -c);
  solver.coefficient(2 * ids[2] + 1, a);
}

struct ParamChartOut {
  std::vector<int64_t> faces;            // global face ids
  std::vector<Vec2d> corner_uvs;         // per corner of `faces`
  int64_t method = 0;                    // 0 projection, 1 lscm, 2 shatter
  double stretch_l2 = 0.0;
  double stretch_linf = 0.0;
};

class ChartParameterizer {
 public:
  ChartParameterizer(const mesh_common::MeshData &mesh, const ParamOptions &options)
      : mesh_(mesh), options_(options) {
    face_areas_.reserve(mesh_.faces.size());
    for (const auto &face : mesh_.faces) {
      const Vec3 a = vertex_position(mesh_, face[0]);
      const Vec3 b = vertex_position(mesh_, face[1]);
      const Vec3 c = vertex_position(mesh_, face[2]);
      const double n = norm(vec_cross(vec_sub(b, a), vec_sub(c, a)));
      face_areas_.push_back(std::isfinite(n) ? 0.5 * n : 0.0);
    }
  }

  // Parameterize one input chart; appends one or more output charts.
  void parameterize(const std::vector<int64_t> &faces, std::vector<ParamChartOut> &out) {
    parameterize_recursive(faces, 0, out);
  }

  int64_t projected_chart_count = 0;
  int64_t projection_fallback_chart_count = 0;
  int64_t lscm_chart_count = 0;
  int64_t shattered_face_chart_count = 0;
  int64_t split_event_count = 0;
  int64_t lscm_unconverged_count = 0;

 private:
  const mesh_common::MeshData &mesh_;
  const ParamOptions options_;
  std::vector<double> face_areas_;

  struct LocalChart {
    std::vector<int64_t> faces;                 // global face ids
    std::vector<int64_t> local_vertices;        // global vertex ids, first-use order
    std::vector<std::array<int64_t, 3>> local_faces;  // local vertex indices
    std::vector<Vec2d> vertex_uvs;              // per local vertex
  };

  LocalChart weld(const std::vector<int64_t> &faces) const {
    LocalChart chart;
    chart.faces = faces;
    std::map<int64_t, int64_t> remap;
    chart.local_faces.reserve(faces.size());
    for (const int64_t f : faces) {
      const auto &fv = mesh_.faces[static_cast<size_t>(f)];
      std::array<int64_t, 3> local{};
      for (int corner = 0; corner < 3; ++corner) {
        const int64_t v = fv[static_cast<size_t>(corner)];
        auto found = remap.find(v);
        if (found == remap.end()) {
          found = remap.emplace(v, static_cast<int64_t>(chart.local_vertices.size())).first;
          chart.local_vertices.push_back(v);
        }
        local[static_cast<size_t>(corner)] = found->second;
      }
      chart.local_faces.push_back(local);
    }
    chart.vertex_uvs.assign(chart.local_vertices.size(), Vec2d{});
    return chart;
  }

  double signed_area(const LocalChart &chart, size_t face_index) const {
    const auto &lf = chart.local_faces[face_index];
    const Vec2d &a = chart.vertex_uvs[static_cast<size_t>(lf[0])];
    const Vec2d &b = chart.vertex_uvs[static_cast<size_t>(lf[1])];
    const Vec2d &c = chart.vertex_uvs[static_cast<size_t>(lf[2])];
    return 0.5 * ((b.u - a.u) * (c.v - a.v) - (c.u - a.u) * (b.v - a.v));
  }

  // Zero-flip + zero-interior-overlap validity over the chart's own faces
  // (deviation 8). Flips are evaluated after orientation normalization by the
  // caller; this checks raw state.
  bool chart_uvs_valid(const LocalChart &chart) const {
    for (size_t i = 0; i < chart.local_faces.size(); ++i) {
      const double area = signed_area(chart, i);
      if (!std::isfinite(area) || area < -kUvAreaEpsilon) {
        return false;
      }
    }
    for (const Vec2d &uv : chart.vertex_uvs) {
      if (!std::isfinite(uv.u) || !std::isfinite(uv.v)) {
        return false;
      }
    }
    // O(T^2) with bbox cull; charts are small.
    std::vector<std::array<Vec2d, 3>> triangles;
    std::vector<std::array<double, 4>> boxes;
    triangles.reserve(chart.local_faces.size());
    for (const auto &lf : chart.local_faces) {
      const std::array<Vec2d, 3> tri{
          chart.vertex_uvs[static_cast<size_t>(lf[0])],
          chart.vertex_uvs[static_cast<size_t>(lf[1])],
          chart.vertex_uvs[static_cast<size_t>(lf[2])]};
      const double min_u = std::min({tri[0].u, tri[1].u, tri[2].u});
      const double max_u = std::max({tri[0].u, tri[1].u, tri[2].u});
      const double min_v = std::min({tri[0].v, tri[1].v, tri[2].v});
      const double max_v = std::max({tri[0].v, tri[1].v, tri[2].v});
      triangles.push_back(tri);
      boxes.push_back({min_u, max_u, min_v, max_v});
    }
    for (size_t i = 0; i < triangles.size(); ++i) {
      const double area_i = std::fabs(signed_area(chart, i));
      if (area_i <= kUvAreaEpsilon) {
        continue;
      }
      for (size_t j = i + 1; j < triangles.size(); ++j) {
        if (std::fabs(signed_area(chart, j)) <= kUvAreaEpsilon) {
          continue;
        }
        if (boxes[i][0] > boxes[j][1] || boxes[j][0] > boxes[i][1] ||
            boxes[i][2] > boxes[j][3] || boxes[j][2] > boxes[i][3]) {
          continue;
        }
        if (triangles_overlap_2d(triangles[i], triangles[j])) {
          return false;
        }
      }
    }
    return true;
  }

  void orientation_normalize(LocalChart &chart) const {
    double total = 0.0;
    for (size_t i = 0; i < chart.local_faces.size(); ++i) {
      total += signed_area(chart, i);
    }
    if (total < 0.0) {
      for (Vec2d &uv : chart.vertex_uvs) {
        uv.u = -uv.u;
      }
    }
  }

  struct StretchOut {
    double l2 = 0.0;
    double linf = 0.0;
  };

  StretchOut chart_stretch(const LocalChart &chart) const {
    double weighted_l2_sq = 0.0;
    double total_area3d = 0.0;
    double linf = 0.0;
    for (size_t i = 0; i < chart.local_faces.size(); ++i) {
      const double uv_area = signed_area(chart, i);
      const double area3d = face_areas_[static_cast<size_t>(chart.faces[i])];
      if (uv_area <= kUvAreaEpsilon || area3d <= 0.0) {
        continue;
      }
      const auto &lf = chart.local_faces[i];
      const Vec3 q0 = vertex_position(
          mesh_, chart.local_vertices[static_cast<size_t>(lf[0])]);
      const Vec3 q1 = vertex_position(
          mesh_, chart.local_vertices[static_cast<size_t>(lf[1])]);
      const Vec3 q2 = vertex_position(
          mesh_, chart.local_vertices[static_cast<size_t>(lf[2])]);
      const Vec2d &t0 = chart.vertex_uvs[static_cast<size_t>(lf[0])];
      const Vec2d &t1 = chart.vertex_uvs[static_cast<size_t>(lf[1])];
      const Vec2d &t2 = chart.vertex_uvs[static_cast<size_t>(lf[2])];
      const double inv2a = 1.0 / (2.0 * uv_area);
      Vec3 ss{0.0, 0.0, 0.0};
      Vec3 st{0.0, 0.0, 0.0};
      for (int axis = 0; axis < 3; ++axis) {
        ss[static_cast<size_t>(axis)] =
            (q0[static_cast<size_t>(axis)] * (t1.v - t2.v) +
             q1[static_cast<size_t>(axis)] * (t2.v - t0.v) +
             q2[static_cast<size_t>(axis)] * (t0.v - t1.v)) * inv2a;
        st[static_cast<size_t>(axis)] =
            (q0[static_cast<size_t>(axis)] * (t2.u - t1.u) +
             q1[static_cast<size_t>(axis)] * (t0.u - t2.u) +
             q2[static_cast<size_t>(axis)] * (t1.u - t0.u)) * inv2a;
      }
      const double a = dot(ss, ss);
      const double b = dot(ss, st);
      const double c = dot(st, st);
      const double disc = std::sqrt(std::max(0.0, (a - c) * (a - c) + 4.0 * b * b));
      weighted_l2_sq += ((a + c) * 0.5) * area3d;
      total_area3d += area3d;
      linf = std::max(linf, std::sqrt(std::max(0.0, ((a + c) + disc) * 0.5)));
    }
    StretchOut result;
    result.l2 = total_area3d > 0.0 ? std::sqrt(weighted_l2_sq / total_area3d) : 0.0;
    result.linf = linf;
    return result;
  }

  enum class ProjectionState { kInvalid, kValidButStretchy, kAccepted };

  ProjectionState try_projection(LocalChart &chart, double *linf_out = nullptr) const {
    std::vector<Vec3> points;
    points.reserve(chart.local_faces.size() * 3);
    for (const int64_t f : chart.faces) {
      const auto &fv = mesh_.faces[static_cast<size_t>(f)];
      points.push_back(vertex_position(mesh_, fv[0]));
      points.push_back(vertex_position(mesh_, fv[1]));
      points.push_back(vertex_position(mesh_, fv[2]));
    }
    Basis basis;
    if (!least_squares_basis(points, &basis)) {
      return ProjectionState::kInvalid;
    }
    for (size_t i = 0; i < chart.local_vertices.size(); ++i) {
      const Vec3 p = vertex_position(mesh_, chart.local_vertices[i]);
      chart.vertex_uvs[i] = Vec2d{dot(basis.tangent, p), dot(basis.bitangent, p)};
    }
    orientation_normalize(chart);
    if (!chart_uvs_valid(chart)) {
      return ProjectionState::kInvalid;
    }
    const double linf = chart_stretch(chart).linf;
    if (linf_out != nullptr) {
      *linf_out = linf;
    }
    return linf <= options_.projection_linf_threshold
        ? ProjectionState::kAccepted
        : ProjectionState::kValidButStretchy;
  }

  // Faces violating the zero-flip / zero-interior-overlap invariant in the
  // chart's current UVs (used for targeted splitting).
  std::set<size_t> offending_faces(const LocalChart &chart) const {
    std::set<size_t> offending;
    std::vector<std::array<Vec2d, 3>> triangles;
    triangles.reserve(chart.local_faces.size());
    for (const auto &lf : chart.local_faces) {
      triangles.push_back({chart.vertex_uvs[static_cast<size_t>(lf[0])],
                           chart.vertex_uvs[static_cast<size_t>(lf[1])],
                           chart.vertex_uvs[static_cast<size_t>(lf[2])]});
    }
    for (size_t i = 0; i < chart.local_faces.size(); ++i) {
      const double area = signed_area(chart, i);
      if (!std::isfinite(area) || area < -kUvAreaEpsilon) {
        offending.insert(i);
      }
    }
    for (size_t i = 0; i < triangles.size(); ++i) {
      if (std::fabs(signed_area(chart, i)) <= kUvAreaEpsilon) {
        continue;
      }
      for (size_t j = i + 1; j < triangles.size(); ++j) {
        if (std::fabs(signed_area(chart, j)) <= kUvAreaEpsilon) {
          continue;
        }
        if (offending.count(i) > 0 && offending.count(j) > 0) {
          continue;
        }
        if (triangles_overlap_2d(triangles[i], triangles[j])) {
          offending.insert(i);
          offending.insert(j);
        }
      }
    }
    return offending;
  }

  bool try_lscm(LocalChart &chart) {
    // Boundary vertices: on an edge with != 2 incident chart faces.
    std::map<mesh_common::EdgeKey, int64_t, EdgeKeyLess> edge_counts;
    for (const auto &lf : chart.local_faces) {
      for (int corner = 0; corner < 3; ++corner) {
        edge_counts[mesh_common::edge_key(
            lf[static_cast<size_t>(corner)], lf[static_cast<size_t>((corner + 1) % 3)])] += 1;
      }
    }
    std::vector<bool> boundary_vertex(chart.local_vertices.size(), false);
    bool any_boundary = false;
    for (const auto &[key, count] : edge_counts) {
      if (count != 2) {
        boundary_vertex[static_cast<size_t>(key.a)] = true;
        boundary_vertex[static_cast<size_t>(key.b)] = true;
        any_boundary = true;
      }
    }
    if (!any_boundary) {
      return false;  // closed chart: reference LSCM refuses too
    }
    // Pins: findApproximateDiameterVertices port (axis-extreme boundary pair,
    // including the reference's v=1 scan start).
    const int64_t vertex_count = static_cast<int64_t>(chart.local_vertices.size());
    std::array<int64_t, 3> min_vertex{-1, -1, -1};
    std::array<int64_t, 3> max_vertex{-1, -1, -1};
    for (int64_t v = 1; v < vertex_count; ++v) {
      if (boundary_vertex[static_cast<size_t>(v)]) {
        min_vertex = {v, v, v};
        max_vertex = {v, v, v};
        break;
      }
    }
    if (min_vertex[0] < 0) {
      return false;
    }
    const auto position = [&](int64_t local) {
      return vertex_position(mesh_, chart.local_vertices[static_cast<size_t>(local)]);
    };
    for (int64_t v = 1; v < vertex_count; ++v) {
      if (!boundary_vertex[static_cast<size_t>(v)]) {
        continue;
      }
      const Vec3 pos = position(v);
      for (int axis = 0; axis < 3; ++axis) {
        if (pos[static_cast<size_t>(axis)] <
            position(min_vertex[static_cast<size_t>(axis)])[static_cast<size_t>(axis)]) {
          min_vertex[static_cast<size_t>(axis)] = v;
        } else if (pos[static_cast<size_t>(axis)] >
                   position(max_vertex[static_cast<size_t>(axis)])[static_cast<size_t>(axis)]) {
          max_vertex[static_cast<size_t>(axis)] = v;
        }
      }
    }
    std::array<double, 3> lengths{};
    for (int axis = 0; axis < 3; ++axis) {
      lengths[static_cast<size_t>(axis)] = norm(vec_sub(
          position(min_vertex[static_cast<size_t>(axis)]),
          position(max_vertex[static_cast<size_t>(axis)])));
    }
    int64_t pin0 = 0;
    int64_t pin1 = 0;
    if (lengths[0] > lengths[1] && lengths[0] > lengths[2]) {
      pin0 = min_vertex[0];
      pin1 = max_vertex[0];
    } else if (lengths[1] > lengths[2]) {
      pin0 = min_vertex[1];
      pin1 = max_vertex[1];
    } else {
      pin0 = min_vertex[2];
      pin1 = max_vertex[2];
    }
    if (pin0 == pin1) {
      return false;
    }
    // Warm start from current (projection) UVs; lock the two pins.
    LscmSolver solver(2 * vertex_count);
    for (const auto &lf : chart.local_faces) {
      const std::array<int64_t, 3> ids{lf[0], lf[1], lf[2]};
      const std::array<Vec3, 3> p{position(lf[0]), position(lf[1]), position(lf[2])};
      if (!add_abf_rows(solver, ids, p)) {
        add_projected_rows(solver, ids, p);
      }
    }
    std::vector<double> x(static_cast<size_t>(2 * vertex_count), 0.0);
    std::vector<bool> locked(static_cast<size_t>(2 * vertex_count), false);
    for (int64_t v = 0; v < vertex_count; ++v) {
      x[static_cast<size_t>(2 * v)] = chart.vertex_uvs[static_cast<size_t>(v)].u;
      x[static_cast<size_t>(2 * v) + 1] = chart.vertex_uvs[static_cast<size_t>(v)].v;
    }
    locked[static_cast<size_t>(2 * pin0)] = true;
    locked[static_cast<size_t>(2 * pin0) + 1] = true;
    locked[static_cast<size_t>(2 * pin1)] = true;
    locked[static_cast<size_t>(2 * pin1) + 1] = true;
    if (!solver.solve(x, locked, options_.lscm_iteration_factor * vertex_count)) {
      lscm_unconverged_count += 1;
      return false;
    }
    for (int64_t v = 0; v < vertex_count; ++v) {
      chart.vertex_uvs[static_cast<size_t>(v)] =
          Vec2d{x[static_cast<size_t>(2 * v)], x[static_cast<size_t>(2 * v) + 1]};
    }
    orientation_normalize(chart);
    return chart_uvs_valid(chart);
  }

  void emit(const LocalChart &chart, int64_t method, std::vector<ParamChartOut> &out) {
    ParamChartOut emitted;
    emitted.faces = chart.faces;
    emitted.corner_uvs.reserve(chart.faces.size() * 3);
    for (const auto &lf : chart.local_faces) {
      for (int corner = 0; corner < 3; ++corner) {
        emitted.corner_uvs.push_back(
            chart.vertex_uvs[static_cast<size_t>(lf[static_cast<size_t>(corner)])]);
      }
    }
    emitted.method = method;
    const StretchOut stretch = chart_stretch(chart);
    emitted.stretch_l2 = stretch.l2;
    emitted.stretch_linf = stretch.linf;
    out.push_back(std::move(emitted));
  }

  void shatter(const std::vector<int64_t> &faces, std::vector<ParamChartOut> &out) {
    for (const int64_t f : faces) {
      LocalChart single = weld({f});
      const auto &fv = mesh_.faces[static_cast<size_t>(f)];
      const std::array<Vec3, 3> p{
          vertex_position(mesh_, fv[0]),
          vertex_position(mesh_, fv[1]),
          vertex_position(mesh_, fv[2])};
      const Vec3 x_axis = normalized_or_zero(vec_sub(p[1], p[0]));
      const Vec3 z_axis = normalized_or_zero(vec_cross(x_axis, vec_sub(p[2], p[0])));
      const Vec3 y_axis = vec_cross(z_axis, x_axis);
      single.vertex_uvs[static_cast<size_t>(single.local_faces[0][0])] = Vec2d{0.0, 0.0};
      single.vertex_uvs[static_cast<size_t>(single.local_faces[0][1])] =
          Vec2d{norm(vec_sub(p[1], p[0])), 0.0};
      single.vertex_uvs[static_cast<size_t>(single.local_faces[0][2])] =
          Vec2d{dot(vec_sub(p[2], p[0]), x_axis), dot(vec_sub(p[2], p[0]), y_axis)};
      orientation_normalize(single);
      shattered_face_chart_count += 1;
      emit(single, 2, out);
    }
  }

  void parameterize_recursive(
      const std::vector<int64_t> &faces, int64_t depth, std::vector<ParamChartOut> &out) {
    LocalChart chart = weld(faces);
    double projection_linf = std::numeric_limits<double>::infinity();
    const ProjectionState projection = try_projection(chart, &projection_linf);
    if (projection == ProjectionState::kAccepted) {
      projected_chart_count += 1;
      emit(chart, 0, out);
      return;
    }
    LocalChart lscm_chart = weld(faces);
    // Warm start: LSCM starts from the projection even when the projection
    // failed acceptance (reference warm-starts from ortho texcoords). The
    // weld order is identical, so the UVs carry over directly.
    lscm_chart.vertex_uvs = chart.vertex_uvs;
    const bool lscm_valid = try_lscm(lscm_chart);
    const double lscm_linf =
        lscm_valid ? chart_stretch(lscm_chart).linf : std::numeric_limits<double>::infinity();

    // Best valid candidate by Linf stretch. Free-boundary LSCM on tube-like
    // charts is valid but decays exponentially (fixture Linf reached 9000+),
    // so a valid projection with lower stretch must win, and an over-stretchy
    // best candidate splits rather than ships while budget remains.
    const bool projection_valid = projection == ProjectionState::kValidButStretchy;
    const bool lscm_is_best = lscm_valid && (!projection_valid || lscm_linf <= projection_linf);
    const double best_linf = lscm_is_best
        ? lscm_linf
        : (projection_valid ? projection_linf : std::numeric_limits<double>::infinity());
    // Accept the best valid candidate under the fallback cap (the reference
    // accepts LSCM unconditionally; the cap only excises the exponential-decay
    // tube pathology). Beyond the cap, split while budget remains.
    if (best_linf <= options_.fallback_linf_cap) {
      if (lscm_is_best) {
        lscm_chart_count += 1;
        emit(lscm_chart, 1, out);
      } else {
        projection_fallback_chart_count += 1;
        emit(chart, 0, out);
      }
      return;
    }
    const bool can_split = faces.size() > 1 && depth < options_.max_split_depth;
    if (!can_split) {
      shatter(faces, out);
      return;
    }
    // Targeted split when the LSCM result violates the invariant: peel the
    // offending faces into their own chart; the cleaned remainder usually
    // flattens. (For valid-but-stretchy candidates the offender set is empty
    // and the geometric bisection below applies.)
    if (!lscm_valid) {
      const std::set<size_t> offending = offending_faces(lscm_chart);
      if (!offending.empty() && offending.size() < faces.size()) {
        std::vector<int64_t> clean;
        std::vector<int64_t> dirty;
        for (size_t i = 0; i < faces.size(); ++i) {
          if (offending.count(i) > 0) {
            dirty.push_back(faces[i]);
          } else {
            clean.push_back(faces[i]);
          }
        }
        split_event_count += 1;
        parameterize_recursive(clean, depth + 1, out);
        parameterize_recursive(dirty, depth + 1, out);
        return;
      }
    }
    // Deterministic longest-axis bisection by face centroid; fall back to a
    // face-order halving when geometry collapses to a point.
    std::array<double, 3> bbox_min{
        std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()};
    std::array<double, 3> bbox_max{
        std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest()};
    std::vector<Vec3> centroids;
    centroids.reserve(faces.size());
    for (const int64_t f : faces) {
      const auto &fv = mesh_.faces[static_cast<size_t>(f)];
      const Vec3 p0 = vertex_position(mesh_, fv[0]);
      const Vec3 p1 = vertex_position(mesh_, fv[1]);
      const Vec3 p2 = vertex_position(mesh_, fv[2]);
      const Vec3 center{(p0[0] + p1[0] + p2[0]) / 3.0, (p0[1] + p1[1] + p2[1]) / 3.0,
                        (p0[2] + p1[2] + p2[2]) / 3.0};
      centroids.push_back(center);
      for (int axis = 0; axis < 3; ++axis) {
        bbox_min[static_cast<size_t>(axis)] =
            std::min(bbox_min[static_cast<size_t>(axis)], center[static_cast<size_t>(axis)]);
        bbox_max[static_cast<size_t>(axis)] =
            std::max(bbox_max[static_cast<size_t>(axis)], center[static_cast<size_t>(axis)]);
      }
    }
    int best_axis = 0;
    double best_extent = -1.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double extent =
          bbox_max[static_cast<size_t>(axis)] - bbox_min[static_cast<size_t>(axis)];
      if (extent > best_extent) {
        best_extent = extent;
        best_axis = axis;
      }
    }
    const double midpoint = 0.5 *
        (bbox_min[static_cast<size_t>(best_axis)] + bbox_max[static_cast<size_t>(best_axis)]);
    std::vector<int64_t> low;
    std::vector<int64_t> high;
    for (size_t i = 0; i < faces.size(); ++i) {
      if (centroids[i][static_cast<size_t>(best_axis)] <= midpoint) {
        low.push_back(faces[i]);
      } else {
        high.push_back(faces[i]);
      }
    }
    if (low.empty() || high.empty()) {
      low.assign(faces.begin(), faces.begin() + static_cast<std::ptrdiff_t>(faces.size() / 2));
      high.assign(faces.begin() + static_cast<std::ptrdiff_t>(faces.size() / 2), faces.end());
    }
    split_event_count += 1;
    parameterize_recursive(low, depth + 1, out);
    parameterize_recursive(high, depth + 1, out);
  }
};

// ---------------------------------------------------------------------------
// Stage B packing (slice 6): rotate charts to their principal axis, scale by
// a binary-searched texels-per-unit, shelf-pack with texel gaps, normalize to
// [0,1]. Reference: xatlas PackOptions defaults (padding 0, bilinear true,
// rotate_charts/rotate_charts_to_axis true, brute_force false). Documented
// deviations:
//  9. Rotate-to-axis uses the chart's UV PCA major axis rather than the
//     convex-hull edge scan; packing is sorted shelf placement over chart
//     rects rather than the reference's rasterized insertion — same
//     semantics (gap >= padding [+1 bilinear] texels, deterministic),
//     coarser utilization (rects, not rasterized outlines).
// ---------------------------------------------------------------------------

struct PackedChart {
  int64_t chart = 0;
  double width = 0.0;   // rotated UV units
  double height = 0.0;
  double min_u = 0.0;
  double min_v = 0.0;
  double cos_r = 1.0;
  double sin_r = 0.0;
  double x = 0.0;  // placement in texels
  double y = 0.0;
};

struct PackResult {
  std::vector<Vec2d> corner_uvs;  // normalized [0,1]
  std::vector<PackedChart> charts;
  double scale = 0.0;             // texels per UV unit
  double packed_height_texels = 0.0;
  int64_t shelf_count = 0;
};

// Shelf placement at a given scale; returns false when it exceeds the atlas.
bool shelf_place(
    std::vector<PackedChart> &charts, double scale, double resolution, double gap,
    double *height_out, int64_t *shelf_count_out) {
  double x = 0.0;
  double y = 0.0;
  double shelf_height = 0.0;
  int64_t shelves = 1;
  for (PackedChart &chart : charts) {
    const double w = chart.width * scale;
    const double h = chart.height * scale;
    if (w > resolution) {
      return false;
    }
    if (x > 0.0 && x + w > resolution) {
      y += shelf_height + gap;
      x = 0.0;
      shelf_height = 0.0;
      shelves += 1;
    }
    chart.x = x;
    chart.y = y;
    shelf_height = std::max(shelf_height, h);
    x += w + gap;
    if (y + shelf_height > resolution) {
      return false;
    }
  }
  *height_out = y + shelf_height;
  *shelf_count_out = shelves;
  return true;
}

PackResult pack_charts(
    const std::vector<int64_t> &chart_ids,
    const std::vector<Vec2d> &corner_uvs,
    int64_t chart_count,
    double resolution,
    double padding,
    bool bilinear,
    bool rotate_to_axis) {
  const size_t corner_count = corner_uvs.size();
  PackResult result;
  result.charts.resize(static_cast<size_t>(chart_count));
  for (int64_t c = 0; c < chart_count; ++c) {
    result.charts[static_cast<size_t>(c)].chart = c;
  }
  // Per-chart PCA rotation (deviation 9) and rotated bounds.
  std::vector<double> sum_u(static_cast<size_t>(chart_count), 0.0);
  std::vector<double> sum_v(static_cast<size_t>(chart_count), 0.0);
  std::vector<int64_t> counts(static_cast<size_t>(chart_count), 0);
  for (size_t corner = 0; corner < corner_count; ++corner) {
    const int64_t c = chart_ids[corner / 3];
    sum_u[static_cast<size_t>(c)] += corner_uvs[corner].u;
    sum_v[static_cast<size_t>(c)] += corner_uvs[corner].v;
    counts[static_cast<size_t>(c)] += 1;
  }
  if (rotate_to_axis) {
    std::vector<double> cov_uu(static_cast<size_t>(chart_count), 0.0);
    std::vector<double> cov_uv(static_cast<size_t>(chart_count), 0.0);
    std::vector<double> cov_vv(static_cast<size_t>(chart_count), 0.0);
    for (size_t corner = 0; corner < corner_count; ++corner) {
      const int64_t c = chart_ids[corner / 3];
      const double du = corner_uvs[corner].u - sum_u[static_cast<size_t>(c)] / counts[static_cast<size_t>(c)];
      const double dv = corner_uvs[corner].v - sum_v[static_cast<size_t>(c)] / counts[static_cast<size_t>(c)];
      cov_uu[static_cast<size_t>(c)] += du * du;
      cov_uv[static_cast<size_t>(c)] += du * dv;
      cov_vv[static_cast<size_t>(c)] += dv * dv;
    }
    for (int64_t c = 0; c < chart_count; ++c) {
      // Major eigenvector angle of the 2x2 covariance; rotate it onto +u.
      const double theta = 0.5 * std::atan2(
          2.0 * cov_uv[static_cast<size_t>(c)],
          cov_uu[static_cast<size_t>(c)] - cov_vv[static_cast<size_t>(c)]);
      result.charts[static_cast<size_t>(c)].cos_r = std::cos(-theta);
      result.charts[static_cast<size_t>(c)].sin_r = std::sin(-theta);
    }
  }
  std::vector<double> min_u(static_cast<size_t>(chart_count), std::numeric_limits<double>::max());
  std::vector<double> min_v(static_cast<size_t>(chart_count), std::numeric_limits<double>::max());
  std::vector<double> max_u(static_cast<size_t>(chart_count), std::numeric_limits<double>::lowest());
  std::vector<double> max_v(static_cast<size_t>(chart_count), std::numeric_limits<double>::lowest());
  const auto rotated = [&](size_t corner) {
    const int64_t c = chart_ids[corner / 3];
    const PackedChart &chart = result.charts[static_cast<size_t>(c)];
    const Vec2d &uv = corner_uvs[corner];
    return Vec2d{uv.u * chart.cos_r - uv.v * chart.sin_r,
                 uv.u * chart.sin_r + uv.v * chart.cos_r};
  };
  for (size_t corner = 0; corner < corner_count; ++corner) {
    const int64_t c = chart_ids[corner / 3];
    const Vec2d uv = rotated(corner);
    min_u[static_cast<size_t>(c)] = std::min(min_u[static_cast<size_t>(c)], uv.u);
    min_v[static_cast<size_t>(c)] = std::min(min_v[static_cast<size_t>(c)], uv.v);
    max_u[static_cast<size_t>(c)] = std::max(max_u[static_cast<size_t>(c)], uv.u);
    max_v[static_cast<size_t>(c)] = std::max(max_v[static_cast<size_t>(c)], uv.v);
  }
  for (int64_t c = 0; c < chart_count; ++c) {
    PackedChart &chart = result.charts[static_cast<size_t>(c)];
    if (counts[static_cast<size_t>(c)] == 0) {
      chart.width = chart.height = chart.min_u = chart.min_v = 0.0;
      continue;
    }
    chart.min_u = min_u[static_cast<size_t>(c)];
    chart.min_v = min_v[static_cast<size_t>(c)];
    chart.width = max_u[static_cast<size_t>(c)] - chart.min_u;
    chart.height = max_v[static_cast<size_t>(c)] - chart.min_v;
  }
  // Sort by scaled height desc (shelf heuristic), chart id ties — restored to
  // chart order for output via the chart field.
  std::vector<PackedChart> order = result.charts;
  std::sort(order.begin(), order.end(), [](const PackedChart &a, const PackedChart &b) {
    if (a.height != b.height) {
      return a.height > b.height;
    }
    return a.chart < b.chart;
  });
  const double gap = padding + (bilinear ? 1.0 : 0.0);
  // Binary-search the largest scale that fits (40 iterations, matching the
  // existing aspect-shelf packer's discipline).
  double total_area = 0.0;
  for (const PackedChart &chart : order) {
    total_area += chart.width * chart.height;
  }
  double height = 0.0;
  int64_t shelves = 0;
  const auto fits = [&](double scale) {
    double h = 0.0;
    int64_t s = 0;
    if (shelf_place(order, scale, resolution, gap, &h, &s)) {
      height = h;
      shelves = s;
      return true;
    }
    return false;
  };
  // Bracket: halve from an optimistic guess down to a feasible scale, then
  // bisect [feasible, first-infeasible].
  double hi = total_area > 0.0 ? resolution / std::sqrt(total_area) * 2.0 : 1.0;
  double lo = hi;
  for (int k = 0; k < 60 && !fits(lo); ++k) {
    lo *= 0.5;
  }
  hi = lo * 2.0;
  for (int iteration = 0; iteration < 40; ++iteration) {
    const double mid = 0.5 * (lo + hi);
    if (fits(mid)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  shelf_place(order, lo, resolution, gap, &height, &shelves);
  result.scale = lo;
  result.packed_height_texels = height;
  result.shelf_count = shelves;
  for (const PackedChart &placed : order) {
    result.charts[static_cast<size_t>(placed.chart)] = placed;
  }
  // Emit normalized UVs.
  result.corner_uvs.resize(corner_count);
  for (size_t corner = 0; corner < corner_count; ++corner) {
    const int64_t c = chart_ids[corner / 3];
    const PackedChart &chart = result.charts[static_cast<size_t>(c)];
    const Vec2d uv = rotated(corner);
    result.corner_uvs[corner] = Vec2d{
        (chart.x + (uv.u - chart.min_u) * lo) / resolution,
        (chart.y + (uv.v - chart.min_v) * lo) / resolution,
    };
  }
  return result;
}

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

nb::dict grow_uv_charts(
    nb::object vertices,
    nb::object faces,
    nb::object cluster_ids,
    double max_cost,
    double normal_deviation_weight,
    double roundness_weight,
    double straightness_weight,
    double normal_seam_weight,
    double texture_seam_weight,
    int64_t max_iterations,
    double projection_linf_threshold,
    double max_chart_area,
    double max_boundary_length) {
  const mesh_common::MeshData mesh = mesh_common::load_mesh(vertices, faces);
  const int64_t face_count = static_cast<int64_t>(mesh.faces.size());
  for (const double knob :
       {max_cost, normal_deviation_weight, roundness_weight, straightness_weight,
        normal_seam_weight, texture_seam_weight, projection_linf_threshold, max_chart_area,
        max_boundary_length}) {
    if (!std::isfinite(knob)) {
      throw nb::value_error("grow_uv_charts weights/limits must be finite");
    }
  }
  if (max_cost <= 0.0) {
    throw nb::value_error("max_cost must be positive");
  }
  if (max_iterations < 0) {
    throw nb::value_error("max_iterations must be non-negative");
  }
  if (projection_linf_threshold <= 0.0) {
    throw nb::value_error("projection_linf_threshold must be positive");
  }

  std::vector<int64_t> clusters(static_cast<size_t>(face_count), 0);
  if (!cluster_ids.is_none()) {
    const auto ndim = nb::cast<int64_t>(nb::getattr(cluster_ids, "ndim"));
    if (ndim != 1) {
      throw nb::value_error("cluster_ids must have rank 1");
    }
    if (mesh_common::dtype_name(cluster_ids, "cluster_ids") != "int64") {
      throw nb::value_error("cluster_ids must have dtype int64");
    }
    if (mesh_common::dimension(cluster_ids, "cluster_ids", 0) != face_count) {
      std::ostringstream message;
      message << "cluster_ids must have shape (" << face_count << ",)";
      throw nb::value_error(message.str().c_str());
    }
    mesh_common::BufferView cluster_buffer(cluster_ids.ptr(), "cluster_ids");
    const Py_buffer &view = cluster_buffer.get();
    const auto *base = static_cast<const char *>(view.buf);
    const Py_ssize_t stride = view.strides != nullptr ? view.strides[0] : view.itemsize;
    for (int64_t row = 0; row < face_count; ++row) {
      int64_t value = 0;
      std::memcpy(&value, base + row * stride, sizeof(int64_t));
      if (value < 0) {
        throw nb::value_error("cluster_ids must be non-negative");
      }
      clusters[static_cast<size_t>(row)] = value;
    }
  }

  uv_unwrap::GrowthOptions options;
  options.max_cost = max_cost;
  options.normal_deviation_weight = normal_deviation_weight;
  options.roundness_weight = roundness_weight;
  options.straightness_weight = straightness_weight;
  options.normal_seam_weight = normal_seam_weight;
  options.texture_seam_weight = texture_seam_weight;
  options.max_iterations = max_iterations;
  options.projection_linf_threshold = projection_linf_threshold;
  options.max_chart_area = max_chart_area;
  options.max_boundary_length = max_boundary_length;

  uv_unwrap::ChartBuildResult built =
      uv_unwrap::ChartBuilder(mesh, clusters, options).run();

  nb::dict result;
  result["chart_ids"] = uv_unwrap::make_int64_vector(std::move(built.chart_ids));
  result["chart_count"] = built.chart_count;
  result["corner_uvs"] = uv_unwrap::make_float64_matrix(
      std::move(built.corner_uvs), static_cast<size_t>(face_count) * 3, 2);
  result["chart_face_counts"] = uv_unwrap::make_int64_vector(std::move(built.chart_face_counts));
  result["chart_stretch_l2"] = uv_unwrap::make_float64_vector(std::move(built.chart_stretch_l2));
  result["chart_stretch_linf"] =
      uv_unwrap::make_float64_vector(std::move(built.chart_stretch_linf));
  result["chart_accepted"] = uv_unwrap::make_int64_vector(std::move(built.chart_accepted));
  result["chart_needs_lscm"] = uv_unwrap::make_int64_vector(std::move(built.chart_needs_lscm));
  result["accepted_chart_count"] = built.accepted_chart_count;
  result["lscm_pending_chart_count"] = built.lscm_pending_chart_count;
  result["planar_region_count"] = built.planar_region_count;
  result["place_seed_chart_count"] = built.place_seed_chart_count;
  result["fill_hole_chart_count"] = built.fill_hole_chart_count;
  result["growth_merge_count"] = built.growth_merge_count;
  result["seed_relocation_count"] = built.seed_relocation_count;
  result["failed_add_count"] = built.failed_add_count;
  result["mirrored_chart_normalized_count"] = built.mirrored_chart_normalized_count;
  // Deviation 1: seam weights are accepted for contract completeness but
  // contribute exactly 0 on welded indexed input (no seams exist).
  result["normal_seam_weight"] = normal_seam_weight;
  result["texture_seam_weight"] = texture_seam_weight;
  return result;
}

nb::dict parameterize_uv_charts(
    nb::object vertices,
    nb::object faces,
    nb::object chart_ids,
    double projection_linf_threshold,
    int64_t max_split_depth,
    int64_t lscm_iteration_factor) {
  const mesh_common::MeshData mesh = mesh_common::load_mesh(vertices, faces);
  const int64_t face_count = static_cast<int64_t>(mesh.faces.size());
  if (!std::isfinite(projection_linf_threshold) || projection_linf_threshold <= 0.0) {
    throw nb::value_error("projection_linf_threshold must be finite and positive");
  }
  if (max_split_depth < 0) {
    throw nb::value_error("max_split_depth must be non-negative");
  }
  if (lscm_iteration_factor <= 0) {
    throw nb::value_error("lscm_iteration_factor must be positive");
  }
  if (chart_ids.is_none()) {
    throw nb::value_error("chart_ids is required (stage-B output)");
  }
  const auto ndim = nb::cast<int64_t>(nb::getattr(chart_ids, "ndim"));
  if (ndim != 1) {
    throw nb::value_error("chart_ids must have rank 1");
  }
  if (mesh_common::dtype_name(chart_ids, "chart_ids") != "int64") {
    throw nb::value_error("chart_ids must have dtype int64");
  }
  if (mesh_common::dimension(chart_ids, "chart_ids", 0) != face_count) {
    std::ostringstream message;
    message << "chart_ids must have shape (" << face_count << ",)";
    throw nb::value_error(message.str().c_str());
  }
  std::vector<int64_t> input_ids(static_cast<size_t>(face_count), 0);
  int64_t input_chart_count = 0;
  {
    mesh_common::BufferView chart_buffer(chart_ids.ptr(), "chart_ids");
    const Py_buffer &view = chart_buffer.get();
    const auto *base = static_cast<const char *>(view.buf);
    const Py_ssize_t stride = view.strides != nullptr ? view.strides[0] : view.itemsize;
    for (int64_t row = 0; row < face_count; ++row) {
      int64_t value = 0;
      std::memcpy(&value, base + row * stride, sizeof(int64_t));
      if (value < 0) {
        throw nb::value_error("chart_ids must be non-negative (dense stage-B ids)");
      }
      input_ids[static_cast<size_t>(row)] = value;
      input_chart_count = std::max(input_chart_count, value + 1);
    }
  }
  std::vector<std::vector<int64_t>> chart_faces(static_cast<size_t>(input_chart_count));
  for (int64_t f = 0; f < face_count; ++f) {
    chart_faces[static_cast<size_t>(input_ids[static_cast<size_t>(f)])].push_back(f);
  }

  uv_unwrap::ParamOptions options;
  options.projection_linf_threshold = projection_linf_threshold;
  options.max_split_depth = max_split_depth;
  options.lscm_iteration_factor = lscm_iteration_factor;
  uv_unwrap::ChartParameterizer parameterizer(mesh, options);
  std::vector<uv_unwrap::ParamChartOut> outs;
  for (const auto &faces_of_chart : chart_faces) {
    if (!faces_of_chart.empty()) {
      parameterizer.parameterize(faces_of_chart, outs);
    }
  }

  std::vector<int64_t> out_ids(static_cast<size_t>(face_count), -1);
  std::vector<double> corner_uvs(static_cast<size_t>(face_count) * 6, 0.0);
  std::vector<int64_t> face_counts;
  std::vector<double> stretch_l2;
  std::vector<double> stretch_linf;
  std::vector<int64_t> methods;
  face_counts.reserve(outs.size());
  for (size_t chart_index = 0; chart_index < outs.size(); ++chart_index) {
    const uv_unwrap::ParamChartOut &chart = outs[chart_index];
    for (size_t i = 0; i < chart.faces.size(); ++i) {
      const int64_t f = chart.faces[i];
      out_ids[static_cast<size_t>(f)] = static_cast<int64_t>(chart_index);
      for (int corner = 0; corner < 3; ++corner) {
        const uv_unwrap::Vec2d &uv = chart.corner_uvs[i * 3 + static_cast<size_t>(corner)];
        corner_uvs[(static_cast<size_t>(f) * 3 + static_cast<size_t>(corner)) * 2] = uv.u;
        corner_uvs[(static_cast<size_t>(f) * 3 + static_cast<size_t>(corner)) * 2 + 1] = uv.v;
      }
    }
    face_counts.push_back(static_cast<int64_t>(chart.faces.size()));
    stretch_l2.push_back(chart.stretch_l2);
    stretch_linf.push_back(chart.stretch_linf);
    methods.push_back(chart.method);
  }
  // Invariant recount (do not trust construction): flipped corners triples.
  int64_t flipped = 0;
  for (int64_t f = 0; f < face_count; ++f) {
    const double au = corner_uvs[(static_cast<size_t>(f) * 3) * 2];
    const double av = corner_uvs[(static_cast<size_t>(f) * 3) * 2 + 1];
    const double bu = corner_uvs[(static_cast<size_t>(f) * 3 + 1) * 2];
    const double bv = corner_uvs[(static_cast<size_t>(f) * 3 + 1) * 2 + 1];
    const double cu = corner_uvs[(static_cast<size_t>(f) * 3 + 2) * 2];
    const double cv = corner_uvs[(static_cast<size_t>(f) * 3 + 2) * 2 + 1];
    if (0.5 * ((bu - au) * (cv - av) - (cu - au) * (bv - av)) < -1e-12) {
      flipped += 1;
    }
  }

  nb::dict result;
  result["chart_ids"] = uv_unwrap::make_int64_vector(std::move(out_ids));
  result["chart_count"] = static_cast<int64_t>(outs.size());
  result["input_chart_count"] = input_chart_count;
  result["corner_uvs"] = uv_unwrap::make_float64_matrix(
      std::move(corner_uvs), static_cast<size_t>(face_count) * 3, 2);
  result["chart_face_counts"] = uv_unwrap::make_int64_vector(std::move(face_counts));
  result["chart_stretch_l2"] = uv_unwrap::make_float64_vector(std::move(stretch_l2));
  result["chart_stretch_linf"] = uv_unwrap::make_float64_vector(std::move(stretch_linf));
  result["chart_method"] = uv_unwrap::make_int64_vector(std::move(methods));
  result["projected_chart_count"] = parameterizer.projected_chart_count;
  result["projection_fallback_chart_count"] = parameterizer.projection_fallback_chart_count;
  result["lscm_chart_count"] = parameterizer.lscm_chart_count;
  result["shattered_face_chart_count"] = parameterizer.shattered_face_chart_count;
  result["split_event_count"] = parameterizer.split_event_count;
  result["lscm_unconverged_count"] = parameterizer.lscm_unconverged_count;
  result["uv_flipped_count"] = flipped;
  return result;
}

nb::dict pack_uv_charts(
    nb::object faces,
    nb::object chart_ids,
    nb::object corner_uvs,
    int64_t resolution,
    double padding,
    bool bilinear,
    bool rotate_charts_to_axis) {
  const int64_t face_count = mesh_common::dimension(faces, "faces", 0);
  if (resolution <= 0) {
    throw nb::value_error("resolution must be positive");
  }
  if (!std::isfinite(padding) || padding < 0.0) {
    throw nb::value_error("padding must be finite and non-negative");
  }
  // chart_ids: [F] int64 dense.
  std::vector<int64_t> ids(static_cast<size_t>(face_count), 0);
  int64_t chart_count = 0;
  {
    if (nb::cast<int64_t>(nb::getattr(chart_ids, "ndim")) != 1 ||
        mesh_common::dtype_name(chart_ids, "chart_ids") != "int64" ||
        mesh_common::dimension(chart_ids, "chart_ids", 0) != face_count) {
      throw nb::value_error("chart_ids must be int64 with shape (F,)");
    }
    mesh_common::BufferView view_holder(chart_ids.ptr(), "chart_ids");
    const Py_buffer &view = view_holder.get();
    const auto *base = static_cast<const char *>(view.buf);
    const Py_ssize_t stride = view.strides != nullptr ? view.strides[0] : view.itemsize;
    for (int64_t row = 0; row < face_count; ++row) {
      int64_t value = 0;
      std::memcpy(&value, base + row * stride, sizeof(int64_t));
      if (value < 0) {
        throw nb::value_error("chart_ids must be non-negative");
      }
      ids[static_cast<size_t>(row)] = value;
      chart_count = std::max(chart_count, value + 1);
    }
  }
  // corner_uvs: [F*3, 2] float64.
  std::vector<uv_unwrap::Vec2d> uvs(static_cast<size_t>(face_count) * 3);
  {
    mesh_common::validate_matrix(corner_uvs, "corner_uvs", 2, "float64");
    if (mesh_common::dimension(corner_uvs, "corner_uvs", 0) != face_count * 3) {
      throw nb::value_error("corner_uvs must have shape (F*3, 2)");
    }
    mesh_common::BufferView view_holder(corner_uvs.ptr(), "corner_uvs");
    const Py_buffer &view = view_holder.get();
    for (int64_t row = 0; row < face_count * 3; ++row) {
      uvs[static_cast<size_t>(row)] = uv_unwrap::Vec2d{
          mesh_common::read_matrix_value<double>(view, row, 0),
          mesh_common::read_matrix_value<double>(view, row, 1),
      };
      if (!std::isfinite(uvs[static_cast<size_t>(row)].u) ||
          !std::isfinite(uvs[static_cast<size_t>(row)].v)) {
        throw nb::value_error("corner_uvs must contain only finite values");
      }
    }
  }

  const uv_unwrap::PackResult packed = uv_unwrap::pack_charts(
      ids, uvs, chart_count, static_cast<double>(resolution), padding, bilinear,
      rotate_charts_to_axis);

  std::vector<double> out_uvs;
  out_uvs.reserve(packed.corner_uvs.size() * 2);
  for (const uv_unwrap::Vec2d &uv : packed.corner_uvs) {
    out_uvs.push_back(uv.u);
    out_uvs.push_back(uv.v);
  }
  std::vector<double> rects;
  rects.reserve(static_cast<size_t>(chart_count) * 4);
  for (const uv_unwrap::PackedChart &chart : packed.charts) {
    rects.push_back(chart.x);
    rects.push_back(chart.y);
    rects.push_back(chart.width * packed.scale);
    rects.push_back(chart.height * packed.scale);
  }
  nb::dict result;
  result["corner_uvs"] = uv_unwrap::make_float64_matrix(
      std::move(out_uvs), static_cast<size_t>(face_count) * 3, 2);
  result["chart_rects_texels"] = uv_unwrap::make_float64_matrix(
      std::move(rects), static_cast<size_t>(chart_count), 4);
  result["atlas_resolution"] = resolution;
  result["texels_per_unit"] = packed.scale;
  result["packed_height_texels"] = packed.packed_height_texels;
  result["shelf_count"] = packed.shelf_count;
  result["gap_texels"] = padding + (bilinear ? 1.0 : 0.0);
  result["rotate_charts_to_axis"] = rotate_charts_to_axis;
  return result;
}

}  // namespace mlx_spatialkit
