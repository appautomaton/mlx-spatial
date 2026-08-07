#include "mesh_processing.hpp"

#include <algorithm>
#include <atomic>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mesh_common.hpp"
#include "parallel.hpp"

namespace nb = nanobind;

namespace mlx_spatialkit {
namespace {

constexpr int64_t kSmallBoundaryLoopThresholdEdges = 32;

struct BoundaryTopology {
  int64_t boundary_vertices = 0;
  int64_t loop_count = 0;
  int64_t open_chain_count = 0;
  int64_t small_loop_count = 0;
  int64_t small_loop_edge_count = 0;
  int64_t open_chain_edge_count = 0;
  int64_t small_open_chain_count = 0;
  int64_t small_open_chain_edge_count = 0;
  int64_t simple_open_chain_count = 0;
  int64_t branched_open_chain_count = 0;
  int64_t open_chain_endpoint_count = 0;
  int64_t open_chain_branch_vertex_count = 0;
  int64_t max_loop_edges = 0;
  int64_t max_open_chain_edges = 0;
  int64_t max_component_edges = 0;
};

struct FaceKeyHash {
  size_t operator()(const std::array<int64_t, 3> &face) const {
    size_t seed = static_cast<size_t>(face[0]) * 73856093u;
    seed ^= static_cast<size_t>(face[1]) * 19349663u;
    seed ^= static_cast<size_t>(face[2]) * 83492791u;
    return seed;
  }
};

struct TopologyShard {
  std::mutex mutex;
  std::unordered_set<std::array<int64_t, 3>, FaceKeyHash> faces;
  std::unordered_map<mesh_common::EdgeKey, int64_t, mesh_common::EdgeKeyHash> edges;
};

struct ComponentMeasure {
  int64_t face_count = 0;
  double surface_area = 0.0;
};

double triangle_surface_area(
    const mesh_common::MeshData &mesh,
    const std::array<int64_t, 3> &face) {
  const auto &a = mesh.vertices[static_cast<size_t>(face[0])];
  const auto &b = mesh.vertices[static_cast<size_t>(face[1])];
  const auto &c = mesh.vertices[static_cast<size_t>(face[2])];
  const double abx = static_cast<double>(b[0]) - a[0];
  const double aby = static_cast<double>(b[1]) - a[1];
  const double abz = static_cast<double>(b[2]) - a[2];
  const double acx = static_cast<double>(c[0]) - a[0];
  const double acy = static_cast<double>(c[1]) - a[1];
  const double acz = static_cast<double>(c[2]) - a[2];
  const double cx = aby * acz - abz * acy;
  const double cy = abz * acx - abx * acz;
  const double cz = abx * acy - aby * acx;
  const double area = 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
  return std::isfinite(area) ? area : 0.0;
}

class ConcurrentUnionFind {
 public:
  explicit ConcurrentUnionFind(size_t count) : parent_(std::make_unique<std::atomic<size_t>[]>(count)) {
    parallel_for(count, [&](size_t begin, size_t end) {
      for (size_t index = begin; index < end; ++index) {
        parent_[index].store(index, std::memory_order_relaxed);
      }
    });
  }

  size_t find(size_t value) const {
    size_t parent = parent_[value].load(std::memory_order_relaxed);
    while (parent != value) {
      value = parent;
      parent = parent_[value].load(std::memory_order_relaxed);
    }
    return value;
  }

  void unite(size_t left, size_t right) {
    while (true) {
      left = find(left);
      right = find(right);
      if (left == right) {
        return;
      }
      const size_t lower = std::min(left, right);
      const size_t higher = std::max(left, right);
      size_t expected = higher;
      if (parent_[higher].compare_exchange_weak(
              expected,
              lower,
              std::memory_order_relaxed,
              std::memory_order_relaxed)) {
        return;
      }
    }
  }

 private:
  std::unique_ptr<std::atomic<size_t>[]> parent_;
};

BoundaryTopology boundary_topology(const std::vector<mesh_common::EdgeKey> &boundary_edges) {
  std::unordered_map<int64_t, std::vector<int64_t>> adjacency;
  adjacency.reserve(boundary_edges.size() * 2);
  for (const auto &edge : boundary_edges) {
    adjacency[edge.a].push_back(edge.b);
    adjacency[edge.b].push_back(edge.a);
  }

  BoundaryTopology topology;
  topology.boundary_vertices = static_cast<int64_t>(adjacency.size());
  if (adjacency.empty()) {
    return topology;
  }

  std::unordered_set<int64_t> visited;
  visited.reserve(adjacency.size());
  std::vector<int64_t> stack;
  stack.reserve(adjacency.size());
  for (const auto &[seed, _] : adjacency) {
    (void)_;
    if (visited.contains(seed)) {
      continue;
    }

    stack.clear();
    stack.push_back(seed);
    visited.insert(seed);
    int64_t component_vertices = 0;
    int64_t edge_stubs = 0;
    int64_t endpoint_vertices = 0;
    int64_t branch_vertices = 0;
    bool every_vertex_degree_two = true;
    while (!stack.empty()) {
      const int64_t vertex = stack.back();
      stack.pop_back();
      component_vertices += 1;
      const auto found = adjacency.find(vertex);
      if (found == adjacency.end()) {
        every_vertex_degree_two = false;
        continue;
      }
      const auto &neighbors = found->second;
      edge_stubs += static_cast<int64_t>(neighbors.size());
      if (neighbors.size() == 1) {
        endpoint_vertices += 1;
      } else if (neighbors.size() > 2) {
        branch_vertices += 1;
      }
      if (neighbors.size() != 2) {
        every_vertex_degree_two = false;
      }
      for (int64_t neighbor : neighbors) {
        if (!visited.contains(neighbor)) {
          visited.insert(neighbor);
          stack.push_back(neighbor);
        }
      }
    }

    const int64_t component_edges = edge_stubs / 2;
    topology.max_component_edges = std::max(topology.max_component_edges, component_edges);
    const bool closed_loop = every_vertex_degree_two && component_vertices >= 3 && component_edges >= 3;
    if (closed_loop) {
      topology.loop_count += 1;
      topology.max_loop_edges = std::max(topology.max_loop_edges, component_edges);
      if (component_edges <= kSmallBoundaryLoopThresholdEdges) {
        topology.small_loop_count += 1;
        topology.small_loop_edge_count += component_edges;
      }
    } else {
      topology.open_chain_count += 1;
      topology.open_chain_edge_count += component_edges;
      topology.max_open_chain_edges = std::max(topology.max_open_chain_edges, component_edges);
      topology.open_chain_endpoint_count += endpoint_vertices;
      topology.open_chain_branch_vertex_count += branch_vertices;
      if (endpoint_vertices == 2 && branch_vertices == 0) {
        topology.simple_open_chain_count += 1;
      } else {
        topology.branched_open_chain_count += 1;
      }
      if (component_edges <= kSmallBoundaryLoopThresholdEdges) {
        topology.small_open_chain_count += 1;
        topology.small_open_chain_edge_count += component_edges;
      }
    }
  }
  return topology;
}

}  // namespace

nb::dict mesh_metrics(nb::object vertices, nb::object faces) {
  mesh_common::MeshData mesh = mesh_common::load_mesh(vertices, faces);
  int64_t degenerate_faces = 0;
  int64_t duplicate_faces = 0;
  int64_t boundary_edges = 0;
  int64_t nonmanifold_edges = 0;
  int64_t nonmanifold_vertices = 0;
  int64_t connected_components = 0;
  std::vector<int64_t> component_face_counts;
  std::vector<ComponentMeasure> components_by_face_count;
  std::vector<double> component_surface_areas;
  double total_surface_area = 0.0;
  BoundaryTopology boundary;

  {
    nb::gil_scoped_release release;
    const size_t face_count = mesh.faces.size();
    const size_t vertex_count = mesh.vertices.size();
    const size_t worker_count = native_worker_count(face_count);
    const size_t shard_count = std::max<size_t>(1, worker_count * 4);
    std::vector<std::unique_ptr<TopologyShard>> shards;
    shards.reserve(shard_count);
    const size_t faces_per_shard = (face_count + shard_count - 1) / shard_count;
    const size_t edges_per_shard = (face_count * 3 + shard_count - 1) / shard_count;
    for (size_t shard = 0; shard < shard_count; ++shard) {
      auto topology_shard = std::make_unique<TopologyShard>();
      topology_shard->faces.reserve(faces_per_shard);
      topology_shard->edges.reserve(edges_per_shard);
      shards.push_back(std::move(topology_shard));
    }

    std::atomic<int64_t> degenerate_count{0};
    std::atomic<int64_t> duplicate_count{0};
    constexpr size_t kMetricsChunkSize = 8192;
    parallel_for(
        face_count,
        [&](size_t begin, size_t end) {
          std::vector<std::vector<std::array<int64_t, 3>>> local_faces(shard_count);
          std::vector<std::vector<mesh_common::EdgeKey>> local_edges(shard_count);
          int64_t local_degenerate_count = 0;
          for (size_t row = begin; row < end; ++row) {
            const auto &face = mesh.faces[row];
            if (mesh_common::face_degenerate(mesh, face)) {
              local_degenerate_count += 1;
            }
            const auto canonical = mesh_common::canonical_face(face);
            const size_t face_shard = FaceKeyHash{}(canonical) % shard_count;
            local_faces[face_shard].push_back(canonical);

            const std::array<mesh_common::EdgeKey, 3> edges = {
                mesh_common::edge_key(face[0], face[1]),
                mesh_common::edge_key(face[1], face[2]),
                mesh_common::edge_key(face[2], face[0]),
            };
            for (const auto &edge : edges) {
              const size_t edge_shard = mesh_common::EdgeKeyHash{}(edge) % shard_count;
              local_edges[edge_shard].push_back(edge);
            }
          }
          degenerate_count.fetch_add(local_degenerate_count, std::memory_order_relaxed);

          int64_t local_duplicate_count = 0;
          for (size_t shard = 0; shard < shard_count; ++shard) {
            if (local_faces[shard].empty() && local_edges[shard].empty()) {
              continue;
            }
            TopologyShard &topology_shard = *shards[shard];
            std::lock_guard<std::mutex> lock(topology_shard.mutex);
            for (const auto &face : local_faces[shard]) {
              if (!topology_shard.faces.insert(face).second) {
                local_duplicate_count += 1;
              }
            }
            for (const auto &edge : local_edges[shard]) {
              topology_shard.edges[edge] += 1;
            }
          }
          duplicate_count.fetch_add(local_duplicate_count, std::memory_order_relaxed);
        },
        2048,
        kMetricsChunkSize);
    degenerate_faces = degenerate_count.load(std::memory_order_relaxed);
    duplicate_faces = duplicate_count.load(std::memory_order_relaxed);

    std::vector<std::vector<mesh_common::EdgeKey>> boundary_edges_by_shard(shard_count);
    std::vector<int64_t> nonmanifold_edges_by_shard(shard_count, 0);
    parallel_for(
        shard_count,
        [&](size_t begin, size_t end) {
          for (size_t shard = begin; shard < end; ++shard) {
            for (const auto &[edge, count] : shards[shard]->edges) {
              if (count == 1) {
                boundary_edges_by_shard[shard].push_back(edge);
              } else if (count > 2) {
                nonmanifold_edges_by_shard[shard] += 1;
              }
            }
          }
        },
        1,
        1);

    std::vector<mesh_common::EdgeKey> mesh_boundary_edges;
    for (size_t shard = 0; shard < shard_count; ++shard) {
      boundary_edges += static_cast<int64_t>(boundary_edges_by_shard[shard].size());
      nonmanifold_edges += nonmanifold_edges_by_shard[shard];
    }
    mesh_boundary_edges.reserve(static_cast<size_t>(boundary_edges));
    for (auto &shard_edges : boundary_edges_by_shard) {
      mesh_boundary_edges.insert(
          mesh_boundary_edges.end(),
          std::make_move_iterator(shard_edges.begin()),
          std::make_move_iterator(shard_edges.end()));
    }
    shards.clear();
    boundary = boundary_topology(mesh_boundary_edges);

    // Count non-manifold vertices (pinches): vertices whose incident triangles
    // form more than one edge-connected fan.  Two incident triangles are in the
    // same fan iff they share an edge that is itself incident to the vertex.
    std::vector<uint32_t> vertex_degrees(vertex_count, 0);
    ConcurrentUnionFind component_union(vertex_count);
    parallel_for(face_count, [&](size_t begin, size_t end) {
      for (size_t face_index = begin; face_index < end; ++face_index) {
        const auto &face = mesh.faces[face_index];
        for (int corner = 0; corner < 3; ++corner) {
          std::atomic_ref<uint32_t> degree(vertex_degrees[static_cast<size_t>(face[corner])]);
          degree.fetch_add(1, std::memory_order_relaxed);
        }
        component_union.unite(static_cast<size_t>(face[0]), static_cast<size_t>(face[1]));
        component_union.unite(static_cast<size_t>(face[0]), static_cast<size_t>(face[2]));
      }
    });

    // Aggregate component face counts and 3D area through deterministic static
    // partitions. A single atomic area accumulator for the dominant component
    // would serialize almost every face on real meshes; per-worker sparse maps
    // avoid that hot spot and merge in a stable order.
    const size_t component_worker_count = native_worker_count(face_count);
    std::vector<std::unordered_map<size_t, ComponentMeasure>> component_partials(component_worker_count);
    std::vector<std::thread> component_workers;
    component_workers.reserve(component_worker_count);
    for (size_t worker = 0; worker < component_worker_count; ++worker) {
      component_workers.emplace_back([&, worker]() {
        const size_t begin = face_count * worker / component_worker_count;
        const size_t end = face_count * (worker + 1) / component_worker_count;
        auto &partial = component_partials[worker];
        partial.reserve(64);
        for (size_t face_index = begin; face_index < end; ++face_index) {
          const auto &face = mesh.faces[face_index];
          const size_t root = component_union.find(static_cast<size_t>(face[0]));
          ComponentMeasure &measure = partial[root];
          measure.face_count += 1;
          measure.surface_area += triangle_surface_area(mesh, face);
        }
      });
    }
    for (auto &worker : component_workers) {
      worker.join();
    }

    std::unordered_map<size_t, ComponentMeasure> components_by_root;
    for (const auto &partial : component_partials) {
      for (const auto &[root, measure] : partial) {
        ComponentMeasure &combined = components_by_root[root];
        combined.face_count += measure.face_count;
        combined.surface_area += measure.surface_area;
      }
    }
    components_by_face_count.reserve(components_by_root.size());
    component_face_counts.reserve(components_by_root.size());
    component_surface_areas.reserve(components_by_root.size());
    for (const auto &[_, measure] : components_by_root) {
      (void)_;
      components_by_face_count.push_back(measure);
      component_face_counts.push_back(measure.face_count);
      component_surface_areas.push_back(measure.surface_area);
      total_surface_area += measure.surface_area;
    }
    std::sort(
        components_by_face_count.begin(),
        components_by_face_count.end(),
        [](const ComponentMeasure &left, const ComponentMeasure &right) {
          if (left.face_count != right.face_count) {
            return left.face_count < right.face_count;
          }
          return left.surface_area < right.surface_area;
        });
    std::sort(component_face_counts.begin(), component_face_counts.end());
    std::sort(component_surface_areas.begin(), component_surface_areas.end());

    std::vector<size_t> incident_offsets(vertex_count + 1, 0);
    for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
      incident_offsets[vertex + 1] = incident_offsets[vertex] + vertex_degrees[vertex];
    }
    std::vector<int64_t> incident_faces(incident_offsets.back());
    std::vector<size_t> incident_cursors(incident_offsets.begin(), incident_offsets.end() - 1);
    parallel_for(face_count, [&](size_t begin, size_t end) {
      for (size_t face_index = begin; face_index < end; ++face_index) {
        const auto &face = mesh.faces[face_index];
        for (int corner = 0; corner < 3; ++corner) {
          const size_t vertex = static_cast<size_t>(face[corner]);
          std::atomic_ref<size_t> cursor(incident_cursors[vertex]);
          const size_t position = cursor.fetch_add(1, std::memory_order_relaxed);
          incident_faces[position] = static_cast<int64_t>(face_index);
        }
      }
    });
    std::vector<size_t>().swap(incident_cursors);

    std::atomic<int64_t> nonmanifold_vertex_count{0};
    std::atomic<int64_t> connected_component_count{0};
    parallel_for(vertex_count, [&](size_t begin, size_t end) {
      int64_t local_nonmanifold_vertices = 0;
      int64_t local_connected_components = 0;
      for (size_t vertex = begin; vertex < end; ++vertex) {
        const size_t degree = static_cast<size_t>(vertex_degrees[vertex]);
        if (degree > 0 && component_union.find(vertex) == vertex) {
          local_connected_components += 1;
        }
        if (degree < 2) {
          // 0 or 1 incident faces: always a single fan, not a pinch.
          continue;
        }
        // Map local position -> face index so we can union by local index.
        mesh_common::UnionFind uf(degree);
        // For each pair of local faces, check if they share an edge through vi.
        // Two faces share an edge through vi iff they both contain vi and share
        // exactly one other vertex (the neighbour across that edge).
        const size_t incident_begin = incident_offsets[vertex];
        for (size_t a = 0; a < degree; ++a) {
          const auto &fa = mesh.faces[static_cast<size_t>(incident_faces[incident_begin + a])];
          for (size_t b = a + 1; b < degree; ++b) {
            const auto &fb = mesh.faces[static_cast<size_t>(incident_faces[incident_begin + b])];
            // Count shared vertices between fa and fb that are NOT vi.
            int shared_non_vi = 0;
            for (int ca = 0; ca < 3; ++ca) {
              if (fa[ca] == static_cast<int64_t>(vertex)) { continue; }
              for (int cb = 0; cb < 3; ++cb) {
                if (fb[cb] == static_cast<int64_t>(vertex)) { continue; }
                if (fa[ca] == fb[cb]) { ++shared_non_vi; }
              }
            }
            // If they share exactly one non-vi vertex, they share an edge through vi.
            if (shared_non_vi >= 1) {
              uf.unite(a, b);
            }
          }
        }
        const size_t first_root = uf.find(0);
        for (size_t face = 1; face < degree; ++face) {
          if (uf.find(face) != first_root) {
            local_nonmanifold_vertices += 1;
            break;
          }
        }
      }
      nonmanifold_vertex_count.fetch_add(local_nonmanifold_vertices, std::memory_order_relaxed);
      connected_component_count.fetch_add(local_connected_components, std::memory_order_relaxed);
    });
    nonmanifold_vertices = nonmanifold_vertex_count.load(std::memory_order_relaxed);
    connected_components = connected_component_count.load(std::memory_order_relaxed);
  }

  nb::list blockers;
  if (mesh.faces.empty()) {
    blockers.append("no_faces");
  }
  if (degenerate_faces > 0) {
    blockers.append("degenerate_faces_present");
  }
  if (duplicate_faces > 0) {
    blockers.append("duplicate_faces_present");
  }
  if (nonmanifold_edges > 0) {
    blockers.append("nonmanifold_edges_present");
  }

  nb::dict result;
  result["vertex_count"] = static_cast<int64_t>(mesh.vertices.size());
  result["face_count"] = static_cast<int64_t>(mesh.faces.size());
  result["degenerate_faces"] = degenerate_faces;
  result["duplicate_faces"] = duplicate_faces;
  result["boundary_edges"] = boundary_edges;
  result["boundary_vertices"] = boundary.boundary_vertices;
  result["boundary_loop_count"] = boundary.loop_count;
  result["boundary_open_chain_count"] = boundary.open_chain_count;
  result["boundary_small_loop_count"] = boundary.small_loop_count;
  result["boundary_small_loop_edge_count"] = boundary.small_loop_edge_count;
  result["boundary_small_loop_threshold_edges"] = kSmallBoundaryLoopThresholdEdges;
  result["boundary_open_chain_edge_count"] = boundary.open_chain_edge_count;
  result["boundary_small_open_chain_count"] = boundary.small_open_chain_count;
  result["boundary_small_open_chain_edge_count"] = boundary.small_open_chain_edge_count;
  result["boundary_simple_open_chain_count"] = boundary.simple_open_chain_count;
  result["boundary_branched_open_chain_count"] = boundary.branched_open_chain_count;
  result["boundary_open_chain_endpoint_count"] = boundary.open_chain_endpoint_count;
  result["boundary_open_chain_branch_vertex_count"] = boundary.open_chain_branch_vertex_count;
  result["boundary_max_loop_edges"] = boundary.max_loop_edges;
  result["boundary_max_open_chain_edges"] = boundary.max_open_chain_edges;
  result["boundary_max_component_edges"] = boundary.max_component_edges;
  result["nonmanifold_edges"] = nonmanifold_edges;
  result["nonmanifold_vertices"] = nonmanifold_vertices;
  result["connected_components"] = connected_components;
  result["surface_area"] = total_surface_area;
  if (component_face_counts.empty()) {
    result["component_face_count_min"] = 0;
    result["component_face_count_median"] = 0;
    result["component_face_count_p95"] = 0;
    result["component_face_count_max"] = 0;
    result["largest_component_face_ratio"] = 0.0;
  } else {
    const size_t count = component_face_counts.size();
    result["component_face_count_min"] = component_face_counts.front();
    result["component_face_count_median"] = component_face_counts[(count - 1) / 2];
    result["component_face_count_p95"] = component_face_counts[(count * 95 + 99) / 100 - 1];
    result["component_face_count_max"] = component_face_counts.back();
    result["largest_component_face_ratio"] = static_cast<double>(component_face_counts.back()) /
        static_cast<double>(std::max<size_t>(1, mesh.faces.size()));
  }
  if (component_surface_areas.empty()) {
    result["component_surface_area_min"] = 0.0;
    result["component_surface_area_median"] = 0.0;
    result["component_surface_area_p95"] = 0.0;
    result["component_surface_area_max"] = 0.0;
    result["largest_component_surface_area_ratio"] = 0.0;
  } else {
    const size_t count = component_surface_areas.size();
    result["component_surface_area_min"] = component_surface_areas.front();
    result["component_surface_area_median"] = component_surface_areas[(count - 1) / 2];
    result["component_surface_area_p95"] = component_surface_areas[(count * 95 + 99) / 100 - 1];
    result["component_surface_area_max"] = component_surface_areas.back();
    result["largest_component_surface_area_ratio"] = total_surface_area > 0.0
        ? component_surface_areas.back() / total_surface_area
        : 0.0;
  }
  nb::dict component_histogram;
  nb::dict component_surface_area_by_face_count;
  for (const int64_t threshold : {32, 64, 128, 256, 512, 1024}) {
    int64_t component_count = 0;
    int64_t component_faces = 0;
    double component_area = 0.0;
    for (const ComponentMeasure &measure : components_by_face_count) {
      if (measure.face_count > threshold) {
        break;
      }
      component_count += 1;
      component_faces += measure.face_count;
      component_area += measure.surface_area;
    }
    nb::dict bucket;
    bucket["component_count"] = component_count;
    bucket["face_count"] = component_faces;
    component_histogram[std::to_string(threshold).c_str()] = std::move(bucket);

    nb::dict area_bucket;
    area_bucket["surface_area"] = component_area;
    area_bucket["surface_area_ratio"] = total_surface_area > 0.0
        ? component_area / total_surface_area
        : 0.0;
    component_surface_area_by_face_count[std::to_string(threshold).c_str()] = std::move(area_bucket);
  }
  result["component_face_histogram_le"] = std::move(component_histogram);
  result["component_surface_area_by_face_count_le"] = std::move(component_surface_area_by_face_count);
  result["cpu_workers"] = static_cast<int64_t>(native_worker_count(mesh.faces.size()));
  result["export_blocking_reasons"] = blockers;
  return result;
}

}  // namespace mlx_spatialkit
