#include "remesh.hpp"

#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "mesh_common.hpp"
#include "parallel.hpp"
#include "triangle_bvh.hpp"

namespace nb = nanobind;

namespace mlx_spatialkit {
namespace {

// 4 voxels sharing each axis-aligned grid edge (dual quad), matching
// remeshing.py:72 / flexi_dual_grid.cpp:104.
constexpr int kEdgeNeighborVoxelOffset[3][4][3] = {
    {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
    {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}},
    {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},
};

struct VoxelCoord {
  int64_t x;
  int64_t y;
  int64_t z;
};

struct CellDualData {
  int component_count = 0;
  std::array<std::array<double, 3>, 12> positions{};
  std::array<int8_t, 12> edge_components{};

  CellDualData() {
    edge_components.fill(-1);
  }
};

constexpr int cube_corner_id(int x, int y, int z) {
  return x * 4 + y * 2 + z;
}

constexpr int cube_edge_slot(int axis, int u, int v) {
  return axis * 4 + u * 2 + v;
}

constexpr int kCubeFaceCorners[6][4] = {
    {0, 2, 3, 1},
    {4, 5, 7, 6},
    {0, 1, 5, 4},
    {2, 6, 7, 3},
    {0, 4, 6, 2},
    {1, 3, 7, 5},
};

constexpr int kCubeFaceEdges[6][4] = {
    {4, 9, 5, 8},
    {10, 7, 11, 6},
    {8, 1, 10, 0},
    {2, 11, 3, 9},
    {0, 6, 2, 4},
    {5, 3, 7, 1},
};

std::array<double, 3> triangle_normal(
    const std::vector<std::array<double, 3>> &vertices,
    int64_t a,
    int64_t b,
    int64_t c) {
  const auto &va = vertices[static_cast<size_t>(a)];
  const auto &vb = vertices[static_cast<size_t>(b)];
  const auto &vc = vertices[static_cast<size_t>(c)];
  const std::array<double, 3> ab{vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]};
  const std::array<double, 3> ac{vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]};
  return {
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0],
  };
}

double normal_alignment(
    const std::vector<std::array<double, 3>> &vertices,
    const std::array<int64_t, 3> &left,
    const std::array<int64_t, 3> &right) {
  const auto left_normal = triangle_normal(vertices, left[0], left[1], left[2]);
  const auto right_normal = triangle_normal(vertices, right[0], right[1], right[2]);
  return std::abs(
      left_normal[0] * right_normal[0]
      + left_normal[1] * right_normal[1]
      + left_normal[2] * right_normal[2]);
}

// One pass of closure-preserving non-manifold-edge repair (CuMesh
// repair_non_manifold_edges semantics): at each vertex, group incident faces
// into fans linked by manifold edges (edges through the vertex used by exactly
// two faces). When a vertex carries more than one fan, the first fan keeps the
// vertex and each other fan gets a duplicate vertex at the same position, so
// every edge ends up incident to <= 2 faces without removing any face.
int64_t repair_non_manifold_edges_pass(mesh_common::MeshData &mesh) {
  const int64_t nv = static_cast<int64_t>(mesh.vertices.size());
  std::vector<std::vector<int64_t>> vert_faces(static_cast<size_t>(nv));
  for (int64_t fi = 0; fi < static_cast<int64_t>(mesh.faces.size()); ++fi) {
    for (int c = 0; c < 3; ++c) {
      vert_faces[static_cast<size_t>(mesh.faces[static_cast<size_t>(fi)][static_cast<size_t>(c)])].push_back(fi);
    }
  }
  int64_t added = 0;
  for (int64_t v = 0; v < nv; ++v) {
    const std::vector<int64_t> &inc = vert_faces[static_cast<size_t>(v)];
    const size_t k = inc.size();
    if (k <= 1) {
      continue;
    }
    std::vector<std::pair<int64_t, size_t>> nbr;
    nbr.reserve(k * 2);
    for (size_t i = 0; i < k; ++i) {
      const std::array<int64_t, 3> &f = mesh.faces[static_cast<size_t>(inc[i])];
      for (int c = 0; c < 3; ++c) {
        if (f[static_cast<size_t>(c)] == v) {
          nbr.emplace_back(f[static_cast<size_t>((c + 1) % 3)], i);
          nbr.emplace_back(f[static_cast<size_t>((c + 2) % 3)], i);
        }
      }
    }
    std::vector<size_t> parent(k);
    for (size_t i = 0; i < k; ++i) {
      parent[i] = i;
    }
    auto find = [&](size_t x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    std::sort(nbr.begin(), nbr.end());
    for (size_t i = 0; i < nbr.size();) {
      size_t j = i;
      while (j < nbr.size() && nbr[j].first == nbr[i].first) {
        ++j;
      }
      if (j - i == 2) {  // manifold edge through v -> the two faces share a fan
        parent[find(nbr[i].second)] = find(nbr[i + 1].second);
      }
      i = j;
    }
    const size_t first_root = find(0);
    bool single = true;
    for (size_t i = 1; i < k; ++i) {
      if (find(i) != first_root) {
        single = false;
        break;
      }
    }
    if (single) {
      continue;
    }
    std::unordered_map<size_t, int64_t> root_to_vertex;
    bool first_assigned = false;
    for (size_t i = 0; i < k; ++i) {
      const size_t r = find(i);
      int64_t target;
      const auto it = root_to_vertex.find(r);
      if (it != root_to_vertex.end()) {
        target = it->second;
      } else if (!first_assigned) {
        target = v;
        first_assigned = true;
        root_to_vertex.emplace(r, target);
      } else {
        const std::array<float, 3> position = mesh.vertices[static_cast<size_t>(v)];
        target = static_cast<int64_t>(mesh.vertices.size());
        mesh.vertices.push_back(position);
        ++added;
        root_to_vertex.emplace(r, target);
      }
      if (target != v) {
        std::array<int64_t, 3> &f = mesh.faces[static_cast<size_t>(inc[i])];
        for (int c = 0; c < 3; ++c) {
          if (f[static_cast<size_t>(c)] == v) {
            f[static_cast<size_t>(c)] = target;
          }
        }
      }
    }
  }
  return added;
}

int64_t repair_non_manifold_edges(mesh_common::MeshData &mesh) {
  int64_t total = 0;
  for (int iter = 0; iter < 6; ++iter) {
    const int64_t added = repair_non_manifold_edges_pass(mesh);
    total += added;
    if (added == 0) {
      break;
    }
  }
  return total;
}

}  // namespace

nb::dict remesh_narrow_band(
    nb::object vertices,
    nb::object faces,
    int64_t resolution,
    double band,
    double project_back,
    bool repair_nonmanifold) {
  if (resolution <= 0) {
    throw nb::value_error("resolution must be positive");
  }
  if (!(band > 0.0) || !std::isfinite(band)) {
    throw nb::value_error("band must be positive and finite");
  }
  if (!std::isfinite(project_back) || project_back < 0.0 || project_back > 1.0) {
    throw nb::value_error("project_back must be in [0, 1]");
  }

  const mesh_common::MeshData mesh = mesh_common::load_mesh(vertices, faces);
  if (mesh.faces.empty()) {
    throw nb::value_error("remesh requires at least one face");
  }
  nb::gil_scoped_release release;
  const TriangleBvh bvh(mesh);

  // AABB, center, padded scale, and isosurface offset (remeshing.py:166-174,100).
  std::array<float, 3> lo{mesh.vertices[0]};
  std::array<float, 3> hi{mesh.vertices[0]};
  for (const auto &v : mesh.vertices) {
    lo = min3(lo, v);
    hi = max3(hi, v);
  }
  const std::array<double, 3> center{
      0.5 * (static_cast<double>(lo[0]) + hi[0]),
      0.5 * (static_cast<double>(lo[1]) + hi[1]),
      0.5 * (static_cast<double>(lo[2]) + hi[2]),
  };
  double extent = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    extent = std::max(extent, static_cast<double>(hi[static_cast<size_t>(axis)]) - lo[static_cast<size_t>(axis)]);
  }
  if (!(extent > 0.0)) {
    throw nb::value_error("remesh input mesh is degenerate (zero extent)");
  }
  const double R = static_cast<double>(resolution);
  const double scale = (R + 3.0 * band) / R * extent;
  const double eps = band * scale / R;

  // Unsigned distance at a lattice point (corner units in [0, resolution]).
  auto udf_at_lattice = [&](double lx, double ly, double lz, double denom) -> double {
    const std::array<float, 3> p{
        static_cast<float>((lx / denom - 0.5) * scale + center[0]),
        static_cast<float>((ly / denom - 0.5) * scale + center[1]),
        static_cast<float>((lz / denom - 0.5) * scale + center[2]),
    };
    return std::sqrt(bvh.closest_point(p).distance2);
  };

  // 1. Narrow band: coarse -> fine subdivision (remeshing.py:103-141).
  int64_t base = resolution;
  while (base > 32 && base % 2 == 0) {
    base /= 2;
  }
  std::vector<VoxelCoord> coords;
  for (int64_t x = 0; x < base; ++x) {
    for (int64_t y = 0; y < base; ++y) {
      for (int64_t z = 0; z < base; ++z) {
        coords.push_back({x, y, z});
      }
    }
  }
  while (true) {
    const double cell = scale / static_cast<double>(base);
    const double thresh = 0.87 * cell;
    std::vector<uint8_t> keep_mask(coords.size(), 0);
    parallel_for(coords.size(), [&](size_t begin, size_t end) {
      for (size_t index = begin; index < end; ++index) {
        const auto &c = coords[index];
        const double d = udf_at_lattice(
                             static_cast<double>(c.x) + 0.5,
                             static_cast<double>(c.y) + 0.5,
                             static_cast<double>(c.z) + 0.5,
                             static_cast<double>(base)) -
                         eps;
        keep_mask[index] = std::abs(d) < thresh ? 1 : 0;
      }
    });
    std::vector<VoxelCoord> kept;
    kept.reserve(coords.size());
    for (size_t index = 0; index < coords.size(); ++index) {
      if (keep_mask[index] != 0) {
        kept.push_back(coords[index]);
      }
    }
    coords.swap(kept);
    if (base >= resolution) {
      break;
    }
    base *= 2;
    std::vector<VoxelCoord> children;
    children.reserve(coords.size() * 8);
    for (const auto &c : coords) {
      for (int dx = 0; dx <= 1; ++dx) {
        for (int dy = 0; dy <= 1; ++dy) {
          for (int dz = 0; dz <= 1; ++dz) {
            children.push_back({c.x * 2 + dx, c.y * 2 + dy, c.z * 2 + dz});
          }
        }
      }
    }
    coords.swap(children);
  }
  if (coords.empty()) {
    nb::gil_scoped_acquire acquire;
    return mesh_common::mesh_result(mesh_common::MeshData{});
  }

  // 2. Voxel lookup at full resolution.
  const int64_t R1 = resolution + 1;
  auto voxel_key = [resolution](int64_t x, int64_t y, int64_t z) {
    return (x * resolution + y) * resolution + z;
  };
  std::unordered_map<int64_t, int64_t> voxel_index;
  voxel_index.reserve(coords.size() * 2);
  for (size_t i = 0; i < coords.size(); ++i) {
    voxel_index[voxel_key(coords[i].x, coords[i].y, coords[i].z)] = static_cast<int64_t>(i);
  }

  // 3. Precompute the unique signed UDF grid corners in parallel. The map is
  // immutable while worker threads query it in the dual-contour pass.
  std::unordered_map<int64_t, size_t> corner_index;
  std::vector<int64_t> corner_keys;
  corner_index.reserve(coords.size() * 2);
  corner_keys.reserve(coords.size() * 2);
  for (const auto &coord : coords) {
    for (int dx = 0; dx <= 1; ++dx) {
      for (int dy = 0; dy <= 1; ++dy) {
        for (int dz = 0; dz <= 1; ++dz) {
          const int64_t key = ((coord.x + dx) * R1 + coord.y + dy) * R1 + coord.z + dz;
          const size_t index = corner_keys.size();
          if (corner_index.emplace(key, index).second) {
            corner_keys.push_back(key);
          }
        }
      }
    }
  }
  std::vector<double> corner_values(corner_keys.size());
  parallel_for(corner_keys.size(), [&](size_t begin, size_t end) {
    const int64_t plane = R1 * R1;
    for (size_t index = begin; index < end; ++index) {
      const int64_t key = corner_keys[index];
      const int64_t x = key / plane;
      const int64_t remainder = key % plane;
      const int64_t y = remainder / R1;
      const int64_t z = remainder % R1;
      corner_values[index] = udf_at_lattice(
                                 static_cast<double>(x),
                                 static_cast<double>(y),
                                 static_cast<double>(z),
                                 R) -
                             eps;
    }
  });
  auto udf_corner = [&](int64_t x, int64_t y, int64_t z) -> double {
    const int64_t key = (x * R1 + y) * R1 + z;
    const auto it = corner_index.find(key);
    if (it == corner_index.end()) {
      throw std::runtime_error("dual-contour corner was not precomputed");
    }
    return corner_values[it->second];
  };

  // 4. Manifold dual contour cell vertices. The reference simple DC kernel
  //    emits one vertex per voxel, which welds disconnected isosurface patches
  //    inside ambiguous cells. Build a contour graph from shared-face pairings
  //    and emit one mean-intersection vertex per graph component.
  std::vector<CellDualData> cell_duals(coords.size());
  std::vector<std::array<int, 3>> crossing(coords.size());
  parallel_for(coords.size(), [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      const int64_t vx = coords[i].x;
      const int64_t vy = coords[i].y;
      const int64_t vz = coords[i].z;
      std::array<double, 8> values{};
      for (int x = 0; x <= 1; ++x) {
        for (int y = 0; y <= 1; ++y) {
          for (int z = 0; z <= 1; ++z) {
            values[static_cast<size_t>(cube_corner_id(x, y, z))] = udf_corner(vx + x, vy + y, vz + z);
          }
        }
      }

      std::array<int, 12> parent{};
      std::iota(parent.begin(), parent.end(), 0);
      auto find_root = [&](int edge) {
        int root = edge;
        while (parent[static_cast<size_t>(root)] != root) {
          root = parent[static_cast<size_t>(root)];
        }
        while (parent[static_cast<size_t>(edge)] != edge) {
          const int next = parent[static_cast<size_t>(edge)];
          parent[static_cast<size_t>(edge)] = root;
          edge = next;
        }
        return root;
      };
      auto unite_edges = [&](int left, int right) {
        const int left_root = find_root(left);
        const int right_root = find_root(right);
        if (left_root != right_root) {
          parent[static_cast<size_t>(std::max(left_root, right_root))] = std::min(left_root, right_root);
        }
      };
      CellDualData &cell = cell_duals[i];
      std::array<uint8_t, 12> edge_active{};
      std::array<std::array<double, 3>, 12> edge_positions{};
      std::array<int, 3> flags{0, 0, 0};
      for (int axis = 0; axis < 3; ++axis) {
        for (int u = 0; u <= 1; ++u) {
          for (int v = 0; v <= 1; ++v) {
            int left = 0;
            int right = 0;
            std::array<double, 3> left_position{};
            std::array<double, 3> right_position{};
            if (axis == 0) {
              left = cube_corner_id(0, u, v);
              right = cube_corner_id(1, u, v);
              left_position = {static_cast<double>(vx), static_cast<double>(vy + u), static_cast<double>(vz + v)};
              right_position = {static_cast<double>(vx + 1), static_cast<double>(vy + u), static_cast<double>(vz + v)};
            } else if (axis == 1) {
              left = cube_corner_id(u, 0, v);
              right = cube_corner_id(u, 1, v);
              left_position = {static_cast<double>(vx + u), static_cast<double>(vy), static_cast<double>(vz + v)};
              right_position = {static_cast<double>(vx + u), static_cast<double>(vy + 1), static_cast<double>(vz + v)};
            } else {
              left = cube_corner_id(u, v, 0);
              right = cube_corner_id(u, v, 1);
              left_position = {static_cast<double>(vx + u), static_cast<double>(vy + v), static_cast<double>(vz)};
              right_position = {static_cast<double>(vx + u), static_cast<double>(vy + v), static_cast<double>(vz + 1)};
            }
            const double left_value = values[static_cast<size_t>(left)];
            const double right_value = values[static_cast<size_t>(right)];
            if ((left_value < 0.0) == (right_value < 0.0)) {
              continue;
            }
            const int edge = cube_edge_slot(axis, u, v);
            edge_active[static_cast<size_t>(edge)] = 1;
            const double t = -left_value / (right_value - left_value);
            for (int coordinate_axis = 0; coordinate_axis < 3; ++coordinate_axis) {
              edge_positions[static_cast<size_t>(edge)][static_cast<size_t>(coordinate_axis)] =
                  left_position[static_cast<size_t>(coordinate_axis)]
                  + t * (right_position[static_cast<size_t>(coordinate_axis)]
                         - left_position[static_cast<size_t>(coordinate_axis)]);
            }
            if (u == 1 && v == 1) {
              flags[static_cast<size_t>(axis)] = left_value < 0.0 ? 1 : -1;
            }
          }
        }
      }

      // Build a contour graph on the six shared cube faces. The 4-crossing
      // checkerboard case uses the face-center sign as a deterministic decider;
      // neighboring voxels see the same four values and therefore make the
      // same pairing decision.
      for (int face = 0; face < 6; ++face) {
        std::array<int, 4> active_face_edges{};
        int active_count = 0;
        for (int edge_index = 0; edge_index < 4; ++edge_index) {
          const int edge = kCubeFaceEdges[face][edge_index];
          if (edge_active[static_cast<size_t>(edge)] != 0) {
            active_face_edges[static_cast<size_t>(active_count++)] = edge;
          }
        }
        if (active_count == 2) {
          unite_edges(active_face_edges[0], active_face_edges[1]);
        } else if (active_count == 4) {
          double face_center_value = 0.0;
          for (int corner_index = 0; corner_index < 4; ++corner_index) {
            face_center_value += values[static_cast<size_t>(kCubeFaceCorners[face][corner_index])];
          }
          const bool center_negative = face_center_value < 0.0;
          for (int corner_index = 0; corner_index < 4; ++corner_index) {
            const bool corner_negative =
                values[static_cast<size_t>(kCubeFaceCorners[face][corner_index])] < 0.0;
            if (corner_negative != center_negative) {
              unite_edges(
                  kCubeFaceEdges[face][(corner_index + 3) % 4],
                  kCubeFaceEdges[face][corner_index]);
            }
          }
        }
      }

      std::array<int8_t, 12> root_to_component{};
      root_to_component.fill(-1);
      std::array<std::array<double, 3>, 12> sums{};
      std::array<int, 12> counts{};
      for (int edge = 0; edge < 12; ++edge) {
        if (edge_active[static_cast<size_t>(edge)] == 0) {
          continue;
        }
        const int root = find_root(edge);
        int component = root_to_component[static_cast<size_t>(root)];
        if (component < 0) {
          component = cell.component_count++;
          root_to_component[static_cast<size_t>(root)] = static_cast<int8_t>(component);
        }
        cell.edge_components[static_cast<size_t>(edge)] = static_cast<int8_t>(component);
        for (int axis = 0; axis < 3; ++axis) {
          sums[static_cast<size_t>(component)][static_cast<size_t>(axis)] +=
              edge_positions[static_cast<size_t>(edge)][static_cast<size_t>(axis)];
        }
        counts[static_cast<size_t>(component)] += 1;
      }
      for (int component = 0; component < cell.component_count; ++component) {
        const double denominator = static_cast<double>(std::max(1, counts[static_cast<size_t>(component)]));
        for (int axis = 0; axis < 3; ++axis) {
          cell.positions[static_cast<size_t>(component)][static_cast<size_t>(axis)] =
              sums[static_cast<size_t>(component)][static_cast<size_t>(axis)] / denominator;
        }
      }
      crossing[i] = flags;
    }
  });

  std::vector<int64_t> cell_vertex_offsets(coords.size() + 1, 0);
  int64_t split_dual_cells = 0;
  for (size_t index = 0; index < cell_duals.size(); ++index) {
    cell_vertex_offsets[index + 1] =
        cell_vertex_offsets[index] + static_cast<int64_t>(cell_duals[index].component_count);
    split_dual_cells += cell_duals[index].component_count > 1 ? 1 : 0;
  }
  std::vector<std::array<double, 3>> dual(static_cast<size_t>(cell_vertex_offsets.back()));
  parallel_for(cell_duals.size(), [&](size_t begin, size_t end) {
    for (size_t cell_index = begin; cell_index < end; ++cell_index) {
      for (int component = 0; component < cell_duals[cell_index].component_count; ++component) {
        dual[static_cast<size_t>(cell_vertex_offsets[cell_index] + component)] =
            cell_duals[cell_index].positions[static_cast<size_t>(component)];
      }
    }
  });

  // 5. Connectivity: each crossed edge -> quad of its 4 neighbor voxels'
  //    dual vertices -> 2 triangles (remeshing.py:191-233). Winding follows the
  //    CuMesh crossing-sign tables. Select the split with the stronger adjacent
  //    triangle-normal alignment, which is the intended reference criterion.
  std::vector<std::array<int64_t, 3>> faces_out;
  int64_t split_1_quads = 0;
  int64_t split_2_quads = 0;
  for (size_t i = 0; i < coords.size(); ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const int dir = crossing[i][static_cast<size_t>(axis)];
      if (dir == 0) {
        continue;
      }
      int64_t quad[4];
      bool ok = true;
      for (int k = 0; k < 4; ++k) {
        const int offset_x = kEdgeNeighborVoxelOffset[axis][k][0];
        const int offset_y = kEdgeNeighborVoxelOffset[axis][k][1];
        const int offset_z = kEdgeNeighborVoxelOffset[axis][k][2];
        const int64_t nx = coords[i].x + offset_x;
        const int64_t ny = coords[i].y + offset_y;
        const int64_t nz = coords[i].z + offset_z;
        const auto it = voxel_index.find(voxel_key(nx, ny, nz));
        if (it == voxel_index.end()) {
          ok = false;
          break;
        }
        int local_u = 0;
        int local_v = 0;
        if (axis == 0) {
          local_u = 1 - offset_y;
          local_v = 1 - offset_z;
        } else if (axis == 1) {
          local_u = 1 - offset_x;
          local_v = 1 - offset_z;
        } else {
          local_u = 1 - offset_x;
          local_v = 1 - offset_y;
        }
        const size_t neighbor_cell = static_cast<size_t>(it->second);
        const int component = cell_duals[neighbor_cell].edge_components[
            static_cast<size_t>(cube_edge_slot(axis, local_u, local_v))];
        if (component < 0) {
          ok = false;
          break;
        }
        quad[k] = cell_vertex_offsets[neighbor_cell] + component;
      }
      if (!ok) {
        continue;
      }
      const std::array<int64_t, 3> split_1_left{quad[0], quad[1], quad[2]};
      const std::array<int64_t, 3> split_1_right{quad[0], quad[2], quad[3]};
      const std::array<int64_t, 3> split_2_left{quad[0], quad[1], quad[3]};
      const std::array<int64_t, 3> split_2_right{quad[3], quad[1], quad[2]};
      const bool use_split_1 = normal_alignment(dual, split_1_left, split_1_right)
          > normal_alignment(dual, split_2_left, split_2_right);
      if (use_split_1) {
        if (dir > 0) {
          faces_out.push_back({quad[0], quad[2], quad[1]});
          faces_out.push_back({quad[0], quad[3], quad[2]});
        } else {
          faces_out.push_back(split_1_left);
          faces_out.push_back(split_1_right);
        }
        split_1_quads += 1;
      } else {
        if (dir > 0) {
          faces_out.push_back({quad[0], quad[3], quad[1]});
          faces_out.push_back({quad[3], quad[2], quad[1]});
        } else {
          faces_out.push_back(split_2_left);
          faces_out.push_back(split_2_right);
        }
        split_2_quads += 1;
      }
    }
  }

  // 6. Lattice -> world; optional project-back onto the original surface.
  mesh_common::MeshData out;
  out.vertices.resize(dual.size());
  parallel_for(dual.size(), [&](size_t begin, size_t end) {
    for (size_t index = begin; index < end; ++index) {
      const auto &d = dual[index];
      std::array<float, 3> world{
          static_cast<float>((d[0] / R - 0.5) * scale + center[0]),
          static_cast<float>((d[1] / R - 0.5) * scale + center[1]),
          static_cast<float>((d[2] / R - 0.5) * scale + center[2]),
      };
      if (project_back > 0.0) {
        const ClosestPointResult cp = bvh.closest_point(world);
        for (int axis = 0; axis < 3; ++axis) {
          world[static_cast<size_t>(axis)] -=
              static_cast<float>(project_back) * (world[static_cast<size_t>(axis)] - cp.point[static_cast<size_t>(axis)]);
        }
      }
      out.vertices[index] = world;
    }
  });
  out.faces = std::move(faces_out);

  const int64_t manifold_repair_vertices_added = repair_nonmanifold ? repair_non_manifold_edges(out) : 0;

  int64_t unreferenced_removed = 0;
  const mesh_common::MeshData compacted = mesh_common::compact_mesh(out, &unreferenced_removed);

  nb::gil_scoped_acquire acquire;
  nb::dict result = mesh_common::mesh_result(compacted);
  nb::dict stats;
  stats["backend"] = "cpu-narrow-band-udf-double-cover-dc";
  stats["framework"] = "native-c++";
  stats["execution_device"] = "cpu";
  stats["surface_representation"] = "udf-offset-double-cover";
  stats["single_surface_ready"] = false;
  stats["resolution"] = resolution;
  stats["band"] = band;
  stats["project_back"] = project_back;
  stats["eps"] = eps;
  stats["scale"] = scale;
  stats["active_voxels"] = static_cast<int64_t>(coords.size());
  stats["grid_vertices_sampled"] = static_cast<int64_t>(corner_values.size());
  stats["dual_contour_topology"] = "shared-face-contour-graph-component-split";
  stats["split_dual_cells"] = split_dual_cells;
  stats["dual_vertices_before_compaction"] = static_cast<int64_t>(dual.size());
  stats["quad_split_strategy"] = "max-adjacent-face-normal-alignment";
  stats["quad_split_1_quads"] = split_1_quads;
  stats["quad_split_2_quads"] = split_2_quads;
  stats["cpu_workers"] = static_cast<int64_t>(native_worker_count(std::max(coords.size(), corner_values.size())));
  stats["manifold_repair_vertices_added"] = manifold_repair_vertices_added;
  stats["bvh_nodes"] = bvh.node_count();
  stats["input_vertices"] = static_cast<int64_t>(mesh.vertices.size());
  stats["input_faces"] = static_cast<int64_t>(mesh.faces.size());
  stats["output_vertices"] = static_cast<int64_t>(compacted.vertices.size());
  stats["output_faces"] = static_cast<int64_t>(compacted.faces.size());
  result["stats"] = stats;
  return result;
}

nb::dict repair_nonmanifold_mesh(nb::object vertices, nb::object faces) {
  mesh_common::MeshData mesh = mesh_common::load_mesh(vertices, faces);
  const int64_t input_vertices = static_cast<int64_t>(mesh.vertices.size());
  const int64_t input_faces = static_cast<int64_t>(mesh.faces.size());
  int64_t vertices_added = 0;
  int64_t unreferenced_removed = 0;
  mesh_common::MeshData compacted;
  {
    nb::gil_scoped_release release;
    vertices_added = repair_non_manifold_edges(mesh);
    compacted = mesh_common::compact_mesh(mesh, &unreferenced_removed);
  }

  nb::dict result = mesh_common::mesh_result(compacted);
  nb::dict stats;
  stats["backend"] = "native-cpp-face-fan-split";
  stats["framework"] = "native-c++";
  stats["execution_device"] = "cpu";
  stats["vertices_added"] = vertices_added;
  stats["unreferenced_vertices_removed"] = unreferenced_removed;
  stats["input_vertices"] = input_vertices;
  stats["input_faces"] = input_faces;
  stats["output_vertices"] = static_cast<int64_t>(compacted.vertices.size());
  stats["output_faces"] = static_cast<int64_t>(compacted.faces.size());
  result["stats"] = stats;
  return result;
}

}  // namespace mlx_spatialkit
