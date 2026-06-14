#include "remesh.hpp"

#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "mesh_common.hpp"
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

double dist2(const std::array<double, 3> &a, const std::array<double, 3> &b) {
  const double dx = a[0] - b[0];
  const double dy = a[1] - b[1];
  const double dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
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
    std::vector<VoxelCoord> kept;
    kept.reserve(coords.size());
    for (const auto &c : coords) {
      const double d = udf_at_lattice(
                           static_cast<double>(c.x) + 0.5,
                           static_cast<double>(c.y) + 0.5,
                           static_cast<double>(c.z) + 0.5,
                           static_cast<double>(base)) -
                       eps;
      if (std::abs(d) < thresh) {
        kept.push_back(c);
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

  // 3. Cached signed UDF at grid corners (corner units in [0, resolution]).
  std::unordered_map<int64_t, double> corner_udf;
  auto udf_corner = [&](int64_t x, int64_t y, int64_t z) -> double {
    const int64_t key = (x * R1 + y) * R1 + z;
    const auto it = corner_udf.find(key);
    if (it != corner_udf.end()) {
      return it->second;
    }
    const double value = udf_at_lattice(
                             static_cast<double>(x),
                             static_cast<double>(y),
                             static_cast<double>(z),
                             R) -
                         eps;
    corner_udf.emplace(key, value);
    return value;
  };

  // 4. Simple dual contour: dual vertex = mean of edge intersections
  //    (simple_dual_contour.cu:52-155); 3 axis crossing flags from the far edge.
  std::vector<std::array<double, 3>> dual(coords.size());
  std::vector<std::array<int, 3>> crossing(coords.size());
  for (size_t i = 0; i < coords.size(); ++i) {
    const int64_t vx = coords[i].x;
    const int64_t vy = coords[i].y;
    const int64_t vz = coords[i].z;
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    int count = 0;
    std::array<int, 3> flags{0, 0, 0};
    for (int u = 0; u <= 1; ++u) {
      for (int v = 0; v <= 1; ++v) {
        // X edge.
        {
          const double v1 = udf_corner(vx, vy + u, vz + v);
          const double v2 = udf_corner(vx + 1, vy + u, vz + v);
          if ((v1 < 0.0) != (v2 < 0.0)) {
            const double t = -v1 / (v2 - v1);
            sx += static_cast<double>(vx) + t;
            sy += static_cast<double>(vy + u);
            sz += static_cast<double>(vz + v);
            ++count;
          }
          if (u == 1 && v == 1) {
            flags[0] = (v1 < 0.0 && v2 >= 0.0) ? 1 : ((v1 >= 0.0 && v2 < 0.0) ? -1 : 0);
          }
        }
        // Y edge.
        {
          const double v1 = udf_corner(vx + u, vy, vz + v);
          const double v2 = udf_corner(vx + u, vy + 1, vz + v);
          if ((v1 < 0.0) != (v2 < 0.0)) {
            const double t = -v1 / (v2 - v1);
            sx += static_cast<double>(vx + u);
            sy += static_cast<double>(vy) + t;
            sz += static_cast<double>(vz + v);
            ++count;
          }
          if (u == 1 && v == 1) {
            flags[1] = (v1 < 0.0 && v2 >= 0.0) ? 1 : ((v1 >= 0.0 && v2 < 0.0) ? -1 : 0);
          }
        }
        // Z edge.
        {
          const double v1 = udf_corner(vx + u, vy + v, vz);
          const double v2 = udf_corner(vx + u, vy + v, vz + 1);
          if ((v1 < 0.0) != (v2 < 0.0)) {
            const double t = -v1 / (v2 - v1);
            sx += static_cast<double>(vx + u);
            sy += static_cast<double>(vy + v);
            sz += static_cast<double>(vz) + t;
            ++count;
          }
          if (u == 1 && v == 1) {
            flags[2] = (v1 < 0.0 && v2 >= 0.0) ? 1 : ((v1 >= 0.0 && v2 < 0.0) ? -1 : 0);
          }
        }
      }
    }
    if (count > 0) {
      dual[i] = {sx / count, sy / count, sz / count};
    } else {
      dual[i] = {static_cast<double>(vx) + 0.5, static_cast<double>(vy) + 0.5, static_cast<double>(vz) + 0.5};
    }
    crossing[i] = flags;
  }

  // 5. Connectivity: each crossed edge -> quad of its 4 neighbor voxels'
  //    dual vertices -> 2 triangles (remeshing.py:191-233). Winding follows the
  //    crossing sign; the diagonal is the shorter one (deterministic).
  std::vector<std::array<int64_t, 3>> faces_out;
  for (size_t i = 0; i < coords.size(); ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const int dir = crossing[i][static_cast<size_t>(axis)];
      if (dir == 0) {
        continue;
      }
      int64_t quad[4];
      bool ok = true;
      for (int k = 0; k < 4; ++k) {
        const int64_t nx = coords[i].x + kEdgeNeighborVoxelOffset[axis][k][0];
        const int64_t ny = coords[i].y + kEdgeNeighborVoxelOffset[axis][k][1];
        const int64_t nz = coords[i].z + kEdgeNeighborVoxelOffset[axis][k][2];
        const auto it = voxel_index.find(voxel_key(nx, ny, nz));
        if (it == voxel_index.end()) {
          ok = false;
          break;
        }
        quad[k] = it->second;
      }
      if (!ok) {
        continue;
      }
      int64_t q[4];
      if (dir > 0) {
        q[0] = quad[0];
        q[1] = quad[1];
        q[2] = quad[2];
        q[3] = quad[3];
      } else {
        q[0] = quad[0];
        q[1] = quad[3];
        q[2] = quad[2];
        q[3] = quad[1];
      }
      if (dist2(dual[static_cast<size_t>(q[0])], dual[static_cast<size_t>(q[2])]) <=
          dist2(dual[static_cast<size_t>(q[1])], dual[static_cast<size_t>(q[3])])) {
        faces_out.push_back({q[0], q[1], q[2]});
        faces_out.push_back({q[0], q[2], q[3]});
      } else {
        faces_out.push_back({q[1], q[2], q[3]});
        faces_out.push_back({q[1], q[3], q[0]});
      }
    }
  }

  // 6. Lattice -> world; optional project-back onto the original surface.
  mesh_common::MeshData out;
  out.vertices.reserve(dual.size());
  for (const auto &d : dual) {
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
    out.vertices.push_back(world);
  }
  out.faces = std::move(faces_out);

  const int64_t manifold_repair_vertices_added = repair_nonmanifold ? repair_non_manifold_edges(out) : 0;

  int64_t unreferenced_removed = 0;
  const mesh_common::MeshData compacted = mesh_common::compact_mesh(out, &unreferenced_removed);

  nb::dict result = mesh_common::mesh_result(compacted);
  nb::dict stats;
  stats["backend"] = "cpu-narrow-band-dc";
  stats["resolution"] = resolution;
  stats["band"] = band;
  stats["project_back"] = project_back;
  stats["eps"] = eps;
  stats["scale"] = scale;
  stats["active_voxels"] = static_cast<int64_t>(coords.size());
  stats["grid_vertices_sampled"] = static_cast<int64_t>(corner_udf.size());
  stats["manifold_repair_vertices_added"] = manifold_repair_vertices_added;
  stats["bvh_nodes"] = bvh.node_count();
  stats["input_vertices"] = static_cast<int64_t>(mesh.vertices.size());
  stats["input_faces"] = static_cast<int64_t>(mesh.faces.size());
  stats["output_vertices"] = static_cast<int64_t>(compacted.vertices.size());
  stats["output_faces"] = static_cast<int64_t>(compacted.faces.size());
  result["stats"] = stats;
  return result;
}

}  // namespace mlx_spatialkit
