# Roadmap

## Phase 1: Define The Root Library Boundary

Objective: Create a concrete root package identity, README, manifest, and minimal module layout for the MLX spatial library.

Why now: The bounded scan found no root package, source tree, README, install command, test command, or lint command.

Likely outputs: Root README, package manifest, `src/` or equivalent package layout, basic import smoke test, and explicit statement that `vendors/` is reference material.

Exit signal: A fresh clone can install the root package and import its top-level module without touching vendor setup flows.

Evidence: `.`, `vendors/`, `vendors/trellis-mac/pyproject.toml`, `vendors/sam-3d-objects/pyproject.toml`.

## Phase 2: Choose The First Spatial Slice

Objective: Select one narrow capability to port or build first.

Why now: The vendor corpus contains several competing directions: TRELLIS/O-Voxel image-to-3D, SAM object reconstruction, and Hunyuan geometric prediction.

Likely outputs: A short spec naming the first model family or shared primitive, supported inputs/outputs, and excluded vendor features.

Exit signal: Implementation can start without re-litigating project scope.

Evidence: `vendors/trellis-mac/README.md`, `vendors/TRELLIS.2/README.md`, `vendors/sam-3d-objects/README.md`, `vendors/HunyuanWorld-Mirror/README.md`.

## Phase 3: Establish MLX Tensor And Geometry Primitives

Objective: Build the smallest reusable MLX-backed primitives required by the first slice, such as tensor shape contracts, camera/depth helpers, voxel or point representations, and export boundaries.

Why now: Vendor references rely on PyTorch, CUDA, MPS, Metal, sparse convolution, voxel, mesh, point, and Gaussian representations.

Likely outputs: Minimal MLX modules, shape/type tests, and documented limitations compared with vendor behavior.

Exit signal: The first selected slice has tested local primitives independent of vendor imports.

Evidence: `vendors/trellis-mac/README.md`, `vendors/TRELLIS.2/README.md`, `vendors/HunyuanWorld-Mirror/README.md`.

## Phase 4: Implement One End-To-End Reference Flow

Objective: Deliver one small, runnable pipeline from input artifact to spatial output using first-party code and explicit vendor parity checks where feasible.

Why now: Vendor commands demonstrate complete flows, but the root project needs its own verified surface.

Likely outputs: Example command or script, fixture input, deterministic smoke test, and documented output format.

Exit signal: A user can run one root command or example without entering a vendor directory.

Evidence: `vendors/trellis-mac/README.md`, `vendors/sam-3d-objects/README.md`, `vendors/TRELLIS.2/README.md`.

## Phase 5: Verify Portability, Licensing, And Performance Boundaries

Objective: Convert the prototype into a maintainable library surface with documented hardware, dependency, and licensing boundaries.

Why now: The reference projects mix licenses, checkpoint access, hardware assumptions, and vendor-specific acceleration stacks.

Likely outputs: License notes, dependency policy, benchmark harness, memory notes, and explicit unsupported paths.

Exit signal: The root project can accept new model slices without re-solving repository policy.

Evidence: `vendors/trellis-mac/README.md`, `vendors/sam-3d-objects/README.md`, `vendors/TRELLIS.2/README.md`, `vendors/HunyuanWorld-Mirror/README.md`.

List the next implementation phases in order.
