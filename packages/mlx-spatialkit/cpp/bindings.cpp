#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "metal_probe.hpp"
#include "pixal3d_contracts.hpp"

namespace nb = nanobind;

NB_MODULE(_native, module) {
  module.doc() = "Native C++/Metal bindings for mlx-spatialkit";

  module.def("metal_device_available", &mlx_spatialkit::metal_device_available,
             "Return whether the default Metal device is available.");

  module.def("backend_info", []() {
    nb::dict info;
    info["native"] = true;
    info["metal_available"] = mlx_spatialkit::metal_device_available();
    info["metal_device"] = mlx_spatialkit::metal_device_name();
    return info;
  }, "Return native backend build and Metal availability information.");

  module.def("validate_pixal3d_shape_fields",
             &mlx_spatialkit::validate_pixal3d_shape_fields,
             "Validate Pixal3D decoded shape coordinate and field arrays.");

  module.def("validate_pixal3d_texture_attributes",
             &mlx_spatialkit::validate_pixal3d_texture_attributes,
             "Validate Pixal3D decoded texture coordinate and attribute arrays.");
}
