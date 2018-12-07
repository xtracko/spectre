#include "array_canonical.h"
#include "array_moving.h"
#include "array_python.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i32 = std::int32_t;
using i64 = std::int64_t;

using f32 = std::float_t;
using f64 = std::double_t;

template <typename Ty, typename Ix, typename Jx>
bool py_is_canonical(pycoo<Ty, Ix, Jx> const& array, int axis) {
  py::call_guard<py::gil_scoped_release>();

  switch (axis) {
  case 0:
    return is_canonical(array, row_major_order{});
  case 1:
    return is_canonical(array, col_major_order{});
  }

  throw std::invalid_argument{"Invalid axis argument"};
}

template <typename Ty, typename Ix, typename Jx>
decltype(auto) py_to_canonical(pycoo<Ty, Ix, Jx> & array, int axis) {
  py::call_guard<py::gil_scoped_release>();

  switch (axis) {
    case 0:
      return to_canonical(array, row_major_order{});
    case 1:
      return to_canonical(array, col_major_order{});
  }

  throw std::invalid_argument{"Invalid axis argument"};
}

template <typename Ty, typename Ix, typename Jx>
decltype(auto) py_moving_min(pycoo<Ty, Ix, Jx>& array, const Jx window) {
  py::call_guard<py::gil_scoped_release>();

  // moving_min<0>(array);
}

PYBIND11_MODULE(sparse2, m) {
  m.def("is_canonical", py_is_canonical<f32, i32, i32>);
  m.def("is_canonical", py_is_canonical<f32, i64, i64>);
  m.def("is_canonical", py_is_canonical<f64, i32, i32>);
  m.def("is_canonical", py_is_canonical<f64, i64, i64>);

  m.def("to_canonical", py_to_canonical<f32, i32, i32>);
  m.def("to_canonical", py_to_canonical<f32, i64, i64>);
  m.def("to_canonical", py_to_canonical<f64, i32, i32>);
  m.def("to_canonical", py_to_canonical<f64, i64, i64>);

  m.def("moving_min", py_moving_min<f32, i32, i32>);
  m.def("moving_min", py_moving_min<f32, i64, i64>);
  m.def("moving_min", py_moving_min<f64, i32, i32>);
  m.def("moving_min", py_moving_min<f64, i64, i64>);
}
