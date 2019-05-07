#pragma once

#include "span.h"
#include <numpy/ndarrayobject.h>
#include <pybind11/pybind11.h>

namespace spectre {
  static void import_numpy()
  {
    // NOTE: the lambda call is necessary to correctly throw exception when
    // importing numpy fails. The thrown exception is later caught by pybind11
    if (![]() { import_array1(false) return true; }())
      throw pybind11::error_already_set{};
  }

  template <typename> struct dtype;

  template <> struct dtype<npy_int32> {
    constexpr static const int value = NPY_INT32;
    constexpr static const char name[] = "int32";
  };

  template <> struct dtype<npy_int64> {
    constexpr static const int value = NPY_INT64;
    constexpr static const char name[] = "int64";
  };

  template <> struct dtype<npy_float32> {
    constexpr static const int value = NPY_FLOAT32;
    constexpr static const char name[] = "float32";
  };

  template <> struct dtype<npy_float64> {
    constexpr static const int value = NPY_FLOAT64;
    constexpr static const char name[] = "float64";
  };
} // namespace spectre

namespace pybind11::detail {
  template <typename T> struct type_caster<::spectre::span<T>> {
    using value_t = ::spectre::span<T>;
    using dtype_t = ::spectre::dtype<std::decay_t<T>>;
    using data_t = typename value_t::pointer;
    using size_t = typename value_t::size_type;

    PYBIND11_TYPE_CASTER(value_t, _("array[") + _(dtype_t::name) + _("]"));

    bool load(handle op, bool)
    {
      auto const arr = reinterpret_cast<PyArrayObject *>(op.ptr());
      auto const flags =
          std::is_const_v<T> ? NPY_ARRAY_IN_ARRAY : NPY_ARRAY_OUT_ARRAY;

      if (!PyArray_Check(op.ptr()) ||
          !PyArray_EquivTypenums(PyArray_TYPE(arr), dtype_t::value) ||
          PyArray_NDIM(arr) != 1 || !PyArray_CHKFLAGS(arr, flags))
        return false;

      value = value_t{static_cast<data_t>(PyArray_DATA(arr)),
                      static_cast<size_t>(PyArray_SIZE(arr))};
      return true;
    }
  };
} // namespace pybind11::detail
