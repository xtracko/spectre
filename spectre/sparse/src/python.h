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

  template <typename> struct py_dtype;

  template <> struct py_dtype<npy_int32> {
    constexpr static const int value = NPY_INT32;
    constexpr static const char name[] = "int32";
  };

  template <> struct py_dtype<npy_int64> {
    constexpr static const int value = NPY_INT64;
    constexpr static const char name[] = "int64";
  };

  template <> struct py_dtype<npy_float32> {
    constexpr static const int value = NPY_FLOAT32;
    constexpr static const char name[] = "float32";
  };

  template <> struct py_dtype<npy_float64> {
    constexpr static const int value = NPY_FLOAT64;
    constexpr static const char name[] = "float64";
  };

  template <typename T> struct py_array {
    constexpr static const int dtype = py_dtype<std::decay_t<T>>::value;
    constexpr static const int ndims = 1;
    constexpr static const int flags =
        std::is_const_v<T> ? NPY_ARRAY_IN_ARRAY : NPY_ARRAY_OUT_ARRAY;

    using size_type = npy_intp;
    using value_type = std::remove_cv_t<T>;
    using element_type = T;
    using pointer = T *;
    using iterator = T *;
    using reference = T &;
    using const_pointer = T const *;
    using const_iterator = T const *;
    using const_reference = T const &;

    py_array() noexcept : _op{nullptr}
    {}

    ~py_array()
    {
      Py_XDECREF(_op);
    }

    py_array(py_array &&other) noexcept : _op{other._op}
    {
      other._op = nullptr;
    }

    py_array(py_array const &other) noexcept : _op{other._op}
    {
      Py_XINCREF(_op);
    }

    py_array &operator=(py_array other) noexcept
    {
      swap(other);
      return *this;
    }

    void swap(py_array &other) noexcept
    {
      using std::swap;
      swap(_op, other._op);
    }

    explicit operator bool() const noexcept
    {
      return bool(_op);
    }

    PyObject *release() && noexcept
    {
      auto temp = _op;
      _op = nullptr;
      return reinterpret_cast<PyObject *>(temp);
    }

    bool empty() const noexcept
    {
      assert(_op);
      return PyArray_SIZE(_op) == 0;
    }

    size_type size() const noexcept
    {
      assert(_op);
      return static_cast<size_type>(PyArray_SIZE(_op));
    }

    pointer data() noexcept
    {
      assert(_op);
      assert(PyArray_CHKFLAGS(_op, NPY_ARRAY_OUT_ARRAY));
      return static_cast<pointer>(PyArray_DATA(_op));
    }

    const_pointer data() const noexcept
    {
      assert(_op);
      assert(PyArray_CHKFLAGS(_op, NPY_ARRAY_IN_ARRAY));
      return static_cast<const_pointer>(PyArray_DATA(_op));
    }

    reference operator[](size_type const i) noexcept
    {
      assert(0 <= i && i < size());
      return data()[i];
    }

    const_reference operator[](size_type const i) const noexcept
    {
      assert(0 <= i && i < size());
      return data()[i];
    }

    iterator begin() noexcept
    {
      return data();
    }

    iterator end() noexcept
    {
      return data() + size();
    }

    const_iterator begin() const noexcept
    {
      return data();
    }

    const_iterator end() const noexcept
    {
      return data() + size();
    }

    static bool check(PyObject *op) noexcept
    {
      return PyArray_Check(op) &&
             PyArray_TYPE(reinterpret_cast<PyArrayObject *>(op)) == dtype &&
             PyArray_NDIM(reinterpret_cast<PyArrayObject *>(op)) == ndims &&
             PyArray_CHKFLAGS(reinterpret_cast<PyArrayObject *>(op), flags);
    }

    static py_array convert(PyObject *op) noexcept
    {
      auto const type = PyArray_DescrFromType(dtype);
      return py_array{PyArray_FromAny(op, type, 0, 0, flags, nullptr)};
    }

    friend void swap(py_array &lhs, py_array &rhs) noexcept
    {
      lhs.swap(rhs);
    }

  protected:
    explicit py_array(PyObject *op) : _op{reinterpret_cast<PyArrayObject *>(op)}
    {}

  private:
    PyArrayObject *_op = nullptr;
  };
} // namespace spectre

namespace pybind11::detail {
  template <typename T> struct type_caster<::spectre::span<T>> {
    using value_t = ::spectre::span<T>;
    using dtype_t = ::spectre::py_dtype<std::decay_t<T>>;
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

  template <typename T> struct type_caster<::spectre::py_array<T>> {
    using value_t = ::spectre::py_array<T>;

    // clang-format off
    PYBIND11_TYPE_CASTER(value_t, _("numpy.ndarray[") +
      _(::spectre::py_dtype<std::decay_t<T>>::name) + _("]"));
    // clang-format on

    bool load(handle source, bool convert)
    {
      if (!convert && !value_t::check(source.ptr()))
        return false;
      if (value = value_t::convert(source.ptr()); !value)
        PyErr_Clear();
      return bool(value);
    }

    static handle cast(value_t source, return_value_policy, handle)
    {
      return std::move(source).release();
    }
  };
} // namespace pybind11::detail
