#pragma once

#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <tuple>
#include <type_traits>
#include <utility>
#include <xs/range_zip.h>
#include <pybind11/numpy.h>

namespace xs::py {
  template <typename> struct numpy_type_impl;

  // clang-format off
  template <> struct numpy_type_impl<bool> : std::integral_constant<int, NPY_BOOL> {};
  template <> struct numpy_type_impl<int> : std::integral_constant<int, NPY_INT> {};
  template <> struct numpy_type_impl<float> : std::integral_constant<int, NPY_FLOAT> {};
  template <> struct numpy_type_impl<double> : std::integral_constant<int, NPY_DOUBLE> {};
  // clang-format on

  template <typename Type>
  constexpr inline int numpy_type = numpy_type_impl<Type>::value;
} // namespace xs::py

namespace xs::py {
  template <typename Type, int Flags> struct numpy_array {
    using size_type = std::size_t;
    using value_type = Type;

    using iterator = std::add_pointer_t<value_type>;
    using const_iterator = std::add_pointer_t<std::add_const_t<value_type>>;

    constexpr explicit numpy_array(const size_type size)
        : _handle{PyArray_New(&PyArray_Type, 1, &static_cast<npy_intp>(size),
                              numpy_type<Type>, nullptr, nullptr, 0, Flags,
                              nullptr)} {
      if (!_handle)
        throw std::bad_alloc{};
    }

    constexpr numpy_array(const numpy_array &other)
        : _handle{
              PyArray_NewLikeArray(other._handle, NPY_ANYORDER, nullptr, 1)} {
      if (!_handle || !PyArray_CopyInto(_handle, other._handle))
        throw std::bad_alloc{};
    }

    constexpr numpy_array(numpy_array &&other) noexcept
        : _handle{other._handle} {
      other._handle = nullptr;
    }

    constexpr numpy_array &operator=(numpy_array other) {
      swap(other);
    };

    constexpr void swap(numpy_array &other) noexcept {
      using std::swap;
      swap(_handle, other._handle);
    }

    constexpr bool empty() const {
      if (_handle)
        return static_cast<bool>(PyArray_SIZE(_handle));
      else
        return false;
    }

    constexpr size_type size() const {
      if (_handle)
        return static_cast<size_type>(PyArray_SIZE(_handle));
      else
        return 0;
    }

    constexpr iterator begin() {
      if (_handle)
        return static_cast<iterator>(PyArray_DATA(_handle));
      else
        return nullptr;
    }

    constexpr const_iterator begin() const {
      if (_handle)
        return static_cast<const_iterator>(PyArray_DATA(_handle));
      else
        return nullptr;
    }

    constexpr iterator end() {
      if (_handle)
        return begin() + size();
      else
        return nullptr;
    }

    constexpr const_iterator end() const {
      if (_handle)
        return begin() + size();
      else
        return nullptr;
    }

  private:
    PyArrayObject *_handle = nullptr;
  };

  template <typename Type, int Flags>
  constexpr void swap(numpy_array<Type, Flags> &lhs,
                      numpy_array<Type, Flags> &rhs) noexcept {
    lhs.swap(rhs);
  }
} // namespace xs::py
/*
namespace xs::py {
  template <typename T, typename X, typename Y> struct coo {
    using size_type = std::size_t;
    using coord_type = std::tuple<X, Y>;
    using shape_type = std::tuple<X, Y>;
    using value_type = T;

    constexpr explicit coo(shape_type shape)
        : _shape{std::move(shape)}
        , _data{}
        , _rows{}
        , _cols{} {}

    constexpr explicit coo(shape_type shape, size_type size)
        : _shape{std::move(shape)}
        , _data{size}
        , _rows{size}
        , _cols{size} {}

    constexpr coo(const coo &) = default;
    constexpr coo(coo &&) noexcept = default;

    constexpr coo &operator=(const coo &) = default;
    constexpr coo &operator=(coo &&) noexcept = default;

    constexpr void swap(coo &other) noexcept {
      using std::swap;

      swap(_shape, other._shape);
      swap(_data, other._data);
      swap(_rows, other._rows);
      swap(_cols, other._cols);
    }

    constexpr bool empty() const {
      return _data.empty();
    }

    constexpr size_type size() const {
      return _data.size();
    }

    constexpr auto &values() const {
      return _data;
    }

    constexpr auto coords() const {
      return range::zip{_rows, _cols};
    }

    constexpr auto sparse() const {
      return range::zip{coords(), values()};
    }

  private:
    shape_type _shape;
    numpy_array<T> _data;
    numpy_array<X> _rows;
    numpy_array<Y> _cols;
  };

  template <typename... Ts>
  constexpr void swap(coo<Ts...> &lhs, coo<Ts...> &rhs) noexcept {
    lhs.swap(rhs);
  }
} // namespace xs::py
*/